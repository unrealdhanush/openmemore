"""search_context: hybrid search over memories and conversation chunks (SPEC.md 7.2).

Up to four ranked lists (keyword (FTS5, BM25) and vector (sqlite-vec, cosine) for
memories and for chunks) are merged with reciprocal rank fusion, plus a mild recency
boost. Without the embedding model, the vector lists are simply missing.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from openmcp import embed, relations, repo
from openmcp.models import CHAT_ORDER_NOTE, ChunkMatch, Kind, Memory, SearchHit, SearchResult
from openmcp.safety.wrap import wrap
from openmcp.tools import Caller

_WORD = re.compile(r"\w+")

# Dropped from queries so natural-language questions ("what do you know about my
# preferences?") match on the words that carry meaning.
STOPWORDS = frozenset(
    """
    a about above after again all also am an and any are as at be been before being but by can
    could did do does doing for from had has have having he her here hers him his how i if in into
    is it its just know knows me more most my myself no nor not now of on or our ours out over own
    same she should so some such than that the their them then there these they this those to too
    under until up very was we were what when where which while who whom why will with would you
    your yours tell remember told said
    """.split()
)

RRF_K = 60
"""Reciprocal rank fusion constant: a result at rank r in a list scores 1 / (RRF_K + r)."""

# Vector search always returns the nearest items, related or not, so vector hits must
# clear a similarity floor and be close to the best one. Calibrated for bge-small on a
# synthetic set (20 paraphrased questions, 8 unrelated ones): 18/20 paraphrases found,
# ~1 unrelated item per query, 2 hits across the 8 unrelated queries. SPEC.md Section 14.
SIMILARITY_FLOOR = 0.55
SIMILARITY_WINDOW = 0.08
VECTOR_CANDIDATES_PER_RESULT = 4
"""Nearest neighbors fetched per requested result, before scope and kind filters."""

RECALL_WEIGHT = 0.25
"""Multiplies the score of chat chunks from recall turns ("what did I ask Copilot?"): a
recap matches the next such question better than the conversation it recaps, and each
recall would otherwise push the original further down. 0.25 puts a recap below anything
found on its own merits (a hit in both lists scores ≈ 0.033) without hiding it."""

RECENCY_WEIGHT = 0.0005
"""At most this much is added to a fused score (a top rank is worth 1/61 ≈ 0.016), so
recency breaks near-ties without overriding relevance."""
RECENCY_HALF_LIFE_DAYS = 30


MAX_CHUNK_KEYWORD_MATCHES = 8000
"""Ranking costs ~1.6ms per thousand matching chunks, so a common word in a large
history ('code' in 40k chunks) would blow the search budget (SPEC.md 4.3) while
adding almost nothing: BM25 gives such words little weight anyway."""


def query_words(text: str) -> list[str]:
    """The meaningful words of a query, in order, without repeats."""
    words = _WORD.findall(text.casefold())
    return list(dict.fromkeys([w for w in words if w not in STOPWORDS] or words))


def fts_query(text: str | Sequence[str]) -> str | None:
    """Free text (or its words) -> an FTS5 query that ORs the meaningful words, each quoted
    so FTS5 syntax in user text (AND, NEAR, *, quotes, colons) is treated literally."""
    words = query_words(text) if isinstance(text, str) else list(text)
    if not words:
        return None
    return " OR ".join(f'"{w}"' for w in words)


def chunk_keyword_query(conn: sqlite3.Connection, words: list[str]) -> str | None:
    """fts_query for chunks, leaving out the most common words once the rest already
    match MAX_CHUNK_KEYWORD_MATCHES chunks between them. None if every word is too common."""
    stems = repo.stems(conn, words)
    counts = repo.chunk_term_counts(conn, [s for s in stems if s])
    by_rarity = sorted(zip(words, stems), key=lambda ws: counts.get(ws[1], 0) if ws[1] else 0)
    kept, total = [], 0
    for word, stem in by_rarity:
        matches = counts.get(stem, 0) if stem else 0
        if total + matches > MAX_CHUNK_KEYWORD_MATCHES:
            break
        kept.append(word)
        total += matches
    return fts_query(kept)


def later_turns_first(chunks: list[ChunkMatch]) -> list[ChunkMatch]:
    """Within each conversation, put later messages ahead of earlier ones, keeping the
    positions that conversation holds in the ranking. A conversation often discusses
    options before settling on one, and the settled answer comes later. Recall chunks
    keep their own positions: a recap later in a chat isn't the settled answer, and
    swapping it into a better slot would hand its weak slot to the real match."""
    positions: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(chunks):
        if not c.recall:
            positions[c.conversation_id].append(i)
    out = list(chunks)
    for slots in positions.values():
        newest_first = sorted((chunks[i] for i in slots), key=lambda c: (c.message_id, c.chunk_id), reverse=True)
        for slot, chunk in zip(slots, newest_first):
            out[slot] = chunk
    return out


def memory_status(m: Memory, now: datetime, disputed: bool = False) -> str:
    if m.superseded_by is not None:
        return "superseded"
    if m.valid_to is not None and m.valid_to <= now:
        return "ended"
    return "contradicted" if disputed else "current"


def _memory_hit(m: Memory, score: float, now: datetime, disputed: bool = False) -> SearchHit:
    return SearchHit(
        type="memory",
        id=m.id,
        content=wrap(m.content, source=m.source_client, id=m.id),
        kind=m.kind,
        subject=m.subject,
        project=m.project,
        source=m.source_client,
        created_at=m.created_at,
        score=score,
        status=memory_status(m, now, disputed),
        replaced_by=m.superseded_by,
        valid_from=m.valid_from,
        valid_to=m.valid_to,
        confidence=m.confidence if m.confidence is not None else relations.DEFAULT_CONFIDENCE,
    )


def _chat_hit(c: ChunkMatch, score: float) -> SearchHit:
    return SearchHit(
        type="chat",
        id=c.chunk_id,
        content=wrap(c.text, source=c.provider, id=c.chunk_id),
        conversation_id=c.conversation_id,
        title=c.title,
        role=c.role,
        source=c.provider,
        created_at=c.created_at,
        score=score,
        recall=c.recall,
        status="recap" if c.recall else "chat",
    )


def similar_enough(
    memories: list[tuple[Memory, float]], chunks: list[ChunkMatch]
) -> tuple[list[tuple[Memory, float]], list[ChunkMatch]]:
    """Drop vector hits that are too far from the query, judged against the best hit in
    either list (distances are cosine distances). Recall chunks don't set the bar: a recap
    of "the pet question I asked Copilot" matches that query far better than the original
    chat does, and would push the original out before ranking could put it first."""
    sims = [embed.similarity(d) for _, d in memories] + [embed.similarity(c.rank) for c in chunks if not c.recall]
    sims = sims or [embed.similarity(c.rank) for c in chunks]
    if not sims:
        return [], []
    cutoff = max(SIMILARITY_FLOOR, max(sims) - SIMILARITY_WINDOW)
    return (
        [(m, d) for m, d in memories if embed.similarity(d) >= cutoff],
        [c for c in chunks if embed.similarity(c.rank) >= cutoff],
    )


CONFIDENCE_WEIGHT = 0.002
"""Up to about this much rank boost for a well-confirmed memory: enough to order otherwise
close matches (a rank step is ~0.0003 near the top), not to outrank a better match."""


def confidence_boost(confidence: float | None) -> float:
    c = confidence if confidence is not None else relations.DEFAULT_CONFIDENCE
    return CONFIDENCE_WEIGHT * (c - relations.DEFAULT_CONFIDENCE) / (1 - relations.DEFAULT_CONFIDENCE)


def recency_boost(when: datetime | None, now: datetime) -> float:
    """Halves every RECENCY_HALF_LIFE_DAYS. Age counts in whole hours, so items written
    close together tie and the usual tie-breaks (memories first) still apply."""
    if when is None:
        return 0.0
    age_days = max(0, int((now - when).total_seconds() // 3600)) / 24
    return RECENCY_WEIGHT * 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


@dataclass
class _Fused:
    score: float
    kind_order: int
    """0 for memories (distilled, durable), 1 for chats: memories win ties."""
    best_rank: int
    item: Memory | ChunkMatch
    signals: dict[str, float] = field(default_factory=dict)
    """What made the score, for `explain`."""


def search_context(
    conn: sqlite3.Connection,
    caller: Caller,
    query: str,
    kinds: Sequence[Kind] = (),
    project: str | None = None,
    include_chats: bool = True,
    limit: int = 10,
    now: datetime | None = None,
    mode: str = "current",
    as_of: datetime | None = None,
    explain: bool = False,
    budget_tokens: int | None = None,
) -> SearchResult:
    """`mode="current"` sees current memories only; `"history"` every version, labelled;
    `as_of` the memories true at that moment (implies history). `now` is the moment
    recency and status are measured from (default: the current time); evaluation passes
    a question's date. `explain` adds what made each result's score; `budget_tokens`
    keeps the results, in rank order, within an estimated token count."""
    history = mode == "history" or as_of is not None
    now = now or datetime.now(UTC)
    read = caller.scopes.read_scopes
    words = query_words(query)
    match = fts_query(words)
    # Chats have no kind or project, so filtering on either excludes them.
    include_chats = include_chats and not kinds and project is None
    embedder = embed.get_embedder()
    vector = embedder.query(query) if embedder is not None else None

    memory_lists: list[tuple[str, list[Memory]]] = []
    chunk_lists: list[tuple[str, list[ChunkMatch]]] = []
    if match is not None:
        memory_lists.append(("memory_keyword", [
            m for m, _ in repo.search_memories(conn, match, scopes=read, kinds=kinds, project=project, limit=limit,
                                               history=history, as_of=as_of)
        ]))
        if include_chats and (chunk_match := chunk_keyword_query(conn, words)):
            chunk_lists.append(("chat_keyword", later_turns_first(
                repo.search_chunks(conn, chunk_match, scopes=read, limit=limit))))
    if vector is not None:
        k = limit * VECTOR_CANDIDATES_PER_RESULT
        near_memories, near_chunks = similar_enough(
            repo.knn_memories(conn, vector, k=k, scopes=read, kinds=kinds, project=project, history=history,
                              as_of=as_of),
            repo.knn_chunks(conn, vector, k=k, scopes=read) if include_chats else [],
        )
        memory_lists.append(("memory_meaning", [m for m, _ in near_memories[:limit]]))
        if include_chats:
            chunk_lists.append(("chat_meaning", later_turns_first(near_chunks[:limit])))

    # Each list contributes 1 / (RRF_K + rank) to every item in it.
    fused: dict[tuple[str, int], _Fused] = {}
    for kind_order, lists in ((0, memory_lists), (1, chunk_lists)):
        for label, ranked in lists:
            for rank, item in enumerate(ranked, start=1):
                key = ("memory", item.id) if isinstance(item, Memory) else ("chat", item.chunk_id)
                entry = fused.setdefault(key, _Fused(0.0, kind_order, rank, item))
                entry.score += 1 / (RRF_K + rank)
                entry.best_rank = min(entry.best_rank, rank)
                entry.signals[f"{label}_rank"] = rank
    for entry in fused.values():
        item = entry.item
        entry.signals["fusion"] = entry.score
        entry.signals["recency"] = recency_boost(item.updated_at if isinstance(item, Memory) else item.created_at, now)
        entry.score += entry.signals["recency"]
        if isinstance(item, Memory):
            entry.signals["confidence"] = confidence_boost(item.confidence)
            entry.score += entry.signals["confidence"]
        if isinstance(item, ChunkMatch) and item.recall:
            entry.signals["recap_weight"] = RECALL_WEIGHT
            entry.score *= RECALL_WEIGHT

    ranked = sorted(fused.values(), key=lambda e: (-e.score, e.kind_order, e.best_rank))[:limit]
    disputed = repo.contradicted(conn, [e.item.id for e in ranked if isinstance(e.item, Memory)])
    hits = [
        _memory_hit(e.item, e.score, now, e.item.id in disputed) if isinstance(e.item, Memory)
        else _chat_hit(e.item, e.score) for e in ranked
    ]
    if explain:
        for hit, entry in zip(hits, ranked):
            hit.explain = {k: round(v, 6) if isinstance(v, float) else v for k, v in entry.signals.items()}
    dropped, tokens = 0, None
    if budget_tokens is not None:
        hits, tokens, dropped = within_budget(hits, budget_tokens)
    has_chats = any(h.type == "chat" for h in hits)
    return SearchResult(
        note=CHAT_ORDER_NOTE if has_chats else None, results=hits, tokens_used=tokens, dropped=dropped,
        trace={
            "mode": "history" if history else "current", "as_of": as_of.isoformat() if as_of else None,
            "words": words, "meaning_search": vector is not None,
            "candidates": {label: len(items) for label, items in [*memory_lists, *chunk_lists]},
        } if explain else None,
    )


def estimate_tokens(text: str) -> int:
    """Roughly how many tokens a model will count: about four characters each."""
    return max(1, math.ceil(len(text) / 4))


def within_budget(hits: list[SearchHit], budget: int) -> tuple[list[SearchHit], int, int]:
    """The hits, in rank order, that fit `budget` estimated tokens (one that doesn't fit is
    skipped, and smaller ones after it may still fit): (kept, tokens used, dropped)."""
    kept, used = [], 0
    for hit in hits:
        cost = estimate_tokens(hit.content)
        if used + cost <= budget:
            kept.append(hit)
            used += cost
    return kept, used, len(hits) - len(kept)
