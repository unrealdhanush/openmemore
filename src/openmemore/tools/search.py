"""search_context: hybrid search over memories and conversation chunks.

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

from openmemore import embed, relations, repo
from openmemore.models import CHAT_ORDER_NOTE, ChunkMatch, Kind, Memory, Profile, SearchHit, SearchResult
from openmemore.safety.wrap import wrap
from openmemore.tools import Caller

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
# ~1 unrelated item per query, 2 hits across the 8 unrelated queries.
SIMILARITY_FLOOR = 0.55
SIMILARITY_WINDOW = 0.08
OFF_TOPIC_WEIGHT = 0.5
"""Memory keyword hits that meaning search judged too far from the query share a word but
maybe not the topic ('Cursor' in a memory about another project). The memory list is short,
so such a memory would be rank 1 of its own list, level with the best chat, plus its boosts.
They're taken out of the keyword list and ranked in one list of their own, by meaning, with
this weight and no boosts: below the chats and memories that match, without hiding exact
matches for rare names and codes. Chat keyword hits keep their rank: the same rule for
chats lowered evidence recall on LongMemEval and LoCoMo (bench/RESULTS.md, 2026-10-20)."""
VECTOR_CANDIDATES_PER_RESULT = 4
"""Nearest neighbors fetched per requested result, before scope and kind filters."""

RECALL_WEIGHT = 0.25
"""Multiplies the score of chat chunks from recall turns ("what did I ask Copilot?"): a
recap matches the next such question better than the conversation it recaps, and each
recall would otherwise push the original further down. 0.25 puts a recap below anything
found on its own merits (a hit in both lists scores ≈ 0.033) without hiding it."""



MAX_CHUNK_KEYWORD_MATCHES = 8000
"""Ranking costs ~1.6ms per thousand matching chunks, so a common word in a large
history ('code' in 40k chunks) would blow the search budget while
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


def similarity_cutoff(memories: list[tuple[Memory, float]], chunks: list[ChunkMatch]) -> float | None:
    """How similar to the query an item must be to count as about it, judged against the
    best vector hit in either list (distances are cosine distances); None if there are
    none. Recall chunks don't set the bar: a recap of "the pet question I asked Copilot"
    matches that query far better than the original chat does, and would push the
    original out before ranking could put it first."""
    sims = [embed.similarity(d) for _, d in memories] + [embed.similarity(c.rank) for c in chunks if not c.recall]
    sims = sims or [embed.similarity(c.rank) for c in chunks]
    return max(SIMILARITY_FLOOR, max(sims) - SIMILARITY_WINDOW) if sims else None


def similar_enough(
    memories: list[tuple[Memory, float]], chunks: list[ChunkMatch], cutoff: float | None
) -> tuple[list[tuple[Memory, float]], list[ChunkMatch]]:
    """Drop vector hits below the cutoff."""
    if cutoff is None:
        return [], []
    return (
        [(m, d) for m, d in memories if embed.similarity(d) >= cutoff],
        [c for c in chunks if embed.similarity(c.rank) >= cutoff],
    )


ACTIVATION_DECAY = 0.5
"""ACT-R's d: how fast a presentation's influence fades (power law, not a half-life)."""
MEMORY_ACTIVATION_WEIGHT = 0.0005
"""Rank boost at full activation, for memories: a tie-breaker by recent and frequent use
and confirmation. Which version is current is decided by the timelines, not by this. Not
measured: the benchmarks have no memories and no usage."""
CHAT_ACTIVATION_WEIGHT = 0.0
"""For conversation messages: none. Tuned on the benchmarks (bench/RESULTS.md, 2026-10-19): any
age weighting lowered evidence recall on LongMemEval and LoCoMo, more so the stronger it
was, since questions often ask about older conversations."""
MIN_AGE_DAYS = 1 / 24
"""Ages are counted from an hour: a just-made item isn't infinitely active."""


def activation(created: datetime | None, presented: int, last: datetime | None, now: datetime,
               decay: float | None = None) -> float:
    """ACT-R base-level activation, ln(sum of age_j ** -d) over an item's presentations: its
    creation, then `presented` more, the last at `last`. Petrov's (2006) approximation needs
    only the first and last ages: ln(t_last^-d + n * (t_1^(1-d) - t_last^(1-d)) / ((1-d)(t_1 - t_last)))."""
    d = ACTIVATION_DECAY if decay is None else decay
    if created is None:
        return -d * math.log(365)  # no date: as if a year old
    first = max((now - created).total_seconds() / 86_400, MIN_AGE_DAYS)
    if not presented or last is None:
        return -d * math.log(first)
    recent = min(max((now - last).total_seconds() / 86_400, MIN_AGE_DAYS), first)
    if first - recent < 1e-9:
        return math.log((presented + 1) * recent ** -d)
    rest = presented * (first ** (1 - d) - recent ** (1 - d)) / ((1 - d) * (first - recent))
    return math.log(recent ** -d + rest)


def activation_boost(created: datetime | None, presented: int, last: datetime | None, now: datetime,
                     weight: float) -> float:
    """Activation as a rank boost: `weight` times its logistic (0 to 1)."""
    if not weight:
        return 0.0
    return weight / (1 + math.exp(-activation(created, presented, last, now)))


CONFIDENCE_WEIGHT = 0.002
"""Up to about this much rank boost for a well-confirmed memory: enough to order otherwise
close matches (a rank step is ~0.0003 near the top), not to outrank a better match."""


def confidence_boost(confidence: float | None) -> float:
    c = confidence if confidence is not None else relations.DEFAULT_CONFIDENCE
    return CONFIDENCE_WEIGHT * (c - relations.DEFAULT_CONFIDENCE) / (1 - relations.DEFAULT_CONFIDENCE)


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
    off_topic: dict[tuple[str, int], float] = {}
    if vector is not None:
        k = limit * VECTOR_CANDIDATES_PER_RESULT
        all_memories = repo.knn_memories(conn, vector, k=k, scopes=read, kinds=kinds, project=project,
                                         history=history, as_of=as_of)
        all_chunks = repo.knn_chunks(conn, vector, k=k, scopes=read) if include_chats else []
        cutoff = similarity_cutoff(all_memories, all_chunks)
        near_memories, near_chunks = similar_enough(all_memories, all_chunks, cutoff)
        memory_lists.append(("memory_meaning", [m for m, _ in near_memories[:limit]]))
        if include_chats:
            chunk_lists.append(("chat_meaning", later_turns_first(near_chunks[:limit])))
        off_topic = off_topic_memories(conn, vector, cutoff, memory_lists)

    lists: list[tuple[str, list[Memory | ChunkMatch]]] = [
        (label, [item for item in ranked if _key(item) not in off_topic])
        for label, ranked in [*memory_lists, *chunk_lists]
    ]
    if off_topic:
        items = {_key(item): item for _, ranked in memory_lists for item in ranked}
        lists.append(("off_topic", [items[key] for key in sorted(off_topic, key=off_topic.__getitem__, reverse=True)]))

    # Each list contributes 1 / (RRF_K + rank) to every item in it.
    fused: dict[tuple[str, int], _Fused] = {}
    for label, ranked in lists:
        for rank, item in enumerate(ranked, start=1):
            entry = fused.setdefault(_key(item), _Fused(0.0, 0 if isinstance(item, Memory) else 1, rank, item))
            entry.score += 1 / (RRF_K + rank)
            entry.best_rank = min(entry.best_rank, rank)
            entry.signals[f"{label}_rank"] = rank
    for key, entry in fused.items():
        item = entry.item
        entry.signals["fusion"] = entry.score
        if key in off_topic:
            entry.score *= OFF_TOPIC_WEIGHT
            entry.signals["off_topic_weight"] = OFF_TOPIC_WEIGHT
            continue
        memory = isinstance(item, Memory)
        entry.signals["activation"] = activation_boost(
            item.updated_at if memory else item.created_at, item.presented, item.last_presented_at, now,
            MEMORY_ACTIVATION_WEIGHT if memory else CHAT_ACTIVATION_WEIGHT)
        entry.score += entry.signals["activation"]
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


def _key(item: Memory | ChunkMatch) -> tuple[str, int]:
    return ("memory", item.id) if isinstance(item, Memory) else ("chat", item.chunk_id)


def off_topic_memories(
    conn: sqlite3.Connection, vector: bytes, cutoff: float | None, memory_lists: list[tuple[str, list[Memory]]],
) -> dict[tuple[str, int], float]:
    """{key: similarity} for memory keyword hits below the meaning cutoff, by their own
    vectors. Memories not embedded yet get the benefit of the doubt."""
    if cutoff is None:
        return {}
    found = {m.id for label, items in memory_lists if label == "memory_meaning" for m in items}
    ids = [m.id for label, items in memory_lists if label == "memory_keyword" for m in items if m.id not in found]
    return {("memory", id_): similarity for id_, distance in repo.distances(conn, "memories", vector, ids).items()
            if (similarity := embed.similarity(distance)) < cutoff}


def presented(conn: sqlite3.Connection, result: SearchResult | Profile) -> None:
    """Count what a client was shown, for activation: the MCP tools and the
    library call this. Evaluation calls search_context directly, so benchmark questions
    don't reinforce each other."""
    if isinstance(result, SearchResult):
        memories = [h.id for h in result.results if h.type == "memory"]
        chunks = [h.id for h in result.results if h.type == "chat"]
    else:
        memories = [i.id for g in (result.preferences, result.facts, result.active_projects, result.recent_decisions)
                    for i in g]
        chunks = []
    repo.record_presented(conn, memories, chunks, datetime.now(UTC))


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
