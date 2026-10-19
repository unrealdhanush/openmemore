"""search_context: hybrid search over memories and conversation chunks (SPEC.md 7.2).

Up to four ranked lists (keyword (FTS5, BM25) and vector (sqlite-vec, cosine) for
memories and for chunks) are merged with reciprocal rank fusion, plus a mild recency
boost. Without the embedding model, the vector lists are simply missing.
"""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from openmcp import embed, repo
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


def _memory_hit(m: Memory, score: float) -> SearchHit:
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


def search_context(
    conn: sqlite3.Connection,
    caller: Caller,
    query: str,
    kinds: Sequence[Kind] = (),
    project: str | None = None,
    include_chats: bool = True,
    limit: int = 10,
) -> SearchResult:
    read = caller.scopes.read_scopes
    words = query_words(query)
    match = fts_query(words)
    # Chats have no kind or project, so filtering on either excludes them.
    include_chats = include_chats and not kinds and project is None
    embedder = embed.get_embedder()
    vector = embedder.query(query) if embedder is not None else None

    memory_lists: list[list[Memory]] = []
    chunk_lists: list[list[ChunkMatch]] = []
    if match is not None:
        memory_lists.append(
            [m for m, _ in repo.search_memories(conn, match, scopes=read, kinds=kinds, project=project, limit=limit)]
        )
        if include_chats and (chunk_match := chunk_keyword_query(conn, words)):
            chunk_lists.append(later_turns_first(repo.search_chunks(conn, chunk_match, scopes=read, limit=limit)))
    if vector is not None:
        k = limit * VECTOR_CANDIDATES_PER_RESULT
        near_memories, near_chunks = similar_enough(
            repo.knn_memories(conn, vector, k=k, scopes=read, kinds=kinds, project=project),
            repo.knn_chunks(conn, vector, k=k, scopes=read) if include_chats else [],
        )
        memory_lists.append([m for m, _ in near_memories[:limit]])
        if include_chats:
            chunk_lists.append(later_turns_first(near_chunks[:limit]))

    # Each list contributes 1 / (RRF_K + rank) to every item in it.
    fused: dict[tuple[str, int], _Fused] = {}
    for kind_order, lists in ((0, memory_lists), (1, chunk_lists)):
        for ranked in lists:
            for rank, item in enumerate(ranked, start=1):
                key = ("memory", item.id) if isinstance(item, Memory) else ("chat", item.chunk_id)
                entry = fused.setdefault(key, _Fused(0.0, kind_order, rank, item))
                entry.score += 1 / (RRF_K + rank)
                entry.best_rank = min(entry.best_rank, rank)
    now = datetime.now(UTC)
    for entry in fused.values():
        item = entry.item
        entry.score += recency_boost(item.updated_at if isinstance(item, Memory) else item.created_at, now)
        if isinstance(item, ChunkMatch) and item.recall:
            entry.score *= RECALL_WEIGHT

    ranked = sorted(fused.values(), key=lambda e: (-e.score, e.kind_order, e.best_rank))[:limit]
    hits = [
        _memory_hit(e.item, e.score) if isinstance(e.item, Memory) else _chat_hit(e.item, e.score) for e in ranked
    ]
    has_chats = any(h.type == "chat" for h in hits)
    return SearchResult(note=CHAT_ORDER_NOTE if has_chats else None, results=hits)
