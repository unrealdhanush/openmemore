"""search_context. M1: keyword search (FTS5, BM25) over memories and conversation
chunks, merged with reciprocal rank fusion (M2 adds vector rankings to the merge)."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence

from openmcp import repo
from openmcp.models import ChunkMatch, Kind, Memory, SearchHit, SearchResult
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


def fts_query(text: str) -> str | None:
    """Free text -> an FTS5 query that ORs the meaningful words, each quoted so FTS5
    syntax in user text (AND, NEAR, *, quotes, colons) is treated literally."""
    words = _WORD.findall(text.casefold())
    terms = [w for w in words if w not in STOPWORDS] or words
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))


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
    )


def search_context(
    conn: sqlite3.Connection,
    caller: Caller,
    query: str,
    kinds: Sequence[Kind] = (),
    project: str | None = None,
    include_chats: bool = True,
    limit: int = 10,
) -> SearchResult:
    match = fts_query(query)
    if match is None:
        return SearchResult()
    read = caller.scopes.read_scopes

    # Each ranked list contributes 1 / (RRF_K + rank). Ties keep list order: memories
    # (distilled, durable) before chats, then by rank within each list.
    scored: list[tuple[float, int, int, SearchHit]] = []
    memories = repo.search_memories(conn, match, scopes=read, kinds=kinds, project=project, limit=limit)
    for rank, (m, _) in enumerate(memories, start=1):
        score = 1 / (RRF_K + rank)
        scored.append((score, 0, rank, _memory_hit(m, score)))
    # Chats have no kind or project, so filtering on either excludes them.
    if include_chats and not kinds and project is None:
        for rank, c in enumerate(repo.search_chunks(conn, match, scopes=read, limit=limit), start=1):
            score = 1 / (RRF_K + rank)
            scored.append((score, 1, rank, _chat_hit(c, score)))

    scored.sort(key=lambda s: (-s[0], s[1], s[2]))
    return SearchResult(results=[hit for *_, hit in scored[:limit]])
