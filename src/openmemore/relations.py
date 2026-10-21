"""Relations between memories (SPEC.md M7.7), after LongMemory's executable edges: a link
isn't a passive label that some other code interprets later. Adding one is the state
change, done here and only here, inside the caller's transaction:

- supersedes (new -> old): the old memory is replaced; its validity closes where the
  new one starts (openmemore.repo.mark_superseded).
- derived_from (memory -> conversation): where a memory came from.
- contradicts (a <-> b): the two disagree; until resolved, neither is current truth.
- supports (memory -> conversation, or -> itself when no conversation is cited): a
  confirmation by a source, which raises confidence (see `confirmed`).

Callers hold a transaction (db.transaction) around these.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Iterable
from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel

from openmemore import repo

LinkType = Literal["supersedes", "contradicts", "supports", "derived_from"]


class Link(BaseModel):
    id: int
    type: LinkType
    from_memory: int
    to_memory: int | None = None
    to_conversation: int | None = None
    source: str | None = None
    created_at: AwareDatetime
    resolved_at: AwareDatetime | None = None
    resolution: str | None = None


def _add(conn: sqlite3.Connection, type_: LinkType, from_memory: int, now: datetime, source: str | None, *,
         to_memory: int | None = None, to_conversation: int | None = None) -> int:
    return conn.execute(
        "INSERT INTO memory_links (type, from_memory, to_memory, to_conversation, source, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (type_, from_memory, to_memory, to_conversation, source, repo.to_iso(now)),
    ).lastrowid


def supersede(conn: sqlite3.Connection, new_id: int, old_id: int, now: datetime, *,
              valid_until: datetime | None = None, source: str | None = None) -> int:
    """`new_id` replaces `old_id`, which stops being true at `valid_until` (default now)."""
    repo.mark_superseded(conn, old_id, new_id, now, valid_until=valid_until)
    return _add(conn, "supersedes", new_id, now, source, to_memory=old_id)


def derived_from(conn: sqlite3.Connection, memory_id: int, conversation_ids: Iterable[int], now: datetime,
                 source: str | None = None) -> list[int]:
    """Record the conversations a memory came from (ones that no longer exist are skipped)."""
    return [
        _add(conn, "derived_from", memory_id, now, source, to_conversation=cid)
        for cid in dict.fromkeys(conversation_ids) if repo.get_conversation(conn, cid) is not None
    ]


def contradict(conn: sqlite3.Connection, a: int, b: int, now: datetime, source: str | None = None) -> int:
    """`a` and `b` disagree. Until resolved, neither is current truth: both are left out of
    current answers and labelled contradicted in history."""
    existing = conn.execute(
        "SELECT id FROM memory_links WHERE type = 'contradicts' AND resolved_at IS NULL "
        "AND ((from_memory = ? AND to_memory = ?) OR (from_memory = ? AND to_memory = ?))", (a, b, b, a)).fetchone()
    return existing[0] if existing else _add(conn, "contradicts", a, now, source, to_memory=b)


Resolution = Literal["keep_from", "keep_to", "both"]


class ResolveError(Exception):
    """The contradiction can't be resolved as asked; the message says why."""


def resolve(conn: sqlite3.Connection, link_id: int, resolution: Resolution, now: datetime,
            source: str | None = None) -> Link:
    """Settle a contradiction: one side was right (the other is superseded by it), or
    both are true after all (nothing changes but the link)."""
    row = conn.execute("SELECT * FROM memory_links WHERE id = ? AND type = 'contradicts'", (link_id,)).fetchone()
    if row is None:
        raise ResolveError(f"conflict {link_id} not found")
    link = Link.model_validate(dict(row))
    if link.resolved_at is not None:
        raise ResolveError(f"conflict {link_id} was already resolved ({link.resolution})")
    if resolution != "both":
        keep, drop = (link.from_memory, link.to_memory) if resolution == "keep_from" else (link.to_memory, link.from_memory)
        dropped = repo.get_memory(conn, drop)
        if dropped is not None and dropped.active:
            supersede(conn, keep, drop, now, source=source)
    conn.execute("UPDATE memory_links SET resolved_at = ?, resolution = ? WHERE id = ?",
                 (repo.to_iso(now), resolution, link_id))
    return Link.model_validate(dict(conn.execute("SELECT * FROM memory_links WHERE id = ?", (link_id,)).fetchone()))


def open_contradictions(conn: sqlite3.Connection, scopes=None) -> list[Link]:
    where, params = "", []
    if scopes is not None:
        where = f" AND m.scope IN ({', '.join('?' * len(scopes))})"
        params = list(scopes)
    return [Link.model_validate(dict(r)) for r in conn.execute(
        "SELECT l.* FROM memory_links l JOIN memories m ON m.id = l.from_memory "
        f"WHERE l.type = 'contradicts' AND l.resolved_at IS NULL{where} ORDER BY l.id", params)]


DEFAULT_CONFIDENCE = 0.6
"""Confidence of a memory saved without one: stated once, by one source."""
CONFIDENCE_CAP = 0.98
"""Confirmation raises confidence towards this, never to certainty."""
SUPPORT_LIKELIHOOD_RATIO = 2.0
"""How much one independent confirmation multiplies the odds that a memory is right."""
SAME_SOURCE_WINDOW_S = 86_400
"""A source confirming the same memory again within a day counts once."""


def confirmed(confidence: float | None, reliability: float = 1.0) -> float:
    """Confidence after one independent confirmation, adding evidence in log-odds (after
    LongMemory's evidence formula): logit(c) + reliability * ln(likelihood ratio)."""
    c = min(max(confidence if confidence is not None else DEFAULT_CONFIDENCE, 0.01), CONFIDENCE_CAP)
    logit = math.log(c / (1 - c)) + reliability * math.log(SUPPORT_LIKELIHOOD_RATIO)
    return round(min(1 / (1 + math.exp(-logit)), CONFIDENCE_CAP), 4)


def support(conn: sqlite3.Connection, memory_id: int, now: datetime, *, source: str,
            conversation_id: int | None = None) -> bool:
    """Record that `source` stated `memory_id` again, citing a conversation or not, and
    raise its confidence. Returns whether it counted: a conversation the memory already
    came from or was confirmed by doesn't, nor does the same source again within a day."""
    if conversation_id is not None:
        if repo.get_conversation(conn, conversation_id) is None or conn.execute(
            "SELECT 1 FROM memory_links WHERE from_memory = ? AND to_conversation = ? "
            "AND type IN ('derived_from', 'supports')", (memory_id, conversation_id)).fetchone():
            return False
    else:
        since = repo.to_iso(datetime.fromtimestamp(now.timestamp() - SAME_SOURCE_WINDOW_S, now.tzinfo))
        memory = repo.get_memory(conn, memory_id)
        if memory is None or memory.source_client == source and memory.created_at.timestamp() > now.timestamp() - SAME_SOURCE_WINDOW_S:
            return False  # the source restating what it just saved
        if conn.execute(
            "SELECT 1 FROM memory_links WHERE from_memory = ? AND type = 'supports' AND to_conversation IS NULL "
            "AND to_memory = ? AND source = ? AND created_at > ?", (memory_id, memory_id, source, since)).fetchone():
            return False
    memory = repo.get_memory(conn, memory_id)
    if memory is None:
        return False
    if conversation_id is not None:
        _add(conn, "supports", memory_id, now, source, to_conversation=conversation_id)
    else:
        _add(conn, "supports", memory_id, now, source, to_memory=memory_id)  # a confirmation with no conversation
    conn.execute("UPDATE memories SET confidence = ? WHERE id = ?", (confirmed(memory.confidence), memory_id))
    repo.record_presented(conn, [memory_id], [], now)  # a confirmation is a presentation too
    return True


def links(conn: sqlite3.Connection, memory_id: int, type_: LinkType | None = None) -> list[Link]:
    """Every link from or to a memory, oldest first."""
    where, params = "(from_memory = ? OR to_memory = ?)", [memory_id, memory_id]
    if type_:
        where += " AND type = ?"
        params.append(type_)
    return [Link.model_validate(dict(r)) for r in conn.execute(
        f"SELECT * FROM memory_links WHERE {where} ORDER BY id", params)]


def sources(conn: sqlite3.Connection, memory_id: int) -> list[int]:
    """The conversations a memory came from."""
    return [r[0] for r in conn.execute(
        "SELECT to_conversation FROM memory_links WHERE from_memory = ? AND type = 'derived_from' ORDER BY id",
        (memory_id,))]
