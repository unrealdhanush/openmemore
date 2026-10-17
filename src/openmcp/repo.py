"""Repository: the only module that reads and writes application data with SQL.

Functions take a connection and Pydantic models, and return models (or ids).
Schema setup and migrations live in db.py.

`scopes` arguments: None means every scope (the user, via the CLI); a sequence
restricts to those scopes, and an empty sequence matches nothing.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime

from openmcp.models import (
    AuditEntry,
    AuditFilter,
    ChunkMatch,
    Conversation,
    Kind,
    Memory,
    Message,
    NewConversation,
    NewMemory,
    NewMessage,
)


def to_iso(dt: datetime) -> str:
    """ISO 8601 UTC with millisecond precision and a Z suffix, so stored timestamps sort as text."""
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _in(column: str, values: Sequence[str]) -> tuple[str, list[str]]:
    if not values:
        return "0", []
    return f"{column} IN ({', '.join('?' * len(values))})", list(values)


# --- audit_log -----------------------------------------------------------------------


def insert_audit(conn: sqlite3.Connection, entry: AuditEntry) -> int:
    cur = conn.execute(
        "INSERT INTO audit_log (ts, client, tool, summary, result_count, duration_ms) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (to_iso(entry.ts), entry.client, entry.tool, entry.summary, entry.result_count, entry.duration_ms),
    )
    return cur.lastrowid


def list_audit(conn: sqlite3.Connection, f: AuditFilter) -> list[AuditEntry]:
    """Newest first."""
    where, params = ["1"], []
    if f.client is not None:
        where.append("client = ?")
        params.append(f.client)
    if f.tool is not None:
        where.append("tool = ?")
        params.append(f.tool)
    rows = conn.execute(
        f"SELECT * FROM audit_log WHERE {' AND '.join(where)} ORDER BY id DESC LIMIT ?",
        [*params, f.limit],
    ).fetchall()
    return [AuditEntry.model_validate(dict(r)) for r in rows]


def audit_durations(conn: sqlite3.Connection) -> dict[str, list[float]]:
    """duration_ms of every logged call, grouped by tool."""
    out: dict[str, list[float]] = defaultdict(list)
    for tool, ms in conn.execute("SELECT tool, duration_ms FROM audit_log"):
        out[tool].append(ms)
    return dict(out)


# --- memories ------------------------------------------------------------------------


def _memory(row: sqlite3.Row) -> Memory:
    data = dict(row)
    data["metadata"] = json.loads(data["metadata"])
    data.pop("rank", None)
    return Memory.model_validate(data)


def insert_memory(conn: sqlite3.Connection, m: NewMemory) -> int:
    cur = conn.execute(
        "INSERT INTO memories (kind, subject, content, project, scope, source_client, confidence, "
        "content_hash, metadata, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            m.kind, m.subject, m.content, m.project, m.scope, m.source_client, m.confidence,
            m.content_hash, json.dumps(m.metadata), to_iso(m.created_at), to_iso(m.updated_at),
        ),
    )
    return cur.lastrowid


def get_memory(conn: sqlite3.Connection, memory_id: int) -> Memory | None:
    row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    return _memory(row) if row else None


def find_active_duplicate(conn: sqlite3.Connection, scope: str, content_hash: str) -> Memory | None:
    row = conn.execute(
        "SELECT * FROM memories WHERE scope = ? AND content_hash = ? AND superseded_by IS NULL LIMIT 1",
        (scope, content_hash),
    ).fetchone()
    return _memory(row) if row else None


def mark_superseded(conn: sqlite3.Connection, old_id: int, new_id: int, now: datetime) -> None:
    conn.execute(
        "UPDATE memories SET superseded_by = ?, updated_at = ? WHERE id = ?",
        (new_id, to_iso(now), old_id),
    )


def memory_chain(conn: sqlite3.Connection, memory_id: int) -> list[Memory]:
    """Every version linked to `memory_id` through superseded_by, oldest first."""
    rows = conn.execute(
        """
        WITH RECURSIVE
          newer(id) AS (
            SELECT ?
            UNION SELECT m.superseded_by FROM memories m JOIN newer ON m.id = newer.id
                  WHERE m.superseded_by IS NOT NULL
          ),
          chain(id) AS (
            SELECT id FROM newer
            UNION SELECT m.id FROM memories m JOIN chain ON m.superseded_by = chain.id
          )
        SELECT m.* FROM memories m JOIN chain ON m.id = chain.id ORDER BY m.id
        """,
        (memory_id,),
    ).fetchall()
    return [_memory(r) for r in rows]


def delete_memories(conn: sqlite3.Connection, ids: Sequence[int]) -> int:
    """Hard delete. One statement, so superseded_by references inside the set don't trip FKs."""
    if not ids:
        return 0
    return conn.execute(
        f"DELETE FROM memories WHERE id IN ({', '.join('?' * len(ids))})", list(ids)
    ).rowcount


def list_memories(
    conn: sqlite3.Connection,
    *,
    scopes: Sequence[str] | None,
    kinds: Sequence[Kind] = (),
    project: str | None = None,
    include_general: bool = False,
    updated_since: datetime | None = None,
    include_superseded: bool = False,
    limit: int = 50,
) -> list[Memory]:
    """Newest first. With `include_general`, a project filter also admits untagged memories."""
    where: list[str] = []
    params: list[object] = []
    if scopes is not None:
        clause, p = _in("scope", scopes)
        where.append(clause)
        params += p
    if kinds:
        clause, p = _in("kind", kinds)
        where.append(clause)
        params += p
    if project is not None:
        where.append("(project = ? OR project IS NULL)" if include_general else "project = ?")
        params.append(project)
    if updated_since is not None:
        where.append("updated_at >= ?")
        params.append(to_iso(updated_since))
    if not include_superseded:
        where.append("superseded_by IS NULL")
    sql = "SELECT * FROM memories"
    if where:
        sql += " WHERE " + " AND ".join(where)
    order = "(project IS NULL), " if project is not None and include_general else ""
    sql += f" ORDER BY {order}updated_at DESC, id DESC LIMIT ?"
    return [_memory(r) for r in conn.execute(sql, [*params, limit])]


def search_memories(
    conn: sqlite3.Connection,
    match: str,
    *,
    scopes: Sequence[str] | None,
    kinds: Sequence[Kind] = (),
    project: str | None = None,
    limit: int = 10,
) -> list[tuple[Memory, float]]:
    """FTS5 search over active memories. `match` is an FTS5 query string.

    Returns (memory, bm25) pairs, best first; bm25 is lower-is-better.
    Subject matches weigh twice as much as content matches.
    """
    where = ["memories_fts MATCH ?", "m.superseded_by IS NULL"]
    params: list[object] = [match]
    if scopes is not None:
        clause, p = _in("m.scope", scopes)
        where.append(clause)
        params += p
    if kinds:
        clause, p = _in("m.kind", kinds)
        where.append(clause)
        params += p
    if project is not None:
        where.append("m.project = ?")
        params.append(project)
    rows = conn.execute(
        f"""
        SELECT m.*, bm25(memories_fts, 2.0, 1.0) AS rank
        FROM memories_fts JOIN memories m ON m.id = memories_fts.rowid
        WHERE {' AND '.join(where)}
        ORDER BY rank, m.updated_at DESC
        LIMIT ?
        """,
        [*params, limit],
    ).fetchall()
    return [(_memory(r), r["rank"]) for r in rows]


# --- conversations -------------------------------------------------------------------


def _iso_or_none(dt: datetime | None) -> str | None:
    return to_iso(dt) if dt is not None else None


def _conversation(row: sqlite3.Row) -> Conversation:
    data = dict(row)
    data["metadata"] = json.loads(data["metadata"])
    return Conversation.model_validate(data)


def insert_conversation(conn: sqlite3.Connection, c: NewConversation) -> int:
    cur = conn.execute(
        "INSERT INTO conversations (provider, external_id, title, scope, metadata, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            c.provider, c.external_id, c.title, c.scope, json.dumps(c.metadata),
            _iso_or_none(c.created_at), _iso_or_none(c.updated_at),
        ),
    )
    return cur.lastrowid


def get_conversation(conn: sqlite3.Connection, conversation_id: int) -> Conversation | None:
    row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
    return _conversation(row) if row else None


def touch_conversation(conn: sqlite3.Connection, conversation_id: int, now: datetime) -> None:
    conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (to_iso(now), conversation_id))


def list_conversations(
    conn: sqlite3.Connection,
    *,
    scopes: Sequence[str] | None,
    source: str | None = None,
    conversation_id: int | None = None,
    limit: int = 50,
) -> list[Conversation]:
    """Most recently updated first. `source` matches the provider case-insensitively,
    anywhere in the name ('claude' matches 'claude-desktop' and 'claude')."""
    where: list[str] = []
    params: list[object] = []
    if scopes is not None:
        clause, p = _in("scope", scopes)
        where.append(clause)
        params += p
    if source is not None:
        where.append("instr(lower(provider), lower(?)) > 0")
        params.append(source)
    if conversation_id is not None:
        where.append("id = ?")
        params.append(conversation_id)
    sql = "SELECT * FROM conversations"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY updated_at IS NULL, updated_at DESC, id DESC LIMIT ?"
    return [_conversation(r) for r in conn.execute(sql, [*params, limit])]


def delete_conversation(conn: sqlite3.Connection, conversation_id: int) -> bool:
    """Hard delete; messages and chunks (and their index entries) go with it."""
    return conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,)).rowcount > 0


# --- messages and chunks -------------------------------------------------------------


def insert_message(conn: sqlite3.Connection, m: NewMessage) -> tuple[int, bool]:
    """Insert unless the conversation already has a message with this content_hash.
    Returns (message id, whether it was inserted)."""
    row = conn.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at, content_hash) VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT (conversation_id, content_hash) DO NOTHING RETURNING id",
        (m.conversation_id, m.role, m.content, _iso_or_none(m.created_at), m.content_hash),
    ).fetchone()
    if row:
        return row[0], True
    existing = conn.execute(
        "SELECT id FROM messages WHERE conversation_id = ? AND content_hash = ?", (m.conversation_id, m.content_hash)
    ).fetchone()
    return existing[0], False


def insert_chunks(conn: sqlite3.Connection, message_id: int, texts: Sequence[str]) -> list[int]:
    return [
        conn.execute(
            "INSERT INTO chunks (message_id, ord, text) VALUES (?, ?, ?)", (message_id, ord_, text)
        ).lastrowid
        for ord_, text in enumerate(texts)
    ]


def conversation_messages(
    conn: sqlite3.Connection, conversation_id: int, last: int | None = None
) -> list[Message]:
    """Messages in order; with `last`, only the final `last` of them."""
    if last is None:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?", (conversation_id, last)
        ).fetchall()[::-1]
    return [Message.model_validate(dict(r)) for r in rows]


def message_counts(conn: sqlite3.Connection, conversation_ids: Sequence[int]) -> dict[int, int]:
    if not conversation_ids:
        return {}
    rows = conn.execute(
        f"SELECT conversation_id, COUNT(*) FROM messages WHERE conversation_id IN "
        f"({', '.join('?' * len(conversation_ids))}) GROUP BY conversation_id",
        list(conversation_ids),
    )
    return {cid: n for cid, n in rows}


def search_chunks(
    conn: sqlite3.Connection, match: str, *, scopes: Sequence[str] | None, limit: int = 10
) -> list[ChunkMatch]:
    """FTS5 search over conversation chunks, best first. `match` is an FTS5 query string."""
    where = ["chunks_fts MATCH ?"]
    params: list[object] = [match]
    if scopes is not None:
        clause, p = _in("conv.scope", scopes)
        where.append(clause)
        params += p
    rows = conn.execute(
        f"""
        SELECT c.id AS chunk_id, c.text, msg.role, msg.created_at, conv.id AS conversation_id,
               conv.title, conv.provider, bm25(chunks_fts) AS rank
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.rowid
        JOIN messages msg ON msg.id = c.message_id
        JOIN conversations conv ON conv.id = msg.conversation_id
        WHERE {' AND '.join(where)}
        ORDER BY rank, msg.created_at DESC
        LIMIT ?
        """,
        [*params, limit],
    ).fetchall()
    return [ChunkMatch.model_validate(dict(r)) for r in rows]
