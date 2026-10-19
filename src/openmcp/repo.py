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
    CaptureFile,
    ChunkMatch,
    Conversation,
    IngestJob,
    Kind,
    Memory,
    Message,
    NewConversation,
    NewMemory,
    NewMessage,
    VectorTable,
)


def to_iso(dt: datetime) -> str:
    """ISO 8601 UTC with millisecond precision and a Z suffix, so stored timestamps sort as text."""
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _in(column: str, values: Sequence[object]) -> tuple[str, list[object]]:
    if not values:
        return "0", []
    return f"{column} IN ({', '.join('?' * len(values))})", list(values)


# --- audit_log -----------------------------------------------------------------------


def insert_audit(conn: sqlite3.Connection, entry: AuditEntry) -> int:
    cur = conn.execute(
        "INSERT INTO audit_log (ts, client, tool, summary, result_count, duration_ms, app) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            to_iso(entry.ts), entry.client, entry.tool, entry.summary, entry.result_count, entry.duration_ms,
            entry.app,
        ),
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


def apps_for(conn: sqlite3.Connection, client: str) -> list[str]:
    """Every application recorded calling as `client`, in order of first appearance."""
    rows = conn.execute(
        "SELECT app FROM audit_log WHERE client = ? AND app IS NOT NULL GROUP BY app ORDER BY MIN(id)", (client,)
    )
    return [r[0] for r in rows]


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


def knn_memories(
    conn: sqlite3.Connection,
    query: bytes,
    *,
    k: int,
    scopes: Sequence[str] | None,
    kinds: Sequence[Kind] = (),
    project: str | None = None,
) -> list[tuple[Memory, float]]:
    """The `k` memories nearest to `query` (an embedding), then filtered like
    search_memories. Returns (memory, cosine distance) pairs, nearest first."""
    where = ["m.superseded_by IS NULL"]
    params: list[object] = [query, k]
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
        WITH knn AS (SELECT rowid, distance FROM memories_vec WHERE embedding MATCH ? AND k = ?)
        SELECT m.*, knn.distance AS rank FROM knn JOIN memories m ON m.id = knn.rowid
        WHERE {' AND '.join(where)}
        ORDER BY knn.distance
        """,
        params,
    ).fetchall()
    return [(_memory(r), r["rank"]) for r in rows]


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


def upsert_conversation(conn: sqlite3.Connection, c: NewConversation) -> int:
    """Insert, or update the conversation with the same provider and external_id. On update
    the scope is kept (the user may have moved it), a None title keeps the stored one,
    metadata is merged, and timestamps only widen."""
    row = conn.execute(
        """
        INSERT INTO conversations (provider, external_id, title, scope, metadata, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (provider, external_id) DO UPDATE SET
          title = COALESCE(excluded.title, conversations.title),
          metadata = json_patch(conversations.metadata, excluded.metadata),
          created_at = COALESCE(conversations.created_at, excluded.created_at),
          updated_at = CASE WHEN excluded.updated_at > COALESCE(conversations.updated_at, '')
                            THEN excluded.updated_at ELSE conversations.updated_at END
        RETURNING id
        """,
        (
            c.provider, c.external_id, c.title, c.scope, json.dumps(c.metadata),
            _iso_or_none(c.created_at), _iso_or_none(c.updated_at),
        ),
    ).fetchone()
    return row[0]


def was_deleted(conn: sqlite3.Connection, provider: str, external_id: str) -> bool:
    """Whether the user deleted this conversation (migration 009 keeps a record)."""
    return (
        conn.execute(
            "SELECT 1 FROM deleted_conversations WHERE provider = ? AND external_id = ?", (provider, external_id)
        ).fetchone()
        is not None
    )


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
        "INSERT INTO messages (conversation_id, role, content, created_at, content_hash, external_id, recall) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (conversation_id, content_hash) DO NOTHING RETURNING id",
        (
            m.conversation_id, m.role, m.content, _iso_or_none(m.created_at), m.content_hash, m.external_id,
            int(m.recall),
        ),
    ).fetchone()
    if row:
        return row[0], True
    existing = conn.execute(
        "SELECT id FROM messages WHERE conversation_id = ? AND content_hash = ?", (m.conversation_id, m.content_hash)
    ).fetchone()
    return existing[0], False


def upsert_message(conn: sqlite3.Connection, m: NewMessage) -> tuple[int, bool]:
    """Insert or update the message with this external_id in the conversation.
    Returns (message id, whether its content changed, i.e. its chunks need rebuilding)."""
    if m.external_id is None:
        raise ValueError("upsert_message needs an external_id")
    row = conn.execute(
        "SELECT id, content, recall FROM messages WHERE conversation_id = ? AND external_id = ?",
        (m.conversation_id, m.external_id),
    ).fetchone()
    if row is None:
        return insert_message(conn, m)
    if bool(row["recall"]) != m.recall:  # a turn still being written may call a recall tool later
        conn.execute("UPDATE messages SET recall = ? WHERE id = ?", (int(m.recall), row["id"]))
    if row["content"] == m.content:
        return row["id"], False
    # OR IGNORE: if the new content duplicates another message in the conversation, keep the old text.
    changed = conn.execute(
        "UPDATE OR IGNORE messages SET content = ?, content_hash = ?, created_at = ? WHERE id = ?",
        (m.content, m.content_hash, _iso_or_none(m.created_at), row["id"]),
    ).rowcount
    return row["id"], changed > 0


def replace_chunks(conn: sqlite3.Connection, message_id: int, texts: Sequence[str]) -> list[int]:
    conn.execute("DELETE FROM chunks WHERE message_id = ?", (message_id,))
    return insert_chunks(conn, message_id, texts)


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


def stems(conn: sqlite3.Connection, words: Sequence[str]) -> list[str | None]:
    """Each word as the chunk index stores it (same tokenizer, so 'deploys' -> 'deploi'),
    or None if it has no indexable token. Uses a scratch FTS table in the connection's
    temp schema."""
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS temp.query_words USING fts5(word, tokenize = 'porter unicode61 remove_diacritics 2')"
    )
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.query_stems USING fts5vocab(temp, query_words, 'instance')")
    conn.execute("DELETE FROM temp.query_words")
    conn.executemany("INSERT INTO temp.query_words (rowid, word) VALUES (?, ?)", list(enumerate(words, start=1)))
    found = {doc: term for term, doc in conn.execute("SELECT term, doc FROM temp.query_stems")}
    return [found.get(i) for i in range(1, len(words) + 1)]


def chunk_term_counts(conn: sqlite3.Connection, terms: Sequence[str]) -> dict[str, int]:
    """How many chunks each indexed term appears in (terms not in the index are left out)."""
    if not terms:
        return {}
    clause, params = _in("term", list(dict.fromkeys(terms)))
    return dict(conn.execute(f"SELECT term, doc FROM chunks_fts_terms WHERE {clause}", params).fetchall())


FTS_CANDIDATES_PER_RESULT = 4
"""Keyword matches fetched per requested chunk, before the scope filter."""


def search_chunks(
    conn: sqlite3.Connection, match: str, *, scopes: Sequence[str] | None, limit: int = 10
) -> list[ChunkMatch]:
    """FTS5 search over conversation chunks, best first. `match` is an FTS5 query string.

    The best-ranked matches are taken from the index first and joined afterwards:
    joining every match before ranking took ~200ms on 100k chunks when a query word
    was common. Out-of-scope matches can crowd the candidates out, so this may return
    fewer than `limit`.
    """
    where = ["1"]
    params: list[object] = [match, limit * FTS_CANDIDATES_PER_RESULT]
    if scopes is not None:
        clause, p = _in("conv.scope", scopes)
        where.append(clause)
        params += p
    rows = conn.execute(
        f"""
        WITH best AS (
          SELECT rowid, rank FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?
        )
        SELECT c.id AS chunk_id, c.message_id, c.text, msg.role, msg.recall, msg.created_at,
               conv.id AS conversation_id, conv.title, conv.provider, best.rank
        FROM best
        JOIN chunks c ON c.id = best.rowid
        JOIN messages msg ON msg.id = c.message_id
        JOIN conversations conv ON conv.id = msg.conversation_id
        WHERE {' AND '.join(where)}
        ORDER BY best.rank, msg.created_at DESC
        LIMIT ?
        """,
        [*params, limit],
    ).fetchall()
    return [ChunkMatch.model_validate(dict(r)) for r in rows]


def knn_chunks(conn: sqlite3.Connection, query: bytes, *, k: int, scopes: Sequence[str] | None) -> list[ChunkMatch]:
    """The `k` chunks nearest to `query` (an embedding), restricted to `scopes`, nearest
    first; `rank` is the cosine distance."""
    where = ["1"]
    params: list[object] = [query, k]
    if scopes is not None:
        clause, p = _in("conv.scope", scopes)
        where.append(clause)
        params += p
    rows = conn.execute(
        f"""
        WITH knn AS (SELECT rowid, distance FROM chunks_vec WHERE embedding MATCH ? AND k = ?)
        SELECT c.id AS chunk_id, c.message_id, c.text, msg.role, msg.recall, msg.created_at,
               conv.id AS conversation_id, conv.title, conv.provider, knn.distance AS rank
        FROM knn
        JOIN chunks c ON c.id = knn.rowid
        JOIN messages msg ON msg.id = c.message_id
        JOIN conversations conv ON conv.id = msg.conversation_id
        WHERE {' AND '.join(where)}
        ORDER BY knn.distance
        """,
        params,
    ).fetchall()
    return [ChunkMatch.model_validate(dict(r)) for r in rows]


# --- embeddings ----------------------------------------------------------------------


def pending_embeddings(
    conn: sqlite3.Connection, table: VectorTable, limit: int
) -> list[tuple[int, str | None, str]]:
    """Rows without a vector yet, newest first, as (id, subject, text); subject is None for chunks."""
    if table == "memories":
        sql = (
            "SELECT id, subject, content FROM memories WHERE embedded = 0 AND superseded_by IS NULL "
            "ORDER BY id DESC LIMIT ?"
        )
    else:
        sql = "SELECT id, NULL, text FROM chunks WHERE embedded = 0 ORDER BY id DESC LIMIT ?"
    return [(r[0], r[1], r[2]) for r in conn.execute(sql, (limit,))]


def conversation_ids(conn: sqlite3.Connection, provider: str, external_ids: Sequence[str]) -> list[int]:
    """Stored ids for these external ids, in the order given (missing ones left out)."""
    if not external_ids:
        return []
    clause, params = _in("external_id", list(external_ids))
    found = dict(conn.execute(f"SELECT external_id, id FROM conversations WHERE provider = ? AND {clause}", [provider, *params]))
    return [found[e] for e in external_ids if e in found]


def pending_chunks_in(conn: sqlite3.Connection, conversation_ids: Sequence[int]) -> list[tuple[int, str]]:
    """Chunks without a vector in these conversations, as (chunk id, text)."""
    if not conversation_ids:
        return []
    clause, params = _in("msg.conversation_id", list(conversation_ids))
    return [
        (r[0], r[1])
        for r in conn.execute(
            f"SELECT c.id, c.text FROM chunks c JOIN messages msg ON msg.id = c.message_id "
            f"WHERE c.embedded = 0 AND {clause} ORDER BY c.id",
            params,
        )
    ]


def count_pending_embeddings(conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT (SELECT COUNT(*) FROM memories WHERE embedded = 0 AND superseded_by IS NULL)"
        " + (SELECT COUNT(*) FROM chunks WHERE embedded = 0)"
    ).fetchone()[0]


def store_vectors(conn: sqlite3.Connection, table: VectorTable, vectors: Sequence[tuple[int, bytes]]) -> int:
    """Store embeddings for rows that still exist (and, for memories, are still active),
    replacing any earlier vector: another server may have embedded the same row
    meanwhile. Marks them embedded. Returns how many were stored."""
    if not vectors:
        return 0
    ids = [id_ for id_, _ in vectors]
    clause, params = _in("id", ids)
    live_filter = " AND superseded_by IS NULL" if table == "memories" else ""
    live = {r[0] for r in conn.execute(f"SELECT id FROM {table} WHERE {clause}{live_filter}", params)}
    stored = [(id_, v) for id_, v in vectors if id_ in live]
    conn.executemany(f"DELETE FROM {table}_vec WHERE rowid = ?", [(id_,) for id_, _ in stored])
    conn.executemany(f"INSERT INTO {table}_vec (rowid, embedding) VALUES (?, ?)", stored)
    conn.executemany(f"UPDATE {table} SET embedded = 1 WHERE id = ?", [(id_,) for id_, _ in stored])
    return len(stored)


# --- capture_files -------------------------------------------------------------------


def get_capture_file(conn: sqlite3.Connection, path: str) -> CaptureFile | None:
    row = conn.execute("SELECT * FROM capture_files WHERE path = ?", (path,)).fetchone()
    return CaptureFile.model_validate(dict(row)) if row else None


def save_capture_file(conn: sqlite3.Connection, f: CaptureFile) -> None:
    conn.execute(
        """
        INSERT INTO capture_files (path, source, mtime_ns, size, resume_offset, conversation_id, ignored, captured_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (path) DO UPDATE SET
          source = excluded.source, mtime_ns = excluded.mtime_ns, size = excluded.size,
          resume_offset = excluded.resume_offset, conversation_id = excluded.conversation_id,
          ignored = excluded.ignored, captured_at = excluded.captured_at
        """,
        (
            f.path, f.source, f.mtime_ns, f.size, f.resume_offset, f.conversation_id, int(f.ignored),
            to_iso(f.captured_at),
        ),
    )


# --- ingest_jobs ---------------------------------------------------------------------


def _job(row: sqlite3.Row) -> IngestJob:
    return IngestJob.model_validate(dict(row))


def job_by_hash(conn: sqlite3.Connection, file_hash: str) -> IngestJob | None:
    row = conn.execute("SELECT * FROM ingest_jobs WHERE file_hash = ?", (file_hash,)).fetchone()
    return _job(row) if row else None


def insert_job(conn: sqlite3.Connection, job: IngestJob) -> int:
    return conn.execute(
        "INSERT INTO ingest_jobs (provider, file_path, file_hash, status, total, processed, cursor, started_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            job.provider, job.file_path, job.file_hash, job.status, job.total, job.processed, job.cursor,
            _iso_or_none(job.started_at),
        ),
    ).lastrowid


def start_job(conn: sqlite3.Connection, job_id: int, *, file_path: str, total: int, now: datetime) -> None:
    """Mark a job running (again): a resumed job keeps its cursor and its first start time."""
    conn.execute(
        "UPDATE ingest_jobs SET status = 'running', file_path = ?, total = ?, error = NULL, "
        "started_at = COALESCE(started_at, ?), finished_at = NULL WHERE id = ?",
        (file_path, total, to_iso(now), job_id),
    )


def advance_job(conn: sqlite3.Connection, job_id: int, processed: int) -> None:
    """Record progress; called in the same transaction as the batch it describes."""
    conn.execute("UPDATE ingest_jobs SET processed = ?, cursor = ? WHERE id = ?", (processed, str(processed), job_id))


def finish_job(conn: sqlite3.Connection, job_id: int, status: str, now: datetime, error: str | None = None) -> None:
    conn.execute(
        "UPDATE ingest_jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
        (status, error, to_iso(now), job_id),
    )


def list_jobs(conn: sqlite3.Connection, limit: int = 20) -> list[IngestJob]:
    """Most recently started first."""
    rows = conn.execute("SELECT * FROM ingest_jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_job(r) for r in rows]


def store_counts(conn: sqlite3.Connection) -> dict[str, object]:
    """What's stored, for `openmcp status`."""
    by_provider = dict(conn.execute("SELECT provider, COUNT(*) FROM conversations GROUP BY provider ORDER BY provider"))
    return {
        "memories": conn.execute("SELECT COUNT(*) FROM memories WHERE superseded_by IS NULL").fetchone()[0],
        "conversations": by_provider,
        "messages": conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
        "pending_embeddings": count_pending_embeddings(conn),
    }
