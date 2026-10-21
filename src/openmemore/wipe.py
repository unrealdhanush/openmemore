"""Bulk deletion (`openmemore wipe`): everything not touched since a date, or everything.

Whole items only. A conversation goes when its last activity is before the date, and a
memory goes with all its versions when its current version is. Splitting a conversation
still being captured would bring its older messages back the next time its transcript
changed. Deleted conversations stay deleted: the existing tombstones (and the ones the
deletes add) keep capture and import from bringing them back.

Afterwards the database is vacuumed with secure_delete on, so deleted text doesn't
linger in free pages or the write-ahead log. Config, the embedding model and the
daemon's consent are untouched.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from openmemore import db, repo

_CONVERSATION_DATE = (
    "COALESCE(c.updated_at, c.created_at, (SELECT MAX(m.created_at) FROM messages m WHERE m.conversation_id = c.id))"
)


@dataclass
class WipePlan:
    before: datetime | None
    """None: everything."""
    memory_ids: list[int] = field(default_factory=list)
    conversation_ids: list[int] = field(default_factory=list)
    messages: int = 0
    proposals: int = 0
    audit_entries: int = 0
    consolidation_runs: int = 0
    import_jobs: int = 0

    @property
    def empty(self) -> bool:
        return not (self.memory_ids or self.conversation_ids or self.proposals or self.audit_entries
                    or self.consolidation_runs or self.import_jobs)

    def older(self, column: str) -> tuple[str, tuple[str, ...]]:
        """An SQL condition for `column` being before the cutoff (always true without
        one), and its parameters. A NULL date never matches a dated wipe: its age is unknown."""
        if self.before is None:
            return "1", ()
        return f"{column} < ?", (repo.to_iso(self.before),)


def _deletes(p: WipePlan) -> list[tuple[str, tuple[str, ...]]]:
    """The statements for everything but memories and conversations, which go by id."""
    proposals, a1 = p.older("created_at")
    audit, a2 = p.older("ts")
    runs, a3 = p.older("started_at")
    jobs, a4 = p.older("COALESCE(started_at, finished_at)")
    return [
        (f"FROM proposals WHERE {proposals}", a1),
        (f"FROM audit_log WHERE {audit}", a2),
        (f"FROM consolidation_runs WHERE finished_at IS NOT NULL AND {runs}", a3),
        (f"FROM ingest_jobs WHERE finished_at IS NOT NULL AND {jobs}", a4),  # running imports are left alone
    ]


def plan(conn: sqlite3.Connection, before: datetime | None) -> WipePlan:
    """What a wipe would delete, without deleting anything."""
    p = WipePlan(before)
    if before is None:
        p.memory_ids = [r[0] for r in conn.execute("SELECT id FROM memories ORDER BY id")]
        p.conversation_ids = [r[0] for r in conn.execute("SELECT id FROM conversations ORDER BY id")]
    else:
        # A chain goes when its current version is old; its earlier versions are older still.
        condition, args = p.older("updated_at")
        heads = [r[0] for r in conn.execute(f"SELECT id FROM memories WHERE superseded_by IS NULL AND {condition}", args)]
        p.memory_ids = sorted({m.id for head in heads for m in repo.memory_chain(conn, head)})
        condition, args = p.older(_CONVERSATION_DATE)
        p.conversation_ids = [r[0] for r in conn.execute(f"SELECT c.id FROM conversations c WHERE {condition} "
                                                         "ORDER BY c.id", args)]
    if p.conversation_ids:
        marks = ", ".join("?" * len(p.conversation_ids))
        p.messages = conn.execute(f"SELECT COUNT(*) FROM messages WHERE conversation_id IN ({marks})",
                                  p.conversation_ids).fetchone()[0]
    p.proposals, p.audit_entries, p.consolidation_runs, p.import_jobs = (
        conn.execute(f"SELECT COUNT(*) {sql}", args).fetchone()[0] for sql, args in _deletes(p)
    )
    return p


def backup(conn: sqlite3.Connection, path: Path) -> Path:
    """A consistent copy of the database, readable by the user only."""
    target = sqlite3.connect(path)
    try:
        conn.backup(target)
    finally:
        target.close()
    path.chmod(0o600)
    return path


def apply(conn: sqlite3.Connection, p: WipePlan) -> None:
    """Delete what `p` lists, then vacuum so the deleted text is gone from the file."""
    conn.execute("PRAGMA secure_delete = ON")
    with db.transaction(conn):
        for sql, args in _deletes(p):
            conn.execute(f"DELETE {sql}", args)
        repo.delete_memories(conn, p.memory_ids)
        for conversation_id in p.conversation_ids:
            repo.delete_conversation(conn, conversation_id)
    for table in ("memories_fts", "chunks_fts"):
        conn.execute(f"INSERT INTO {table} ({table}) VALUES ('optimize')")  # merge away deleted index entries
    conn.execute("VACUUM")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
