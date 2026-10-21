"""SQLite connection setup and schema migrations.

Several `openmemore serve` processes (one per MCP client) can share one database, so
connections use WAL mode and a busy timeout, and run in autocommit mode: callers
open short explicit transactions with `transaction()`.
"""

from __future__ import annotations

import functools
import importlib.util
import re
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path

BUSY_TIMEOUT_MS = 5000
MMAP_BYTES = 1 << 30

_MIGRATION_NAME = re.compile(r"^(\d+)_.*\.sql$")


@functools.cache
def _vec_extension() -> str:
    """sqlite-vec's loadable extension (vec0.dylib/.so/.dll; SQLite adds the suffix),
    found without importing the sqlite_vec package, whose __init__ pulls in numpy:
    ~35ms of server startup (SPEC.md Section 4.3)."""
    spec = importlib.util.find_spec("sqlite_vec")
    if spec is None or spec.origin is None:
        raise RuntimeError("sqlite-vec is not installed")
    return str(Path(spec.origin).parent / "vec0")


def _use_wal(conn: sqlite3.Connection) -> None:
    """Switch to WAL mode (persistent, so usually already done). The switch fails with
    'database is locked' without waiting on busy_timeout when another connection holds
    the file, as when several servers open a new database at once, so retry it."""
    deadline = time.monotonic() + BUSY_TIMEOUT_MS / 1000
    while True:
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            return
        except sqlite3.OperationalError as e:
            if "locked" not in str(e) or time.monotonic() > deadline:
                raise
            time.sleep(0.01)


def connect(path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None, timeout=BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    # sqlite-vec's vec0 tables (migration 006) and their triggers need it on every connection.
    conn.enable_load_extension(True)
    conn.load_extension(_vec_extension())
    conn.enable_load_extension(False)
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    _use_wal(conn)
    # Each tool call opens its own connection, which starts with an empty page cache;
    # memory-mapping lets connections share the OS's copy of the file instead of
    # re-reading it (vector search over 100k chunks: ~66ms -> ~59ms, fewer outliers).
    conn.execute(f"PRAGMA mmap_size = {MMAP_BYTES}")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE ... COMMIT, rolling back on error.

    IMMEDIATE takes the write lock up front, so concurrent writers wait on
    busy_timeout instead of failing when a read transaction tries to upgrade.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def migrations() -> list[tuple[int, str]]:
    """All bundled migrations as (version, sql), sorted by version."""
    found = []
    for entry in resources.files("openmemore.migrations").iterdir():
        match = _MIGRATION_NAME.match(entry.name)
        if match:
            found.append((int(match.group(1)), entry.read_text(encoding="utf-8")))
    found.sort()
    versions = [v for v, _ in found]
    if len(set(versions)) != len(versions):
        raise RuntimeError(f"duplicate migration versions: {versions}")
    return found


def schema_version(conn: sqlite3.Connection) -> int:
    has_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_version'"
    ).fetchone()
    if not has_table:
        return 0
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return row[0] or 0


def _statements(sql: str) -> Iterator[str]:
    """Split a script into statements (trigger bodies included) without executescript,
    which would commit our open transaction."""
    buf = ""
    for line in sql.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            if buf.strip():
                yield buf
            buf = ""
    if buf.strip() and not buf.strip().startswith("--"):
        raise ValueError(f"incomplete SQL statement in migration: {buf.strip()[:80]!r}")


def migrate(conn: sqlite3.Connection) -> list[int]:
    """Apply pending migrations in a single transaction. Returns the versions applied.

    The version is read inside the write transaction, so two processes starting
    at once can't both apply the same migration.
    """
    applied = []
    with transaction(conn):
        current = schema_version(conn)
        for version, sql in migrations():
            if version <= current:
                continue
            for stmt in _statements(sql):
                conn.execute(stmt)
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            applied.append(version)
    return applied


def open_db(path: Path | str) -> sqlite3.Connection:
    """Connect and bring the schema up to date."""
    conn = connect(path)
    migrate(conn)
    return conn
