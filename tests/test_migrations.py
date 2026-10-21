import sqlite3
import threading

import pytest

from openmemore import db

EXPECTED_TABLES = {
    "schema_version",
    "conversations",
    "messages",
    "chunks",
    "memories",
    "audit_log",
    "ingest_jobs",
}


def tables(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {r[0] for r in rows}


def test_migrations_are_numbered_from_one():
    versions = [v for v, _ in db.migrations()]
    assert versions == list(range(1, len(versions) + 1))


def test_fresh_db_gets_all_tables(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    assert db.schema_version(conn) == 0
    applied = db.migrate(conn)
    assert applied == [v for v, _ in db.migrations()]
    assert EXPECTED_TABLES <= tables(conn)
    assert conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1


def test_migrate_twice_is_noop(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.migrate(conn)
    assert db.migrate(conn) == []


def test_concurrent_migrate_applies_once(tmp_path):
    """Several serve processes may start at once; only one should apply each migration."""
    path = tmp_path / "t.db"
    results, errors = [], []

    def run():
        try:
            conn = db.connect(path)
            results.append(db.migrate(conn))
            conn.close()
        except Exception as e:  # pragma: no cover - surfaced by the assert below
            errors.append(e)

    threads = [threading.Thread(target=run) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert sorted(results, key=len) == [[]] * 4 + [[v for v, _ in db.migrations()]]


def test_failed_migration_rolls_back(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "t.db")
    good = db.migrations()
    bad = (len(good) + 1, "CREATE TABLE half_done (x INTEGER);\nNOT VALID SQL;\n")
    monkeypatch.setattr(db, "migrations", lambda: good + [bad])

    with pytest.raises(sqlite3.OperationalError):
        db.migrate(conn)

    assert db.schema_version(conn) == 0
    assert "half_done" not in tables(conn)


def test_constraints(tmp_path):
    conn = db.open_db(tmp_path / "t.db")
    now = "2026-01-01T00:00:00Z"
    insert = (
        "INSERT INTO memories (kind, content, source_client, content_hash, created_at, updated_at) "
        "VALUES (?, 'x', 'test', 'h', ?, ?)"
    )
    conn.execute(insert, ("fact", now, now))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(insert, ("not-a-kind", now, now))

    conn.execute("INSERT INTO conversations (provider, external_id) VALUES ('claude', 'c1')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO conversations (provider, external_id) VALUES ('claude', 'c1')")


def test_foreign_keys_cascade(tmp_path):
    conn = db.open_db(tmp_path / "t.db")
    conv = conn.execute("INSERT INTO conversations (provider, external_id) VALUES ('claude', 'c1')").lastrowid
    msg = conn.execute(
        "INSERT INTO messages (conversation_id, role, content, content_hash) VALUES (?, 'user', 'hi', 'h')",
        (conv,),
    ).lastrowid
    conn.execute("INSERT INTO chunks (message_id, ord, text) VALUES (?, 0, 'hi')", (msg,))

    conn.execute("DELETE FROM conversations WHERE id = ?", (conv,))

    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0


def test_metadata_columns_default_and_require_valid_json(tmp_path):
    conn = db.open_db(tmp_path / "t.db")
    conv = conn.execute("INSERT INTO conversations (provider, external_id) VALUES ('claude', 'c1')").lastrowid
    assert conn.execute("SELECT metadata FROM conversations WHERE id = ?", (conv,)).fetchone()[0] == "{}"

    conn.execute("UPDATE conversations SET metadata = '{\"model\": \"x\"}' WHERE id = ?", (conv,))
    assert conn.execute("SELECT metadata ->> '$.model' FROM conversations").fetchone()[0] == "x"
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE conversations SET metadata = 'not json' WHERE id = ?", (conv,))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO memories (kind, content, source_client, content_hash, created_at, updated_at, metadata) "
            "VALUES ('fact', 'x', 't', 'h', 'now', 'now', '{oops')"
        )


def test_upgrade_from_version_1_keeps_rows(tmp_path, monkeypatch):
    """A DB created by the original M0 (migration 001 only) upgrades cleanly."""
    path = tmp_path / "t.db"
    all_migrations = db.migrations()
    monkeypatch.setattr(db, "migrations", lambda: all_migrations[:1])
    conn = db.open_db(path)
    conn.execute("INSERT INTO audit_log (ts, client, tool) VALUES ('2026-01-01T00:00:00.000Z', 'c', 'ping')")
    monkeypatch.undo()

    assert db.migrate(conn) == [v for v, _ in all_migrations[1:]]
    row = conn.execute("SELECT client, duration_ms FROM audit_log").fetchone()
    assert (row["client"], row["duration_ms"]) == ("c", 0)


def test_statement_splitter_handles_triggers():
    sql = (
        "CREATE TABLE a (x);\n"
        "CREATE TRIGGER t AFTER INSERT ON a BEGIN\n"
        "  SELECT 1;\n"
        "  SELECT 2;\n"
        "END;\n"
        "-- trailing comment\n"
    )
    stmts = list(db._statements(sql))
    assert len(stmts) == 2
    assert stmts[1].strip().endswith("END;")


def test_deleting_a_captured_conversation_marks_its_transcript_ignored(tmp_path):
    conn = db.open_db(tmp_path / "t.db")
    conn.execute("INSERT INTO conversations (id, provider, external_id) VALUES (7, 'claude-code', 's')")
    conn.execute(
        "INSERT INTO capture_files (path, source, mtime_ns, size, conversation_id, captured_at) "
        "VALUES ('/t/s.jsonl', 'claude-code', 1, 1, 7, '2026-10-17T00:00:00.000Z')"
    )
    conn.execute("DELETE FROM conversations WHERE id = 7")
    assert conn.execute("SELECT ignored FROM capture_files").fetchone()[0] == 1
