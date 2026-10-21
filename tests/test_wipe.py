"""`openmemore wipe`: bulk deletion by date or of everything, whole items only, with a
backup and the deleted text erased from the file."""

import os
import stat
from contextlib import closing
from datetime import UTC, datetime, timedelta

import pytest

from conftest import insert, plain
from openmemore import cli, db, embed, repo, wipe
from openmemore.cli import app, initialize
from openmemore.ingest import store_conversation
from openmemore.models import ParsedConversation, ParsedMessage
from typer.testing import CliRunner

NOW = datetime.now(UTC)
OLD = NOW - timedelta(days=10)
CUTOFF = NOW - timedelta(days=1)


def chat(conn, external_id, *times, text="Planning a vegetable garden."):
    cid, _ = store_conversation(conn, ParsedConversation(
        provider="vscode-copilot", external_id=external_id, messages=[
            ParsedMessage(external_id=str(i), role="user", content=f"{text} ({i})", created_at=t)
            for i, t in enumerate(times)]), "personal")
    return cid


@pytest.fixture(autouse=True)
def password(monkeypatch):
    """Stands in for sudo's password check; tests flip `ok` and read `asked`."""
    check = type("Check", (), {"ok": True, "asked": 0})()

    def ask():
        check.asked += 1
        return check.ok
    monkeypatch.setattr(cli, "_password_ok", ask)
    return check


@pytest.fixture
def store(home, fake_embedder):
    initialize(home)
    with closing(db.connect(home / "openmemore.db")) as conn:
        ids = {
            "old memory": insert(conn, "Prefers tea.", when=OLD),
            "new memory": insert(conn, "Prefers metric units.", when=NOW),
            "old chat": chat(conn, "old", OLD),
            "mixed chat": chat(conn, "mixed", OLD, NOW),  # still active: kept whole
            "new chat": chat(conn, "new", NOW),
        }
        # An old memory corrected recently keeps its history; one corrected long ago doesn't.
        first = insert(conn, "Uses Go 1.21.", when=OLD - timedelta(days=5))
        second = insert(conn, "Uses Go 1.23.", when=NOW)
        repo.mark_superseded(conn, first, second, NOW)
        stale = insert(conn, "Lives in Lisbon.", when=OLD - timedelta(days=5))
        moved = insert(conn, "Lives in Porto.", when=OLD)
        repo.mark_superseded(conn, stale, moved, OLD)
        ids |= {"corrected recently": (first, second), "corrected long ago": (stale, moved)}
        conn.execute("INSERT INTO audit_log (ts, client, tool) VALUES (?, 'cli', 'status'), (?, 'cli', 'status')",
                     (repo.to_iso(OLD), repo.to_iso(NOW)))
        embed.embed_pending(conn, fake_embedder)
        yield conn, ids


def run(*args, input=None):
    result = CliRunner().invoke(app, list(args), input=input, env={"COLUMNS": "200"})
    return result.exit_code, plain(result.output)


def ids(conn, table):
    return {r[0] for r in conn.execute(f"SELECT id FROM {table}")}


def test_a_dated_wipe_deletes_whole_items_not_touched_since(store):
    conn, s = store
    p = wipe.plan(conn, CUTOFF)
    assert p.conversation_ids == [s["old chat"]]
    assert set(p.memory_ids) == {s["old memory"], *s["corrected long ago"]}
    assert p.audit_entries == 1
    wipe.apply(conn, p)
    assert ids(conn, "conversations") == {s["mixed chat"], s["new chat"]}
    assert ids(conn, "memories") == {s["new memory"], *s["corrected recently"]}
    assert conn.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (s["mixed chat"],)).fetchone()[0] == 2


def test_everything_goes_and_nothing_is_left_behind(store):
    conn, _ = store
    wipe.apply(conn, wipe.plan(conn, None))
    for table in ("memories", "conversations", "messages", "chunks", "proposals", "audit_log", "memory_links"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table
    for table in ("memories_vec", "chunks_vec", "memories_fts", "chunks_fts"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table


def test_deleted_text_is_erased_from_the_file(store, home):
    conn, _ = store
    wipe.apply(conn, wipe.plan(conn, None))
    conn.close()
    for path in home.glob("openmemore.db*"):
        assert b"vegetable garden" not in path.read_bytes(), path.name


def test_deleted_conversations_stay_deleted(store):
    conn, s = store
    wipe.apply(conn, wipe.plan(conn, None))
    assert conn.execute("SELECT COUNT(*) FROM deleted_conversations").fetchone()[0] == 3
    assert chat(conn, "old", OLD) is None  # captured again: refused


def test_the_command_asks_backs_up_and_records_itself(store, home):
    conn, s = store
    code, out = run("wipe", "--all", input="no\n")
    assert code == 1 and "Nothing deleted" in out and ids(conn, "memories")
    code, out = run("wipe", "--all", input="wipe\n")
    assert code == 0 and "Deleted 6 memories and 3 conversations" in out
    [backup] = home.glob("openmemore.db.before-wipe-*")
    if os.name != "nt":  # Windows has no Unix file modes
        assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert str(backup) in out.replace("\n", "")
    with closing(db.connect(backup)) as copy:
        assert len(ids(copy, "memories")) == 6
    assert [r[0] for r in conn.execute("SELECT tool FROM audit_log")] == ["wipe"]


def test_the_command_needs_the_password_even_with_yes(store, password):
    conn, s = store
    password.ok = False
    code, out = run("wipe", "--all", "--yes")
    assert code == 1 and "Nothing deleted" in out and ids(conn, "memories") and password.asked == 1
    run("wipe", "--all", input="no\n")
    assert password.asked == 1  # declined before the password was asked


def test_the_password_is_asked_through_sudo_without_its_cache(monkeypatch):
    monkeypatch.undo()  # the real check, with sudo itself faked
    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda args: calls.append(args) or type("R", (), {"returncode": 1}))
    monkeypatch.setattr(cli.sys, "platform", "linux")
    assert not cli._password_ok() and calls[0][:3] == ["sudo", "-k", "-v"]
    monkeypatch.setattr(cli.sys, "platform", "win32")  # no sudo there: the typed confirmation stands
    assert cli._password_ok() and len(calls) == 1


def test_the_command_takes_a_local_date(store, home):
    conn, s = store
    day = (NOW - timedelta(days=2)).astimezone().date().isoformat()
    code, out = run("wipe", "--before", day, "--yes", "--no-backup")
    assert code == 0 and s["old chat"] not in ids(conn, "conversations") and s["new chat"] in ids(conn, "conversations")
    assert not list(home.glob("openmemore.db.before-wipe-*"))


@pytest.mark.parametrize("args", [[], ["--all", "--before", "2026-01-01"], ["--before", "last tuesday"]])
def test_the_command_needs_exactly_one_valid_choice(store, args):
    code, _ = run("wipe", *args, "--yes")
    assert code == 2


def test_nothing_to_delete(store):
    code, out = run("wipe", "--before", "2000-01-01", "--yes")
    assert code == 0 and "Nothing to delete" in out
