"""SPEC.md M4: `openmcp export` dumps every memory and conversation, and `openmcp import`
reads it back: into an empty store, everything current returns with its scopes; into the
same store, nothing is added."""

import json
import os
import stat
from contextlib import closing
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import caller, plain
from openmcp import db, repo
from openmcp.cli import app, initialize
from openmcp.export import export_document
from openmcp.ingest import store_conversation
from openmcp.ingest.jobs import import_export
from openmcp.models import MemoryInput, ParsedConversation, ParsedMessage
from openmcp.tools.conversations import record_exchange
from openmcp.tools.memories import save_memories, update_memory

ME = caller("claude-ai", read=["personal", "work"], write="personal")
AT_WORK = caller("cursor-vscode", read=["work"], write="work")


def fill(conn):
    """A store with a bit of everything: memories with history, in two scopes; a recorded
    chat; a captured chat with transcript ids and a recall turn."""
    [pref] = save_memories(conn, ME, [MemoryInput(kind="preference", content="Prefers tea.")]).saved
    update_memory(conn, ME, pref, "Prefers green tea.")
    save_memories(conn, AT_WORK, [MemoryInput(kind="fact", content="Uses Go at work.", project="Billing")])
    record_exchange(conn, ME, "How do I brew green tea?", "80°C water, two minutes.")
    store_conversation(conn, ParsedConversation(
        provider="vscode-copilot", external_id="vs-1", title="Billing service", metadata={"capture": "transcript"},
        messages=[
            ParsedMessage(external_id="r1", role="user", content="Where's the invoice code?"),
            ParsedMessage(external_id="r1a", role="assistant", content="In billing/invoice.go."),
            ParsedMessage(external_id="r2", role="user", content="What did I ask Claude?", recall=True),
        ],
    ), "work")


def snapshot(conn):
    memories = sorted((m.kind, m.content, m.scope, m.source_client, m.project)
                      for m in repo.list_memories(conn, scopes=None, limit=100))
    conversations = sorted(
        (c.provider, c.scope, c.title, tuple((m.role, m.content, m.recall) for m in repo.conversation_messages(conn, c.id)))
        for c in repo.list_conversations(conn, scopes=None, limit=100)
    )
    return memories, conversations


@pytest.fixture
def source(tmp_path):
    with closing(db.open_db(tmp_path / "source.db")) as conn:
        fill(conn)
        yield conn


def write(doc, path):
    path.write_text(json.dumps(doc))
    return path


def test_export_holds_everything_with_history_and_scopes(source):
    doc = export_document(source)
    assert (doc["format"], doc["version"]) == ("openmcp.export", 1)
    [old, new, work] = doc["memories"]
    assert (old["content"], old["superseded_by"]) == ("Prefers tea.", new["id"])
    assert (work["scope"], work["project"], work["source"]) == ("work", "Billing", "cursor-vscode")
    captured = next(c for c in doc["conversations"] if c["provider"] == "vscode-copilot")
    assert (captured["id"], captured["scope"]) == ("vs-1", "work")
    assert [m["id"] for m in captured["messages"]] == ["r1", "r1a", "r2"]
    assert captured["messages"][2]["recall"] is True
    recorded = next(c for c in doc["conversations"] if c["provider"] == "claude-ai")
    assert all(m["id"].startswith("sha256:") for m in recorded["messages"])


def test_a_round_trip_restores_everything_current(source, tmp_path):
    path = write(export_document(source), tmp_path / "export.json")
    with closing(db.open_db(tmp_path / "fresh.db")) as fresh:
        report = import_export(fresh, path, scope="personal")
        assert (report.memories, report.memories_present) == (2, 0)  # current versions, not history
        assert snapshot(fresh) == (
            [m for m in snapshot(source)[0] if m[1] != "Prefers tea."],
            snapshot(source)[1],
        )


def test_importing_an_export_into_the_same_store_adds_nothing(source, tmp_path):
    before = snapshot(source)
    report = import_export(source, write(export_document(source), tmp_path / "export.json"), scope="personal")
    assert (report.conversations, report.messages, report.memories, report.memories_present) == (0, 0, 0, 2)
    assert snapshot(source) == before


def test_scopes_can_be_overridden_on_import(source, tmp_path):
    path = write(export_document(source), tmp_path / "export.json")
    with closing(db.open_db(tmp_path / "fresh.db")) as fresh:
        import_export(fresh, path, scope="archive", keep_scopes=False)
        memories, conversations = snapshot(fresh)
        assert {m[2] for m in memories} == {c[1] for c in conversations} == {"archive"}


def test_exporting_some_scopes(source):
    doc = export_document(source, scopes=["work"])
    assert [m["content"] for m in doc["memories"]] == ["Uses Go at work."]
    assert [c["provider"] for c in doc["conversations"]] == ["vscode-copilot"]


def test_imported_memories_go_through_the_secret_filter(tmp_path):
    doc = {"format": "openmcp.export", "version": 1, "conversations": [], "memories": [
        {"kind": "fact", "content": "Deploy key is ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789",
         "created_at": "2026-10-01T00:00:00Z", "updated_at": "2026-10-01T00:00:00Z"},
        {"kind": "fact", "content": "Has a cat.", "created_at": "2026-10-01T00:00:00Z", "updated_at": "2026-10-01T00:00:00Z"},
        {"kind": "skill", "content": "Not a kind", "created_at": "2026-10-01T00:00:00Z", "updated_at": "2026-10-01T00:00:00Z"},
    ]}
    with closing(db.open_db(tmp_path / "t.db")) as conn:
        report = import_export(conn, write(doc, tmp_path / "x.json"), scope="personal")
        assert report.memories == 1 and len(report.errors) == 2
        assert [m.content for m in repo.list_memories(conn, scopes=None)] == ["Has a cat."]


# --- CLI -------------------------------------------------------------------------------


runner = CliRunner()


def cli(*args):
    result = runner.invoke(app, list(args), env={"COLUMNS": "200"})
    return result.exit_code, " ".join(plain(result.output).split()), result.output


def test_export_and_import_commands(home, tmp_path):
    initialize(home)
    with closing(db.connect(home / "openmcp.db")) as conn:
        fill(conn)
    target = tmp_path / "backup.json"
    code, out, _ = cli("export", "--output", str(target))
    assert code == 0 and "Exported 2 memories (1 earlier versions) and 2 conversations (5 messages)" in out
    if os.name != "nt":
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert json.loads(target.read_text())["format"] == "openmcp.export"

    code, out, raw = cli("export", "--output", "-", "--scope", "work")
    assert code == 0 and [m["content"] for m in json.loads(raw)["memories"]] == ["Uses Go at work."]

    code, out, _ = cli("import", str(target))
    assert code == 0 and "Imported the OpenMCP export keeping its scopes" in out
    assert "Memories: 0 added, 2 already here." in out
