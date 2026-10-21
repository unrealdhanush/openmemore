"""Any MCP client, through one generic snippet, named by what it reports."""

import json
import os
import sys
import tomllib
from pathlib import Path

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types import Implementation

from conftest import connect, plain
from openmemore import config
from openmemore.cli import initialize
from openmemore.clients import client_name
from openmemore.server import build_server

SOME_EDITOR = Implementation(name="Some Editor 2", version="1.0")
COMMAND = ("/opt/openmemore/bin/openmemore",)


def generic(home, app):
    """The generic snippet's server: `openmemore serve`, in its own process."""
    return Client(
        StdioServerParameters(command=sys.executable, args=["-m", "openmemore.cli", "serve"],
                              env={"OPENMEMORE_HOME": str(home)}),
        client_info=app, read_timeout_seconds=30,
    )


# --- naming clients from what they report ----------------------------------------------


def test_client_names_are_the_apps_own():
    """No table of known clients and no renaming: a name is what the app reports."""
    assert client_name("claude-ai") == "claude-ai"
    assert client_name("Visual Studio Code") == "visual-studio-code"
    assert client_name("cursor-vscode") == "cursor-vscode"
    assert client_name("Some Editor 2") == "some-editor-2"
    assert client_name("Zed / Preview!") == "zed-preview"
    assert client_name(None) == client_name("???") == "mcp-client"


@pytest.mark.anyio
async def test_an_unknown_client_works_from_the_generic_snippet(home):
    """Nothing about this client is known: the server names it from clientInfo, applies
    the scopes configured for that name, and it records and recalls like any client."""
    initialize(home)
    path = home / "config.toml"
    cfg = config.load_config(path)
    cfg.clients["some-editor-2"] = config.ClientOverride(read_scopes=("work",), write_scope="work")
    config.save_config(path, cfg)

    async with generic(home, SOME_EDITOR) as editor:
        ping = (await editor.call_tool("ping", {})).structured_content
        recorded = await editor.call_tool("record_exchange", {
            "question": "How do I profile a slow SQLite query?", "answer_summary": "Use EXPLAIN QUERY PLAN.",
        })
        assert not recorded.is_error
    assert ping["client"] == "some-editor-2"
    assert (ping["read_scopes"], ping["write_scope"]) == (["work"], "work")

    async with generic(home, Implementation(name="Another Tool", version="3")) as other:
        # Unconfigured: [defaults] scopes (personal), so it can't see the work-scoped chat...
        assert (await other.call_tool("ping", {})).structured_content["client"] == "another-tool"
        found = await other.call_tool("search_context", {"query": "SQLite query plan"})
        assert found.structured_content["results"] == []

    async with generic(home, SOME_EDITOR) as editor:
        recent = await editor.call_tool("recent_conversations", {})
        [conv] = recent.structured_content["conversations"]
        assert conv["source"] == "some-editor-2"
        found = await editor.call_tool("search_context", {"query": "slow SQLite query"})
        assert found.structured_content["results"][0]["conversation_id"] == conv["id"]


@pytest.mark.anyio
async def test_a_captured_app_isnt_recorded_twice(home):
    """VS Code, with its transcripts captured: its record_exchange calls are skipped,
    not stored on top of the captured chat."""
    initialize(home)
    config.enable_capture(home / "config.toml", "vscode-copilot")
    vscode = Implementation(name="Visual Studio Code", version="1.99.3")
    async with Client(build_server(home), client_info=vscode) as c:
        result = (await c.call_tool("record_exchange", {"question": "q", "answer_summary": "a"})).structured_content
    assert result["conversation_id"] is None and "captured" in result["note"]
    async with Client(build_server(home), client_info=SOME_EDITOR) as c:
        stored = (await c.call_tool("record_exchange", {"question": "q", "answer_summary": "a"})).structured_content
    assert stored["conversation_id"] is not None


@pytest.mark.anyio
async def test_each_call_records_the_application(home):
    """The audit log keeps what each application called itself, beside its client name;
    resources (which get no context) use the session's application."""
    initialize(home)
    async with Client(build_server(home), client_info=Implementation(name="claude-ai", version="1")) as c:
        assert (await c.call_tool("ping", {})).structured_content["connected_app"] == "claude-ai"
        await c.read_resource("openmemore://profile")
    rows = [tuple(r) for r in config_db(home).execute("SELECT client, app, tool FROM audit_log ORDER BY id")]
    assert rows == [("claude-ai", "claude-ai", "ping"), ("claude-ai", "claude-ai", "openmemore://profile")]


def config_db(home):
    from openmemore import db
    return db.connect(home / "openmemore.db")


# --- CLI -------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_clients_command_lists_the_apps_that_connected(home):
    from typer.testing import CliRunner
    from openmemore.cli import app
    initialize(home)
    async with Client(build_server(home), client_info=SOME_EDITOR) as c:
        await c.call_tool("ping", {})
    result = CliRunner().invoke(app, ["clients"], env={"COLUMNS": "200"})
    out = plain(result.output)
    assert result.exit_code == 0
    assert "some-editor-2" in out and "Some Editor 2" in out and "record_exchange" in out


def test_clients_command_before_anything_connects(home):
    from typer.testing import CliRunner
    from openmemore.cli import app
    initialize(home)
    assert "No MCP client has connected yet" in plain(CliRunner().invoke(app, ["clients"]).output)


# --- importers from any provider ---------------------------------------------------------

from openmemore.ingest import registry  # noqa: E402
from openmemore.ingest.detect import ExportError, load  # noqa: E402
from openmemore.ingest.jobs import import_export  # noqa: E402
from openmemore.tools.conversations import recent_conversations  # noqa: E402
from conftest import FakeEmbedder, caller  # noqa: E402

READER = caller("reader", read=["personal"], write="personal")


def openmemore_doc(*conversations, version=1):
    return {"format": "openmemore.export", "version": version, "conversations": list(conversations)}


def conv(provider, id_, *pairs, updated="2026-10-01T10:00:00Z"):
    return {
        "provider": provider, "id": id_, "updated_at": updated,
        "messages": [{"id": f"{id_}-{n}", "role": role, "content": text} for n, (role, text) in enumerate(pairs)],
    }


def write(tmp_path, doc, name="history.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return path


def test_the_openmemore_format_brings_in_any_providers_history(conn, tmp_path):
    path = write(tmp_path, openmemore_doc(
        conv("gemini", "g1", ("user", "Plan a vegetable garden"), ("assistant", "Start with tomatoes and basil.")),
        conv("perplexity", "p1", ("user", "Best trail shoes?"), ("assistant", "Look for a rock plate."),
             updated="2026-10-02T10:00:00Z"),
    ))
    report = import_export(conn, path, scope="personal", embedder=FakeEmbedder())

    assert (report.provider, report.conversations, report.embedded) == ("openmemore", 2, 4)
    assert [c.source for c in recent_conversations(conn, READER).conversations] == ["perplexity", "gemini"]
    assert recent_conversations(conn, READER, source="gemini").conversations[0].title == "Plan a vegetable garden"


def test_openmemore_format_problems_are_reported(conn, tmp_path):
    bad_role = conv("gemini", "g2", ("system", "You are helpful"), ("user", "hi"))
    report = import_export(conn, write(tmp_path, openmemore_doc(bad_role, conv("gemini", "g3", ("user", "ok")))), scope="personal")
    assert report.conversations == 1 and len(report.errors) == 1 and "role" in report.errors[0]
    with pytest.raises(ExportError, match="version 2 isn't supported"):
        load(write(tmp_path, openmemore_doc(version=2), "v2.json"))


def test_a_new_provider_needs_only_a_registry_entry(conn, tmp_path, monkeypatch):
    """A provider's importer added through the registry alone.
    An invented format, imported end to end with no other code changed."""
    from openmemore.models import ParsedConversation, ParsedMessage

    toy = registry.Importer(
        name="toy", display="Toy Assistant",
        matches=lambda doc: isinstance(doc, dict) and "toy_chats" in doc,
        items=lambda doc: doc["toy_chats"],
        updated=lambda raw: raw["t"],
        identity=lambda raw: ("toy", raw["key"]),
        conversation=lambda raw: ParsedConversation(
            provider="toy", external_id=raw["key"], title=raw["q"], metadata={"capture": "import"},
            messages=[ParsedMessage(external_id=f"{raw['key']}-q", role="user", content=raw["q"]),
                      ParsedMessage(external_id=f"{raw['key']}-a", role="assistant", content=raw["a"])],
        ),
    )
    monkeypatch.setattr(registry, "_registered", [])  # undone after the test
    registry.register(toy)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(toy)
    path = write(tmp_path, {"toy_chats": [{"key": "k1", "t": 1, "q": "Old question", "a": "Old answer"},
                                          {"key": "k2", "t": 2, "q": "New question", "a": "New answer"}]})

    report = import_export(conn, path, scope="personal", embedder=FakeEmbedder())
    assert (report.provider, report.conversations, report.embedded) == ("toy", 2, 4)
    assert [c.title for c in recent_conversations(conn, READER, source="toy").conversations][0] == "New question"


def test_importer_registry_entries_are_distinct():
    names = [i.name for i in registry.importers()]
    assert len(names) == len(set(names)) and {"chatgpt", "claude", "openmemore"} <= set(names)


@pytest.mark.anyio
async def test_clients_command_hides_earlier_names_unless_asked(home):
    from typer.testing import CliRunner
    from openmemore.cli import app
    from openmemore import db as openmemore_db
    initialize(home)
    async with Client(build_server(home), client_info=SOME_EDITOR) as c:
        await c.call_tool("ping", {})
    conn = openmemore_db.connect(home / "openmemore.db")  # rows as earlier versions wrote them
    conn.execute("INSERT INTO audit_log (ts, client, tool, duration_ms, app) VALUES ('2026-10-01T00:00:00.000Z', 'old-name', 'ping', 1, 'Some Editor 2')")
    conn.execute("INSERT INTO audit_log (ts, client, tool, duration_ms) VALUES ('2026-10-01T00:00:00.000Z', 'older', 'ping', 1)")
    out = plain(CliRunner().invoke(app, ["clients"], env={"COLUMNS": "200"}).output)
    assert "some-editor-2" in out and "old-name" not in out and "2 earlier names" in out
    assert "under [clients.<name>] in config.toml" in " ".join(out.split())
    out = plain(CliRunner().invoke(app, ["clients", "--all"], env={"COLUMNS": "200"}).output)
    assert "old-name" in out and "older" in out
