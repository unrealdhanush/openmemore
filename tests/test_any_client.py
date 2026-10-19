"""SPEC.md M3.5: any MCP client, through one generic snippet, named by what it reports."""

import json
import os
import sys
import tomllib
from pathlib import Path

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types import Implementation

from conftest import plain
from openmcp import config
from openmcp.cli import initialize
from openmcp.clients import client_for_app, slug
from openmcp.server import build_server

SOME_EDITOR = Implementation(name="Some Editor 2", version="1.0")
COMMAND = ("/opt/openmcp/bin/openmcp",)


def generic(home, app):
    """The generic snippet's server: `openmcp serve`, no --client, in its own process."""
    return Client(
        StdioServerParameters(command=sys.executable, args=["-m", "openmcp.cli", "serve"],
                              env={"OPENMCP_HOME": str(home)}),
        client_info=app, read_timeout_seconds=30,
    )


# --- naming clients from what they report ----------------------------------------------


def test_client_names_come_from_pins_then_the_app_itself():
    """No table of known clients: a name is what the app reports, unless config pins one."""
    assert client_for_app("Some Editor", {"Some Editor": "my-editor"}) == "my-editor"
    assert client_for_app("claude-ai", {}) == "claude-ai"
    assert client_for_app("Visual Studio Code", {}) == "visual-studio-code"
    assert client_for_app("cursor-vscode", {}) == "cursor-vscode"
    assert client_for_app("Some Editor 2", {}) == "some-editor-2"
    assert client_for_app(None, {}) == "mcp-client"
    assert slug("Zed / Preview!") == "zed-preview" and slug("???") == "mcp-client"


@pytest.mark.anyio
async def test_m3_5_done_when_an_unknown_client_works_from_the_generic_snippet(home):
    """No registry entry, no --client: the server names the client from clientInfo, applies
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
    assert (ping["client"], ping["identified_by"]) == ("some-editor-2", "clientInfo")
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
async def test_configs_from_before_keep_their_names_through_pins(home):
    """Existing [clients.claude-desktop] scopes keep applying once the app is pinned to it."""
    initialize(home)
    config.set_client_app(home / "config.toml", "claude-desktop", "claude-ai")
    async with Client(build_server(None, home), client_info=Implementation(name="claude-ai", version="1")) as c:
        ping = (await c.call_tool("ping", {})).structured_content
    assert (ping["client"], ping["read_scopes"]) == ("claude-desktop", ["personal", "work"])


@pytest.mark.anyio
async def test_a_captured_app_without_a_client_name_isnt_recorded_twice(home):
    """VS Code started from the generic snippet, with its transcripts captured: its
    record_exchange calls are skipped, not stored on top of the captured chat."""
    initialize(home)
    config.enable_capture(home / "config.toml", "vscode-copilot")
    vscode = Implementation(name="Visual Studio Code", version="1.99.3")
    async with Client(build_server(None, home), client_info=vscode) as c:
        result = (await c.call_tool("record_exchange", {"question": "q", "answer_summary": "a"})).structured_content
    assert result["conversation_id"] is None and "captured" in result["note"]
    async with Client(build_server(None, home), client_info=SOME_EDITOR) as c:
        stored = (await c.call_tool("record_exchange", {"question": "q", "answer_summary": "a"})).structured_content
    assert stored["conversation_id"] is not None


@pytest.mark.anyio
async def test_a_pin_in_config_names_an_app(home):
    initialize(home)
    config.set_client_app(home / "config.toml", "my-editor", "Some Editor 2")
    async with Client(build_server(None, home), client_info=SOME_EDITOR) as c:
        ping = (await c.call_tool("ping", {})).structured_content
    assert (ping["client"], ping["warning"]) == ("my-editor", None)


@pytest.mark.anyio
async def test_an_explicit_client_still_wins(home):
    initialize(home)
    async with Client(build_server("cursor", home), client_info=SOME_EDITOR) as c:
        ping = (await c.call_tool("ping", {})).structured_content
    assert (ping["client"], ping["identified_by"]) == ("cursor", "--client")


# --- CLI -------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_clients_command_lists_the_apps_that_connected(home):
    from typer.testing import CliRunner
    from openmcp.cli import app
    initialize(home)
    async with Client(build_server(None, home), client_info=SOME_EDITOR) as c:
        await c.call_tool("ping", {})
    result = CliRunner().invoke(app, ["clients"], env={"COLUMNS": "200"})
    out = plain(result.output)
    assert result.exit_code == 0
    assert "some-editor-2" in out and "Some Editor 2" in out and "record_exchange" in out


def test_clients_command_before_anything_connects(home):
    from typer.testing import CliRunner
    from openmcp.cli import app
    initialize(home)
    assert "No MCP client has connected yet" in plain(CliRunner().invoke(app, ["clients"]).output)


# --- importers from any provider ---------------------------------------------------------

from openmcp.ingest import registry  # noqa: E402
from openmcp.ingest.detect import ExportError, load  # noqa: E402
from openmcp.ingest.jobs import import_export  # noqa: E402
from openmcp.tools.conversations import recent_conversations  # noqa: E402
from conftest import FakeEmbedder, caller  # noqa: E402

READER = caller("reader", read=["personal"], write="personal")


def openmcp_doc(*conversations, version=1):
    return {"format": "openmcp.conversations", "version": version, "conversations": list(conversations)}


def conv(provider, id_, *pairs, updated="2026-10-01T10:00:00Z"):
    return {
        "provider": provider, "id": id_, "updated_at": updated,
        "messages": [{"id": f"{id_}-{n}", "role": role, "content": text} for n, (role, text) in enumerate(pairs)],
    }


def write(tmp_path, doc, name="history.json"):
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return path


def test_the_openmcp_format_brings_in_any_providers_history(conn, tmp_path):
    path = write(tmp_path, openmcp_doc(
        conv("gemini", "g1", ("user", "Plan a vegetable garden"), ("assistant", "Start with tomatoes and basil.")),
        conv("perplexity", "p1", ("user", "Best trail shoes?"), ("assistant", "Look for a rock plate."),
             updated="2026-10-02T10:00:00Z"),
    ))
    report = import_export(conn, path, scope="personal", embedder=FakeEmbedder())

    assert (report.provider, report.conversations, report.embedded) == ("openmcp", 2, 4)
    assert [c.source for c in recent_conversations(conn, READER).conversations] == ["perplexity", "gemini"]
    assert recent_conversations(conn, READER, source="gemini").conversations[0].title == "Plan a vegetable garden"


def test_openmcp_format_problems_are_reported(conn, tmp_path):
    bad_role = conv("gemini", "g2", ("system", "You are helpful"), ("user", "hi"))
    report = import_export(conn, write(tmp_path, openmcp_doc(bad_role, conv("gemini", "g3", ("user", "ok")))), scope="personal")
    assert report.conversations == 1 and len(report.errors) == 1 and "role" in report.errors[0]
    with pytest.raises(ExportError, match="version 2 isn't supported"):
        load(write(tmp_path, openmcp_doc(version=2), "v2.json"))


def test_a_new_provider_needs_only_a_registry_entry(conn, tmp_path, monkeypatch):
    """SPEC M3.5 done-when: a provider's importer added through the registry alone.
    An invented format, imported end to end with no other code changed."""
    from openmcp.models import ParsedConversation, ParsedMessage

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
    assert len(names) == len(set(names)) and {"chatgpt", "claude", "openmcp"} <= set(names)
