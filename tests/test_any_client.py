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
from openmcp.clients import CLIENTS, client_for_app, slug
from openmcp.models import CAPTURE_SOURCES
from openmcp.server import build_server
from openmcp.setup import Context, SetupError, plan

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


def test_client_names_come_from_pins_then_the_registry_then_the_app():
    assert client_for_app("Some Editor", {"Some Editor": "my-editor"}) == "my-editor"
    assert client_for_app("claude-ai", {}) == "claude-desktop"
    assert client_for_app("Visual Studio Code", {}) == "vscode-copilot"
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
async def test_known_apps_get_their_registry_name_without_client(home):
    initialize(home)
    async with Client(build_server(None, home), client_info=Implementation(name="claude-ai", version="1")) as c:
        ping = (await c.call_tool("ping", {})).structured_content
    assert (ping["client"], ping["read_scopes"]) == ("claude-desktop", ["personal", "work"])  # claude-desktop's scopes


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


# --- the registry ----------------------------------------------------------------------


def test_registry_entries_are_complete():
    for spec in CLIENTS.values():
        assert spec.name == slug(spec.name) and spec.display
        assert spec.server is not None or spec.name in ("vscode-copilot", "claude-code")  # special-cased in setup
        assert spec.capture in (None, *CAPTURE_SOURCES)
        if spec.verified:
            assert spec.app, f"{spec.name}: verified entries confirm their clientInfo name"
        for platform in ("darwin", "win32", "linux"):
            if spec.server:
                assert spec.server.path(Path("/home/u"), platform).name
            if spec.instruction.path:
                assert spec.instruction.path(Path("/home/u"), platform).name


@pytest.fixture
def ctx(tmp_path) -> Context:
    user_home = tmp_path / "user"
    base = user_home / ".openmcp"
    initialize(base)
    return Context(base=base, user_home=user_home, platform="darwin", command=COMMAND, which=lambda _: None)


def run(steps):
    errors = []
    for step in steps:
        if step.apply:
            try:
                step.apply()
            except SetupError as e:
                errors.append(str(e))
    return errors


def json_block(text):
    """The first JSON object printed in a step's instructions."""
    return json.JSONDecoder().raw_decode(text[text.index("{"):])[0]


def written(ctx):
    return sorted(str(p.relative_to(ctx.user_home)) for p in ctx.user_home.rglob("*")
                  if p.is_file() and ".openmcp" not in p.parts)


def test_unverified_clients_are_printed_unless_tried(ctx, clipboard):
    steps = plan("cursor", ctx)
    assert run(steps) == [] and written(ctx) == []
    manual = "\n".join(s.manual for s in steps if s.manual)
    assert "~/.cursor/mcp.json" not in manual and str(ctx.user_home / ".cursor" / "mcp.json") in manual
    assert "--try" in manual and json_block(manual)["mcpServers"]["openmcp"]["args"] == ["serve", "--client", "cursor"]

    run(plan("cursor", ctx, try_unverified=True))
    entry = json.loads((ctx.user_home / ".cursor" / "mcp.json").read_text())["mcpServers"]["openmcp"]
    assert entry["args"] == ["serve", "--client", "cursor"]


def test_trying_a_markdown_instruction_client(ctx):
    run(plan("gemini-cli", ctx, try_unverified=True))
    assert json.loads((ctx.user_home / ".gemini" / "settings.json").read_text())["mcpServers"]["openmcp"]
    assert "<!-- openmcp:start -->" in (ctx.user_home / ".gemini" / "GEMINI.md").read_text()


def test_toml_clients_keep_other_settings_and_refuse_commented_files(ctx):
    path = ctx.user_home / ".codex" / "config.toml"
    path.parent.mkdir(parents=True)
    path.write_text('model = "o4"\n')
    assert run(plan("codex-cli", ctx, try_unverified=True)) == []
    data = tomllib.loads(path.read_text())
    assert data["model"] == "o4" and data["mcp_servers"]["openmcp"]["args"] == ["serve", "--client", "codex-cli"]

    path.write_text('# my settings\nmodel = "o4"\n')
    [error] = run(plan("codex-cli", ctx, try_unverified=True))
    assert "comments" in error and path.read_text().startswith("# my settings")


def test_unknown_clients_get_the_generic_snippet_without_a_client_name(ctx, clipboard):
    steps = plan("some-new-editor", ctx)
    assert run(steps) == [] and written(ctx) == []
    manual = "\n".join(s.manual for s in steps if s.manual)
    assert json_block(manual)["mcpServers"]["openmcp"]["args"] == ["serve"]
    assert "any MCP client works" in manual
    assert len(clipboard) == 1


# --- CLI -------------------------------------------------------------------------------


def test_clients_command_lists_the_registry(home):
    from typer.testing import CliRunner
    from openmcp.cli import app
    result = CliRunner().invoke(app, ["clients"], env={"COLUMNS": "200"})
    out = plain(result.output)
    assert result.exit_code == 0
    for spec in CLIENTS.values():
        assert spec.name in out
    assert "printed (--try writes)" in out and "Any other MCP client" in out


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
