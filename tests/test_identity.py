"""The application behind each call (MCP clientInfo) is recorded, and one that isn't the
usual one for a client name is flagged. Found in the 2026-10-18 manual test: VS Code
started the server configured for Claude Desktop, so Copilot ran as claude-desktop."""

import pytest
from mcp import Client
from mcp.types import Implementation

from openmcp import config, db
from openmcp.cli import initialize
from openmcp.server import build_server

VSCODE = Implementation(name="Visual Studio Code", version="1.99.3")
CLAUDE = Implementation(name="claude-ai", version="0.14")


def audit_rows(home):
    conn = db.connect(home / "openmcp.db")
    return [tuple(r) for r in conn.execute("SELECT client, app, tool, summary FROM audit_log ORDER BY id")]


@pytest.mark.anyio
async def test_each_call_records_the_application(home):
    initialize(home)
    async with Client(build_server("claude-desktop", home), client_info=CLAUDE) as client:
        ping = (await client.call_tool("ping", {})).structured_content
        await client.call_tool("search_context", {"query": "anything"})
        await client.read_resource("openmcp://profile")  # resources get no context: the session's app is used
    assert (ping["connected_app"], ping["warning"]) == ("claude-ai", None)
    assert [(c, a, t) for c, a, t, _ in audit_rows(home)] == [
        ("claude-desktop", "claude-ai", "ping"),
        ("claude-desktop", "claude-ai", "search_context"),
        ("claude-desktop", "claude-ai", "openmcp://profile"),
    ]


async def ping_as(home, client, app):
    async with Client(build_server(client, home), client_info=app) as session:
        return (await session.call_tool("ping", {})).structured_content["warning"]


@pytest.fixture
def pinned(home):
    """What `openmcp setup vscode-copilot` records."""
    initialize(home)
    config.set_client_app(home / "config.toml", "vscode-copilot", "Visual Studio Code")
    return home


@pytest.mark.anyio
@pytest.mark.parametrize("vscode_first", [True, False])
async def test_vscode_using_claude_desktops_entry_is_flagged_in_either_order(pinned, vscode_first):
    """The manual test: VS Code started the server configured for Claude Desktop. Which
    app connects first must not decide which one is flagged."""
    order = [VSCODE, CLAUDE] if vscode_first else [CLAUDE, VSCODE]
    warnings = {app.name: await ping_as(pinned, "claude-desktop", app) for app in order}

    assert warnings["claude-ai"] is None
    assert "'Visual Studio Code' is the app for client 'vscode-copilot' but is using 'claude-desktop'" in warnings[
        "Visual Studio Code"
    ]
    assert "chat.mcp.discovery.enabled" in warnings["Visual Studio Code"]


@pytest.mark.anyio
async def test_a_pinned_client_name_flags_any_other_app(pinned):
    config.set_client_app(pinned / "config.toml", "claude-desktop", "claude-ai")
    assert await ping_as(pinned, "claude-desktop", CLAUDE) is None
    other = Implementation(name="Some Other Editor", version="1")
    assert "which is for 'claude-ai'" in await ping_as(pinned, "claude-desktop", other)


@pytest.mark.anyio
async def test_unpinned_names_used_by_two_apps_flag_both_without_picking(home):
    initialize(home)
    other = Implementation(name="Some Other Editor", version="1")
    assert await ping_as(home, "my-client", CLAUDE) is None  # one app so far: nothing to compare
    second = await ping_as(home, "my-client", other)
    first_again = await ping_as(home, "my-client", CLAUDE)  # a new server process, as after a restart
    for warning in (second, first_again):
        assert "used by several apps (Some Other Editor, claude-ai)" in warning
        assert '[clients.my-client]' in warning


@pytest.mark.anyio
async def test_flagged_calls_are_noted_in_the_audit_log(pinned):
    async with Client(build_server("claude-desktop", pinned), client_info=VSCODE) as session:
        await session.call_tool("search_context", {"query": "living room"})
    client, app, tool, summary = audit_rows(pinned)[-1]
    assert (client, app, tool) == ("claude-desktop", "Visual Studio Code", "search_context")
    assert summary.endswith("; unexpected app")


@pytest.mark.anyio
async def test_the_usual_application_is_not_flagged_across_restarts(home):
    initialize(home)
    for _ in range(2):  # a new server process each time, as after an app restart
        assert await ping_as(home, "claude-desktop", CLAUDE) is None
