import sys

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from openmcp.cli import initialize
from openmcp.config import ClientScopes, DEFAULT_CONFIG, resolve_client
from openmcp.server import build_server


@pytest.mark.anyio
async def test_ping_in_process(home):
    initialize(home)
    async with Client(build_server("cursor", home)) as client:
        tools = await client.list_tools()
        assert "ping" in [t.name for t in tools.tools]

        result = await client.call_tool("ping", {})
        assert not result.is_error
        assert result.structured_content["ok"] is True
        assert result.structured_content["client"] == "cursor"
        assert result.structured_content["read_scopes"] == ["work"]


@pytest.mark.anyio
async def test_ping_over_stdio(home):
    """End to end: spawn `openmcp serve` as a real subprocess, the way MCP clients do.

    Serve must also bootstrap a data dir that `init` never created.
    """
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "openmcp.cli", "serve", "--client", "claude-desktop"],
        env={"OPENMCP_HOME": str(home)},
    )
    async with Client(params, read_timeout_seconds=30) as client:
        result = await client.call_tool("ping", {})
        assert not result.is_error
        assert result.structured_content["client"] == "claude-desktop"
        assert result.structured_content["schema_version"] >= 1
    assert (home / "openmcp.db").is_file()


def test_resolve_known_and_unknown_clients():
    assert resolve_client(DEFAULT_CONFIG, "cursor") == ClientScopes(("work",), "work")
    assert resolve_client(DEFAULT_CONFIG, "gemini-cli") == ClientScopes(("personal",), "personal")


def test_resolve_partial_client_entry_falls_back_per_key():
    cfg = {"defaults": {"read_scopes": ["personal"], "write_scope": "personal"},
           "clients": {"x": {"write_scope": "work"}}}
    assert resolve_client(cfg, "x") == ClientScopes(("personal",), "work")
