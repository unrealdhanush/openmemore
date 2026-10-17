import json
import os
import subprocess
import sys
import time

import pytest
import tomli_w
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from pydantic import ValidationError
from typer.testing import CliRunner

from openmcp import db
from openmcp.cli import app, initialize
from openmcp.config import DEFAULT_CONFIG, ClientScopes, Config, load_config, resolve_client
from openmcp.server import build_server

runner = CliRunner()


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


@pytest.mark.anyio
async def test_every_tool_call_is_audited(home):
    initialize(home)
    async with Client(build_server("cursor", home)) as client:
        await client.call_tool("ping", {})
        await client.call_tool("ping", {})

    conn = db.connect(home / "openmcp.db")
    rows = conn.execute("SELECT ts, client, tool, duration_ms FROM audit_log ORDER BY id").fetchall()
    assert [(r["client"], r["tool"]) for r in rows] == [("cursor", "ping")] * 2
    assert all(r["duration_ms"] > 0 for r in rows)
    assert all(r["ts"].endswith("Z") for r in rows)


def test_startup_under_budget(home):
    """SPEC 4.3: `openmcp serve` answers its first request in under 500ms.

    Best of 3, so a one-off cold start (bytecode compile, disk cache) doesn't fail the build.
    """
    initialize(home)
    init = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
    }) + "\n"
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        proc = subprocess.Popen(
            [sys.executable, "-m", "openmcp.cli", "serve", "--client", "bench"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            env={**os.environ, "OPENMCP_HOME": str(home)},
        )
        try:
            proc.stdin.write(init)
            proc.stdin.flush()
            assert '"result"' in proc.stdout.readline()
            timings.append(time.perf_counter() - start)
        finally:
            proc.kill()
            proc.wait()
    assert min(timings) < 0.5, f"startup timings: {[f'{t * 1000:.0f}ms' for t in timings]}"


@pytest.mark.parametrize("name", ["", " ", "has space", "-leading-dash", "a/b", "x" * 65])
def test_serve_rejects_bad_client_names(home, name):
    result = runner.invoke(app, ["serve", "--client", name])
    assert result.exit_code != 0
    assert "--client" in result.output
    assert not (home / "openmcp.db").exists()


def test_resolve_known_and_unknown_clients():
    cfg = Config.model_validate(DEFAULT_CONFIG)
    assert resolve_client(cfg, "cursor") == ClientScopes(read_scopes=("work",), write_scope="work")
    assert resolve_client(cfg, "gemini-cli") == ClientScopes(read_scopes=("personal",), write_scope="personal")


def test_resolve_partial_client_entry_falls_back_per_key():
    cfg = Config.model_validate({
        "defaults": {"read_scopes": ["personal"], "write_scope": "personal"},
        "clients": {"x": {"write_scope": "work"}},
    })
    assert resolve_client(cfg, "x") == ClientScopes(read_scopes=("personal",), write_scope="work")


@pytest.mark.parametrize("bad", [
    {"defaults": {"read_scopes": "work", "write_scope": "work"}},     # string, not a list
    {"clients": {"cursor": {"read_scope": ["work"]}}},                # typo'd key
    {"clients": {"cursor": {"write_scope": ""}}},                     # empty scope
])
def test_invalid_config_is_rejected(home, bad):
    initialize(home)
    (home / "config.toml").write_text(tomli_w.dumps(bad))
    with pytest.raises(ValidationError):
        load_config(home / "config.toml")
