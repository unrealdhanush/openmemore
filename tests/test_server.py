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

from conftest import plain
from openmcp import config, db
from openmcp.cli import app, initialize
from openmcp.config import DEFAULT_CONFIG, ClientScopes, Config, load_config, resolve_client
from conftest import connect, example_clients
from mcp.types import Implementation

runner = CliRunner()


@pytest.mark.anyio
async def test_ping_in_process(home):
    initialize(home)
    example_clients(home)
    async with connect(home, "cursor") as client:
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
        args=["-m", "openmcp.cli", "serve"],
        env={"OPENMCP_HOME": str(home)},
    )
    async with Client(params, client_info=Implementation(name="claude-desktop", version="1"), read_timeout_seconds=30) as client:
        result = await client.call_tool("ping", {})
        assert not result.is_error
        assert result.structured_content["client"] == "claude-desktop"
        assert result.structured_content["schema_version"] >= 1
    assert (home / "openmcp.db").is_file()


@pytest.mark.anyio
async def test_every_tool_call_is_audited(home):
    initialize(home)
    async with connect(home, "cursor") as client:
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
    CI sets OPENMCP_STARTUP_BUDGET_MS higher: shared runners start processes far slower
    than the machines the budget is for.
    """
    budget_ms = float(os.environ.get("OPENMCP_STARTUP_BUDGET_MS", "500"))
    initialize(home)
    init = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
    }) + "\n"
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        proc = subprocess.Popen(
            [sys.executable, "-m", "openmcp.cli", "serve"],
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
    assert min(timings) * 1000 < budget_ms, f"startup timings: {[f'{t * 1000:.0f}ms' for t in timings]}"


def test_serve_takes_no_client_name(home):
    """The client is named from what it reports; there's no option to say who it is."""
    result = runner.invoke(app, ["serve", "--client", "claude-desktop"])
    assert result.exit_code != 0 and "No such option" in plain(result.output)


def test_resolve_configured_and_unconfigured_clients():
    cfg = Config.model_validate({**DEFAULT_CONFIG, "clients": {"cursor-vscode": {"read_scopes": ["work"], "write_scope": "work"}}})
    assert resolve_client(cfg, "cursor-vscode") == ClientScopes(read_scopes=("work",), write_scope="work")
    assert resolve_client(cfg, "gemini-cli") == ClientScopes(read_scopes=("personal",), write_scope="personal")


def test_the_default_config_names_no_clients(tmp_path):
    """Clients are named by what their apps report, so a shipped [clients.<name>] would only guess."""
    path = tmp_path / "config.toml"
    config.write_default_config(path)
    assert config.load_config(path).clients == {}
    assert "[clients.cursor-vscode]" in path.read_text()  # as a commented example


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


def test_capture_config_is_validated(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[capture]\nsources = ["slack"]\n')
    with pytest.raises(ValidationError):
        config.load_config(path)


def test_enable_capture_keeps_the_rest_of_the_config(tmp_path):
    path = tmp_path / "config.toml"
    config.write_default_config(path)
    assert config.enable_capture(path, "claude-code") is True
    assert config.enable_capture(path, "claude-code") is False
    cfg = config.load_config(path)
    assert cfg.capture.sources == ("claude-code",)
    assert path.read_text().startswith("# OpenMCP configuration")  # the header survives rewrites


def test_app_pins_are_gone(tmp_path):
    """Config from before naming by app: an `app` pin is now an error that names the key."""
    path = tmp_path / "config.toml"
    path.write_text('[clients.cursor]\napp = "cursor-vscode"\n')
    with pytest.raises(ValidationError, match="app"):
        config.load_config(path)



def test_ignore_scopes_resolves_every_client_to_all_scopes(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('ignore_scopes = true\n[clients.x]\nread_scopes = ["work"]\nwrite_scope = "work"\n')
    cfg = config.load_config(path)
    assert resolve_client(cfg, "x") == ClientScopes(read_scopes=None, write_scope="work")
    assert resolve_client(cfg, "anyone").read_scopes is None
    config.write_default_config(tmp_path / "fresh.toml")
    assert "ignore_scopes = false" in (tmp_path / "fresh.toml").read_text()  # off unless asked


def test_saving_a_config_without_clients_writes_no_empty_table(tmp_path):
    path = tmp_path / "config.toml"
    config.save_config(path, Config())
    assert "[clients]" not in path.read_text() and config.load_config(path).clients == {}
