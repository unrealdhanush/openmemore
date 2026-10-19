"""`openmcp setup`: every file it edits lives under a temporary home directory."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from openmcp import config
from openmcp.capture import vscode_copilot
from openmcp.cli import initialize
from openmcp.instructions import STANDING_PREFIX, server_instructions, standing_instruction
from openmcp.setup import (
    COPILOT_INSTRUCTIONS,
    VSCODE_DISCOVERY,
    Context,
    SetupError,
    claude_desktop_config,
    installed_command,
    plan,
)

COMMAND = ("/opt/openmcp/bin/openmcp",)


@pytest.fixture
def ctx(tmp_path) -> Context:
    user_home = tmp_path / "user"
    base = user_home / ".openmcp"
    initialize(base)
    return Context(base=base, user_home=user_home, platform="darwin", command=COMMAND, which=lambda _: None)


def run(steps):
    """Apply every automatic step; return the SetupErrors raised."""
    errors = []
    for step in steps:
        if step.apply:
            try:
                step.apply()
            except SetupError as e:
                errors.append(str(e))
    return errors


# --- instructions --------------------------------------------------------------------


def test_standing_instruction_variants():
    recording, captured = standing_instruction(captured=False), standing_instruction(captured=True)
    assert recording.startswith(STANDING_PREFIX) and captured.startswith(STANDING_PREFIX)
    assert "call record_exchange" in recording and "`code`" in recording
    assert "Don't call record_exchange" in captured
    for text in (recording, captured):
        assert "get_profile" in text and "save_memories" in text and "recent_conversations" in text


def test_server_instructions_variants():
    assert "call record_exchange" in server_instructions(captured=False)
    assert "record_exchange" not in server_instructions(captured=True)
    assert "kind 'decision'" in server_instructions(captured=True)


# --- Claude Desktop ------------------------------------------------------------------


def test_claude_desktop_adds_a_top_level_server_and_keeps_other_settings(ctx):
    path = claude_desktop_config(ctx)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"preferences": {"sidebarMode": "chat"}, "mcpServers": {"other": {"command": "x"}}}))

    steps = plan("claude-desktop", ctx)
    assert run(steps) == []

    data = json.loads(path.read_text())
    assert data["preferences"] == {"sidebarMode": "chat"}
    assert data["mcpServers"]["other"] == {"command": "x"}
    assert data["mcpServers"]["openmcp"] == {"command": COMMAND[0], "args": ["serve", "--client", "claude-desktop"]}
    assert json.loads(path.with_name(path.name + ".bak").read_text())["mcpServers"] == {"other": {"command": "x"}}
    manual = "\n".join(s.manual for s in steps if s.manual)
    assert "Settings → Profile" in manual and standing_instruction(captured=False) in manual


def test_claude_desktop_creates_the_config_if_missing(ctx):
    assert run(plan("claude-desktop", ctx)) == []
    assert "openmcp" in json.loads(claude_desktop_config(ctx).read_text())["mcpServers"]


def test_invalid_json_is_left_alone(ctx):
    path = claude_desktop_config(ctx)
    path.parent.mkdir(parents=True)
    path.write_text("{ broken")
    [error] = run(plan("claude-desktop", ctx))
    assert "isn't plain JSON" in error and '"mcpServers"' in error
    assert path.read_text() == "{ broken"


def test_a_non_default_data_directory_is_passed_to_the_client(ctx, tmp_path):
    other = Context(base=tmp_path / "elsewhere", user_home=ctx.user_home, platform="darwin", command=COMMAND)
    run(plan("claude-desktop", other))
    entry = json.loads(claude_desktop_config(other).read_text())["mcpServers"]["openmcp"]
    assert entry["env"] == {"OPENMCP_HOME": str(tmp_path / "elsewhere")}


# --- VS Code -------------------------------------------------------------------------


def vscode_settings(ctx) -> Path:
    return ctx.user_home / "Library" / "Application Support" / "Code" / "User" / "settings.json"


def test_vscode_adds_server_instruction_and_capture(ctx):
    settings = vscode_settings(ctx)
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"editor.fontSize": 13, "chat.agent.enabled": False}))

    assert run(plan("vscode-copilot", ctx)) == []
    run(plan("vscode-copilot", ctx))  # again: replaces, doesn't duplicate

    data = json.loads(settings.read_text())
    assert data["editor.fontSize"] == 13
    assert data["chat.agent.enabled"] is False  # an explicit user choice is kept
    assert data["mcp"]["servers"]["openmcp"]["args"] == ["serve", "--client", "vscode-copilot"]
    assert data[COPILOT_INSTRUCTIONS] == [{"text": standing_instruction(captured=True)}]
    assert data[VSCODE_DISCOVERY] is False  # or Copilot would also start Claude Desktop's entry
    assert config.load_config(config.config_path(ctx.base)).capture.sources == ("vscode-copilot",)
    assert config.load_config(config.config_path(ctx.base)).clients["vscode-copilot"].app == "Visual Studio Code"


def test_vscode_uses_mcp_json_when_present(ctx):
    mcp_json = vscode_settings(ctx).with_name("mcp.json")
    mcp_json.parent.mkdir(parents=True)
    mcp_json.write_text("{}")
    run(plan("vscode-copilot", ctx))
    assert "openmcp" in json.loads(mcp_json.read_text())["servers"]
    assert "mcp" not in json.loads(vscode_settings(ctx).read_text())


def test_vscode_settings_with_comments_are_not_rewritten(ctx):
    settings = vscode_settings(ctx)
    settings.parent.mkdir(parents=True)
    settings.write_text('{\n  // my font\n  "editor.fontSize": 13,\n}\n')
    errors = run(plan("vscode-copilot", ctx))
    assert len(errors) == 3 and all("comments" in e for e in errors)
    assert "// my font" in settings.read_text()


# --- Claude Code ---------------------------------------------------------------------


def test_claude_code_without_the_cli_prints_the_command(ctx):
    [error] = run(plan("claude-code", ctx))
    assert "claude mcp add --scope user openmcp -- /opt/openmcp/bin/openmcp serve --client claude-code" in error


def test_claude_code_instruction_block_is_replaced_not_duplicated(ctx):
    memory = ctx.user_home / ".claude" / "CLAUDE.md"
    memory.parent.mkdir(parents=True)
    memory.write_text("# My notes\n\nUse tabs.\n")
    run(plan("claude-code", ctx))
    run(plan("claude-code", ctx))
    text = memory.read_text()
    assert text.startswith("# My notes\n\nUse tabs.\n\n<!-- openmcp:start -->")
    assert text.count("<!-- openmcp:start -->") == 1
    assert standing_instruction(captured=True) in text
    assert config.load_config(config.config_path(ctx.base)).capture.sources == ("claude-code",)
    assert config.load_config(config.config_path(ctx.base)).clients["claude-code"].app == "claude-code"


# --- other clients -------------------------------------------------------------------


def test_unverified_clients_get_printed_instructions_only(ctx):
    steps = plan("cursor", ctx)
    assert all(s.apply is None for s in steps)
    manual = "\n".join(s.manual for s in steps)
    assert '"--client"' in manual and '"cursor"' in manual and standing_instruction(captured=False) in manual


# --- operating systems ---------------------------------------------------------------

CLAUDE_CFG = "Claude/claude_desktop_config.json"


@pytest.mark.parametrize("platform, env, expected", [
    ("darwin", {}, f"Library/Application Support/{CLAUDE_CFG}"),
    ("win32", {"APPDATA": "Roaming-Redirected"}, f"Roaming-Redirected/{CLAUDE_CFG}"),
    ("win32", {}, f"AppData/Roaming/{CLAUDE_CFG}"),
    ("linux", {"XDG_CONFIG_HOME": "xdg"}, f"xdg/{CLAUDE_CFG}"),
    ("linux", {}, f".config/{CLAUDE_CFG}"),
])
def test_app_paths_follow_each_os(ctx, monkeypatch, platform, env, expected):
    for var, value in env.items():
        monkeypatch.setenv(var, str(ctx.user_home / value))
    other = Context(base=ctx.base, user_home=ctx.user_home, platform=platform, command=COMMAND)
    assert claude_desktop_config(other).relative_to(ctx.user_home).as_posix() == expected
    vscode = vscode_copilot.user_dir(ctx.user_home, platform).relative_to(ctx.user_home).as_posix()
    assert vscode == expected.replace(CLAUDE_CFG, "Code/User")


def test_files_are_read_and_written_as_utf8(ctx):
    path = claude_desktop_config(ctx)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"name": "Zoë → 東京"}, ensure_ascii=False).encode())  # with a BOM
    assert run(plan("claude-desktop", ctx)) == []
    assert json.loads(path.read_text(encoding="utf-8"))["name"] == "Zoë → 東京"

    memory = ctx.user_home / ".claude" / "CLAUDE.md"
    memory.parent.mkdir(parents=True)
    memory.write_text("Notes: café, naïve, 日本語\n", encoding="utf-8")
    run(plan("claude-code", ctx))
    assert memory.read_text(encoding="utf-8").startswith("Notes: café, naïve, 日本語\n")


def test_claude_code_is_run_by_its_resolved_path(ctx, monkeypatch):
    """On Windows an npm install is claude.cmd, which subprocess can't start by bare name."""
    calls = []
    monkeypatch.setattr(
        "openmcp.setup.subprocess.run",
        lambda args, **kw: calls.append(args) or subprocess.CompletedProcess(args, 0, "", ""),
    )
    windows = Context(
        base=ctx.base, user_home=ctx.user_home, platform="win32", command=("C:/venv/Scripts/openmcp.exe",),
        which=lambda name: "C:/npm/claude.cmd",
    )
    assert run(plan("claude-code", windows)) == []
    assert [c[0] for c in calls] == ["C:/npm/claude.cmd"] * 2
    assert calls[1][-4:] == ["C:/venv/Scripts/openmcp.exe", "serve", "--client", "claude-code"]


def test_vscode_discovery_off_per_application(ctx):
    """Newer VS Code keys discovery by application; only Claude Desktop's is turned off."""
    settings = vscode_settings(ctx)
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({VSCODE_DISCOVERY: {"claude-desktop": True, "cursor-global": True}}))
    run(plan("vscode-copilot", ctx))
    assert json.loads(settings.read_text())[VSCODE_DISCOVERY] == {"claude-desktop": False, "cursor-global": True}


# --- development checkouts -----------------------------------------------------------


def test_tests_run_from_a_development_checkout():
    from conftest import REAL_DEVELOPMENT_CHECKOUT
    root = REAL_DEVELOPMENT_CHECKOUT()
    assert root is not None and (root / "src" / "openmcp" / "setup.py").exists()


def make_exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def test_installed_copy_prefers_uv_tools_and_skips_the_checkout(tmp_path):
    checkout = tmp_path / "repo"
    dev = make_exe(checkout / ".venv" / "bin" / "openmcp")
    tool = make_exe(tmp_path / "tools" / "openmcp" / "bin" / "openmcp")
    on_path = make_exe(tmp_path / "bin" / "openmcp")
    uv_dir = lambda *a, **k: subprocess.CompletedProcess(a, 0, str(tmp_path / "tools") + "\n", "")  # noqa: E731
    path = f"{dev.parent}{os.pathsep}{on_path.parent}"

    assert installed_command(checkout, path=path, which=lambda _: "uv", run=uv_dir) == (str(tool.resolve()),)
    assert installed_command(checkout, path=path, which=lambda _: None) == (str(on_path.resolve()),)
    assert installed_command(checkout, path=str(dev.parent), which=lambda _: None) is None
