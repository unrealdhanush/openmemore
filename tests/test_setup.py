"""`openmemore setup`: the generic snippet, vendor installers, and --config. Nothing here
touches real apps: installers get fake commands, and files live in a temporary home."""

import base64
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import pytest

from openmemore import config
from openmemore.capture import vscode_copilot
from openmemore.cli import initialize
from openmemore.instructions import STANDING_PREFIX, pointer_instruction, server_instructions, standing_instruction
from openmemore.setup import Context, SetupError, cursor_link, installed_command, open_link, plan, server_entry

COMMAND = ("/opt/openmemore/bin/openmemore",)


class FakeRun:
    """Records commands; answers `--help` with the given text."""

    def __init__(self, help_text="", returncode=0, stdout="Added MCP servers: openmemore"):
        self.help_text, self.returncode, self.stdout, self.calls = help_text, returncode, stdout, []

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))
        out = self.help_text if "--help" in args else self.stdout
        return subprocess.CompletedProcess(args, 0 if "--help" in args else self.returncode, out, "")


@pytest.fixture
def ctx(tmp_path) -> Context:
    user_home = tmp_path / "user"
    base = user_home / ".openmemore"
    initialize(base)
    return Context(base=base, user_home=user_home, platform="darwin", command=COMMAND, which=lambda _: None)


def with_(ctx, **changes) -> Context:
    fields = {k: getattr(ctx, k) for k in ("base", "user_home", "platform", "command", "which", "run", "copy", "open_link")}
    return Context(**{**fields, **changes})


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


def written(ctx):
    return [p for p in ctx.user_home.rglob("*") if p.is_file() and ".openmemore" not in p.parts]


def manual(steps):
    return "\n".join(s.manual for s in steps if s.manual)


def json_block(text):
    return json.JSONDecoder().raw_decode(text[text.index("{"):])[0]


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


def test_the_pointer_names_the_key_calls_and_defers_to_the_server():
    text = pointer_instruction()
    assert text.startswith(STANDING_PREFIX) and "server's instructions" in text
    # The manual tests: Claude Desktop and Cursor followed details found only in the server
    # instructions, but didn't save a decision until the pointer named save_memories.
    for tool in ("get_profile", "record_exchange", "save_memories", "recent_conversations", "search_context"):
        assert tool in text
    assert len(text) < 450 and text.isascii()  # survives every clipboard tool, including Windows' clip
    server = server_instructions(captured=False)
    for detail in ("code", "conversation_id", "kind 'decision'", "topic itself", "secrets"):
        assert detail in server, detail


def test_the_captured_pointer_says_not_to_record():
    text = pointer_instruction(captured=True)
    assert "Don't call record_exchange" in text and "record_exchange after every answer" not in text


# --- any client --------------------------------------------------------------------------


def test_any_client_gets_the_generic_snippet_and_nothing_is_written(ctx, clipboard):
    steps = plan("some-new-editor", ctx)
    assert run(steps) == [] and written(ctx) == []
    assert json_block(manual(steps))["mcpServers"]["openmemore"] == {"command": COMMAND[0], "args": ["serve"]}
    assert clipboard == [pointer_instruction()]


def test_the_snippet_is_the_same_for_every_client():
    """No client name in it: the server names the client from what it reports."""
    default = Context(base=Path("/h/.openmemore"), user_home=Path("/h"), platform="linux", command=COMMAND)
    assert server_entry(default) == {"command": COMMAND[0], "args": ["serve"]}
    assert plan(None, default)[0].manual


def test_a_non_default_data_directory_is_passed_to_the_client(ctx, tmp_path):
    other = with_(ctx, base=tmp_path / "elsewhere")
    assert server_entry(other)["env"] == {"OPENMEMORE_HOME": str(tmp_path / "elsewhere")}


def test_config_option_merges_into_a_named_file(ctx):
    path = ctx.user_home / "some-app" / "mcp.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"theme": "Zoë", "mcpServers": {"other": {"command": "x"}}},
                                                  ensure_ascii=False).encode())  # with a BOM
    assert run(plan("some-app", ctx, config_file=path)) == []
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["theme"] == "Zoë" and data["mcpServers"]["other"] == {"command": "x"}
    assert data["mcpServers"]["openmemore"]["args"] == ["serve"]
    assert path.with_name("mcp.json.bak").exists()


def test_config_option_leaves_unparseable_files_alone(ctx):
    path = ctx.user_home / "settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{\n  // a comment\n  "a": 1,\n}\n')
    [error] = run(plan("x", ctx, config_file=path))
    assert "isn't plain JSON" in error and "// a comment" in path.read_text()


# --- vendor installers ---------------------------------------------------------------------


def test_vscode_uses_its_own_add_mcp_command(ctx, clipboard):
    fake = FakeRun(help_text="  --add-mcp <json>   Adds a Model Context Protocol server definition")
    vscode = with_(ctx, which=lambda name: "/usr/local/bin/code" if name == "code" else None, run=fake)
    assert run(plan("vscode-copilot", vscode)) == []
    add = fake.calls[-1]
    assert add[:2] == ["/usr/local/bin/code", "--add-mcp"]
    assert json.loads(add[2]) == {"name": "openmemore", "command": COMMAND[0], "args": ["serve"]}
    assert config.load_config(config.config_path(ctx.base)).capture.sources == ("vscode-copilot",)
    assert clipboard == [pointer_instruction(captured=True)]
    assert written(ctx) == []  # VS Code wrote its own settings; setup touched no file of VS Code's


def test_vscode_without_add_mcp_or_code_says_what_to_do(ctx):
    [error] = run(plan("vscode-copilot", ctx))  # no `code` on PATH
    assert "code --add-mcp" in error and "Install code command in PATH" in error
    old = with_(ctx, which=lambda name: "/usr/bin/code", run=FakeRun(help_text="--list-extensions"))
    [error] = run(plan("vscode-copilot", old))
    assert "can't add MCP servers from the command line" in error


def test_cursor_gets_its_own_install_link(ctx, clipboard):
    opened = []
    cursor = with_(ctx, open_link=lambda url: opened.append(url) or True)
    assert run(plan("cursor", cursor)) == []
    [url] = opened
    parsed = urlparse(url)
    assert (parsed.scheme, parsed.netloc, parsed.path) == ("cursor", "anysphere.cursor-deeplink", "/mcp/install")
    query = parse_qs(parsed.query)
    assert query["name"] == ["openmemore"]
    assert json.loads(base64.b64decode(unquote(query["config"][0]))) == {"command": COMMAND[0], "args": ["serve"]}
    assert clipboard == [pointer_instruction()]  # Cursor isn't captured: it records exchanges


def test_cursor_link_failure_prints_the_link(ctx):
    [error] = run(plan("cursor", with_(ctx, open_link=lambda url: False)))
    assert cursor_link(ctx) in error


def test_claude_code_uses_its_own_cli_by_resolved_path(ctx):
    """On Windows an npm install is claude.cmd, which subprocess can't start by bare name."""
    fake = FakeRun()
    windows = with_(ctx, platform="win32", command=("C:/venv/Scripts/openmemore.exe",),
                    which=lambda name: "C:/npm/claude.cmd", run=fake)
    assert run(plan("claude-code", windows)) == []
    assert [c[0] for c in fake.calls] == ["C:/npm/claude.cmd"] * 2  # remove, then add
    assert fake.calls[1][-3:] == ["--", "C:/venv/Scripts/openmemore.exe", "serve"]
    assert config.load_config(config.config_path(ctx.base)).capture.sources == ("claude-code",)


def test_claude_code_without_the_cli_prints_the_command(ctx):
    [error] = run(plan("claude-code", ctx))
    assert "claude mcp add --scope user openmemore -- /opt/openmemore/bin/openmemore serve" in error


@pytest.mark.parametrize("platform, expected", [
    ("darwin", ["open", "x://y"]), ("win32", ["cmd", "/c", "start", "", "x://y"]), ("linux", ["xdg-open", "x://y"]),
])
def test_open_link_uses_the_platform_opener(platform, expected):
    calls = []
    assert open_link("x://y", platform, run=lambda args, **kw: calls.append(args) or subprocess.CompletedProcess(args, 0))
    assert calls == [expected]


# --- operating systems ---------------------------------------------------------------------


@pytest.mark.parametrize("platform, env, expected", [
    ("darwin", {}, "Library/Application Support/Code/User"),
    ("win32", {"APPDATA": "Roaming-Redirected"}, "Roaming-Redirected/Code/User"),
    ("win32", {}, "AppData/Roaming/Code/User"),
    ("linux", {"XDG_CONFIG_HOME": "xdg"}, "xdg/Code/User"),
    ("linux", {}, ".config/Code/User"),
])
def test_app_data_paths_follow_each_os(tmp_path, monkeypatch, platform, env, expected):
    for var, value in env.items():
        monkeypatch.setenv(var, str(tmp_path / value))
    assert vscode_copilot.user_dir(tmp_path, platform).relative_to(tmp_path).as_posix() == expected


# --- development checkouts -----------------------------------------------------------------


def test_tests_run_from_a_development_checkout():
    from conftest import REAL_DEVELOPMENT_CHECKOUT
    root = REAL_DEVELOPMENT_CHECKOUT()
    assert root is not None and (root / "src" / "openmemore" / "setup.py").exists()


def make_exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def test_installed_copy_prefers_uv_tools_and_skips_the_checkout(tmp_path):
    # The layout uv and venvs use on this OS: Scripts\openmemore.exe on Windows.
    bin_dir, exe = ("Scripts", "openmemore.exe") if os.name == "nt" else ("bin", "openmemore")
    checkout = tmp_path / "repo"
    dev = make_exe(checkout / ".venv" / bin_dir / exe)
    tool = make_exe(tmp_path / "tools" / "openmemore" / bin_dir / exe)
    on_path = make_exe(tmp_path / "bin" / exe)
    uv_dir = lambda *a, **k: subprocess.CompletedProcess(a, 0, str(tmp_path / "tools") + "\n", "")  # noqa: E731
    path = f"{dev.parent}{os.pathsep}{on_path.parent}"

    assert installed_command(checkout, path=path, which=lambda _: "uv", run=uv_dir) == (str(tool.resolve()),)
    assert installed_command(checkout, path=path, which=lambda _: None) == (str(on_path.resolve()),)
    assert installed_command(checkout, path=str(dev.parent), which=lambda _: None) is None


# --- the clipboard ---------------------------------------------------------------------------


def test_no_clipboard_tool_doesnt_fail_setup(ctx):
    copy_step = next(s for s in plan("x", with_(ctx, copy=lambda text: False)) if s.description.startswith("Copy"))
    assert copy_step.apply() == "no clipboard tool found: copy it from below"


@pytest.mark.parametrize("platform, available, expected", [
    ("darwin", {"pbcopy"}, ["pbcopy"]),
    ("win32", {"clip"}, ["clip"]),
    ("linux", {"xclip", "xsel"}, ["xclip", "-selection", "clipboard"]),  # wl-copy missing: next one
    ("linux", set(), None),
])
def test_copy_to_clipboard_uses_the_platform_tool(platform, available, expected):
    from conftest import REAL_COPY_TO_CLIPBOARD as real_copy  # the module attribute is patched for safety
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs["input"]))
        return subprocess.CompletedProcess(args, 0)

    ok = real_copy("hello", platform=platform, which=lambda name: name if name in available else None, run=fake_run)
    assert ok is (expected is not None)
    assert calls == ([(expected, b"hello")] if expected else [])
