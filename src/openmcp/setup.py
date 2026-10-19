"""`openmcp setup <client>`: connect a client to OpenMCP and install its standing instruction.

Each client gets a list of steps. Steps with `apply` change a file or run a command;
steps with only `manual` are things the user has to do in the app, printed for them.
File edits keep a `.bak` copy and refuse to touch a file they can't parse.

Verified clients: Claude Desktop, VS Code Copilot Chat (1.99), Claude Code. Others get
printed instructions only, until their config formats are checked on a real install.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openmcp
from openmcp import config
from openmcp.capture import vscode_copilot
from openmcp.instructions import STANDING_PREFIX, standing_instruction
from openmcp.models import CaptureSource
from openmcp.paths import app_data_dir

SERVER_NAME = "openmcp"
BLOCK_START = "<!-- openmcp:start -->"
BLOCK_END = "<!-- openmcp:end -->"
COPILOT_INSTRUCTIONS = "github.copilot.chat.codeGeneration.instructions"

# How each application names itself over MCP (clientInfo.name), read from the apps
# themselves: VS Code sends product.json's nameLong; Claude Code's identity is bundled in
# Claude.app. Claude Desktop's own name couldn't be confirmed, so it isn't pinned (VS
# Code still can't pass as claude-desktop: it's pinned to vscode-copilot).
APP_NAMES = {"vscode-copilot": "Visual Studio Code", "claude-code": "claude-code"}
VSCODE_DISCOVERY = "chat.mcp.discovery.enabled"


class SetupError(Exception):
    """A step that can't be done automatically; the message says what to do by hand."""


@dataclass
class Step:
    description: str
    apply: Callable[[], str | None] | None = None
    """Does the step; may return a detail to show."""
    manual: str | None = None
    """What the user does by hand (shown for every step that has it)."""


@dataclass(frozen=True)
class Context:
    base: Path
    """OpenMCP data directory."""
    user_home: Path
    platform: str
    command: tuple[str, ...]
    """How to run `openmcp`, e.g. ('/path/.venv/bin/openmcp',)."""
    which: Callable[[str], str | None] = shutil.which


def openmcp_command() -> tuple[str, ...]:
    """The openmcp executable next to the running interpreter (the venv it is installed in)."""
    candidate = Path(sys.executable).with_name("openmcp.exe" if sys.platform == "win32" else "openmcp")
    if candidate.exists():
        return (str(candidate),)
    found = shutil.which("openmcp")
    if found:
        return (found,)
    return (sys.executable, "-m", "openmcp.cli")


def development_checkout() -> Path | None:
    """The source checkout this openmcp runs from, if it's a development (editable)
    install rather than an installed copy."""
    package = Path(openmcp.__file__).resolve().parent
    if "site-packages" in package.parts:
        return None
    return next((p for p in package.parents if (p / "pyproject.toml").is_file()), None)


def installed_command(
    checkout: Path,
    *,
    path: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> tuple[str, ...] | None:
    """An installed openmcp outside `checkout`: a `uv tool install` first, then any
    `openmcp` on PATH. None if there isn't one."""
    exe = "openmcp.exe" if os.name == "nt" else "openmcp"
    candidates: list[Path] = []
    if which("uv"):
        done = run(["uv", "tool", "dir"], capture_output=True, text=True)
        if done.returncode == 0 and done.stdout.strip():
            tool = Path(done.stdout.strip()) / "openmcp"
            candidates += [tool / "bin" / exe, tool / "Scripts" / exe]
    for directory in (os.environ.get("PATH", "") if path is None else path).split(os.pathsep):
        if directory:
            candidates.append(Path(directory) / exe)
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if checkout not in resolved.parents and os.access(resolved, os.X_OK):
            return (str(resolved),)
    return None


def server_entry(ctx: Context, client: str) -> dict[str, Any]:
    """The {command, args, env} block most clients use for a stdio MCP server."""
    entry: dict[str, Any] = {"command": ctx.command[0], "args": [*ctx.command[1:], "serve", "--client", client]}
    if ctx.base != ctx.user_home / ".openmcp":
        entry["env"] = {"OPENMCP_HOME": str(ctx.base)}
    return entry


# --- file helpers --------------------------------------------------------------------


def _edit_json(path: Path, change: Callable[[dict[str, Any]], None], hint: str, indent: int = 4) -> str:
    """Load a JSON object (or start one), apply `change`, and write it back with a backup."""
    if path.exists():
        text = path.read_text(encoding="utf-8-sig")  # tolerate a BOM (common on Windows)
        try:
            data = json.loads(text) if text.strip() else {}
        except ValueError:
            # VS Code allows comments and trailing commas; rewriting would drop them.
            raise SetupError(f"{path} isn't plain JSON (comments?), so it wasn't changed. Add this by hand:\n{hint}")
        if not isinstance(data, dict):
            raise SetupError(f"{path} doesn't hold a JSON object, so it wasn't changed. Add this by hand:\n{hint}")
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    else:
        data = {}
        path.parent.mkdir(parents=True, exist_ok=True)
    change(data)
    path.write_text(json.dumps(data, indent=indent) + "\n", encoding="utf-8")
    return str(path)


def _set_block(path: Path, text: str) -> str:
    """Put `text` between OpenMCP markers in a Markdown file, replacing an earlier block."""
    block = f"{BLOCK_START}\n{text}\n{BLOCK_END}"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    pattern = re.compile(re.escape(BLOCK_START) + r".*?" + re.escape(BLOCK_END), re.DOTALL)
    if pattern.search(existing):
        updated = pattern.sub(lambda _: block, existing)
    else:
        updated = existing.rstrip("\n") + ("\n\n" if existing.strip() else "") + block + "\n"
    if existing:
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8")
    return str(path)


def _pin_app(ctx: Context, client: str) -> Step:
    app = APP_NAMES[client]

    def apply() -> str:
        config.set_client_app(config.config_path(ctx.base), client, app)
        return str(config.config_path(ctx.base))

    return Step(
        f"Record in config.toml that client name '{client}' is for '{app}', so another app using it is flagged",
        apply=apply,
    )


def _enable_capture(ctx: Context, source: CaptureSource) -> Step:
    def apply() -> str:
        config.enable_capture(config.config_path(ctx.base), source)
        return str(config.config_path(ctx.base))

    return Step(
        f"Turn on transcript capture for {source} in config.toml, so its conversations are stored "
        "without the model calling record_exchange",
        apply=apply,
    )


# --- clients -------------------------------------------------------------------------


def claude_desktop_config(ctx: Context) -> Path:
    # Linux has no official Claude Desktop; community builds use the same layout.
    return app_data_dir(ctx.user_home, ctx.platform) / "Claude" / "claude_desktop_config.json"


def _claude_desktop(ctx: Context) -> list[Step]:
    path = claude_desktop_config(ctx)
    entry = server_entry(ctx, "claude-desktop")

    def add_server(data: dict[str, Any]) -> None:
        # Top level: a block nested anywhere else (e.g. inside "preferences") is ignored.
        data.setdefault("mcpServers", {})[SERVER_NAME] = entry

    return [
        Step(
            f"Add the openmcp server to {path}",
            apply=lambda: _edit_json(
                path, add_server, json.dumps({"mcpServers": {SERVER_NAME: entry}}, indent=2), indent=2
            ),
        ),
        Step(
            "Add the standing instruction to Claude Desktop",
            manual="In Claude Desktop, open Settings → Profile and paste this into your personal preferences "
            "(replacing any earlier OpenMCP instruction):\n\n" + standing_instruction(captured=False),
        ),
        Step(
            "Restart Claude Desktop",
            manual="Quit Claude Desktop completely (macOS: ⌘Q; Windows: right-click its tray icon → Quit) "
            "and reopen it.",
        ),
    ]


def _vscode_copilot(ctx: Context) -> list[Step]:
    user_dir = vscode_copilot.user_dir(ctx.user_home, ctx.platform)
    settings = user_dir / "settings.json"
    mcp_json = user_dir / "mcp.json"  # VS Code 1.102+ keeps MCP servers here instead of settings.json
    entry = {"type": "stdio", **server_entry(ctx, "vscode-copilot")}
    instruction = standing_instruction(captured=True)

    def add_server_to_settings(data: dict[str, Any]) -> None:
        data.setdefault("mcp", {}).setdefault("servers", {})[SERVER_NAME] = entry
        data.setdefault("chat.mcp.enabled", True)
        data.setdefault("chat.agent.enabled", True)

    def add_server_to_mcp_json(data: dict[str, Any]) -> None:
        data.setdefault("servers", {})[SERVER_NAME] = entry

    def no_discovery(data: dict[str, Any]) -> None:
        current = data.get(VSCODE_DISCOVERY)
        if isinstance(current, dict):  # per-application in newer VS Code
            current["claude-desktop"] = False
        else:
            data[VSCODE_DISCOVERY] = False

    def add_instruction(data: dict[str, Any]) -> None:
        existing = data.get(COPILOT_INSTRUCTIONS)
        items = existing if isinstance(existing, list) else []
        kept = [i for i in items if not (isinstance(i, dict) and str(i.get("text", "")).startswith(STANDING_PREFIX))]
        data[COPILOT_INSTRUCTIONS] = [*kept, {"text": instruction}]

    if mcp_json.exists():
        server_step = Step(
            f"Add the openmcp server to {mcp_json}",
            apply=lambda: _edit_json(mcp_json, add_server_to_mcp_json, json.dumps({"servers": {SERVER_NAME: entry}}, indent=4)),
        )
    else:
        server_step = Step(
            f"Add the openmcp server to {settings}",
            apply=lambda: _edit_json(
                settings, add_server_to_settings, json.dumps({"mcp": {"servers": {SERVER_NAME: entry}}}, indent=4)
            ),
        )
    return [
        server_step,
        Step(
            "Stop VS Code from also starting MCP servers configured for Claude Desktop "
            f"({VSCODE_DISCOVERY}): Copilot would run as claude-desktop, with its scopes and tools",
            apply=lambda: _edit_json(settings, no_discovery, json.dumps({VSCODE_DISCOVERY: False}, indent=4)),
        ),
        Step(
            f"Add the standing instruction to Copilot's custom instructions in {settings}",
            apply=lambda: _edit_json(
                settings, add_instruction, json.dumps({COPILOT_INSTRUCTIONS: [{"text": instruction}]}, indent=4)
            ),
        ),
        _pin_app(ctx, "vscode-copilot"),
        _enable_capture(ctx, "vscode-copilot"),
        Step(
            "Reload VS Code",
            manual="Run 'Developer: Reload Window', then use Copilot Chat in Agent mode with the openmcp tools enabled.",
        ),
    ]


def _claude_code(ctx: Context) -> list[Step]:
    entry = server_entry(ctx, "claude-code")
    env = [f"--env={k}={v}" for k, v in entry.get("env", {}).items()]
    add = ["claude", "mcp", "add", "--scope", "user", *env, SERVER_NAME, "--", entry["command"], *entry["args"]]
    memory = ctx.user_home / ".claude" / "CLAUDE.md"

    def register() -> str:
        # The resolved path, not "claude": on Windows an npm install is claude.cmd, which
        # subprocess can't start by bare name.
        claude = ctx.which("claude")
        if claude is None:
            raise SetupError("The claude command isn't on PATH. Run this once it is:\n" + " ".join(add))
        # Re-running setup replaces the registration instead of failing on a duplicate.
        subprocess.run([claude, "mcp", "remove", "--scope", "user", SERVER_NAME], capture_output=True)
        done = subprocess.run([claude, *add[1:]], capture_output=True, text=True)
        if done.returncode != 0:
            raise SetupError(f"claude mcp add failed: {done.stderr.strip() or done.stdout.strip()}")
        return " ".join(add)

    return [
        Step("Register the openmcp server with Claude Code (user scope)", apply=register),
        Step(
            f"Add the standing instruction to {memory}",
            apply=lambda: _set_block(memory, standing_instruction(captured=True)),
        ),
        _pin_app(ctx, "claude-code"),
        _enable_capture(ctx, "claude-code"),
        Step("Start a new Claude Code session", manual="Running sessions don't pick up new MCP servers."),
    ]


def _manual(ctx: Context, client: str) -> list[Step]:
    snippet = json.dumps({"mcpServers": {SERVER_NAME: server_entry(ctx, client)}}, indent=2)
    return [
        Step(
            f"Add the openmcp server to {client}",
            manual=f"openmcp setup doesn't configure {client} automatically yet. Most clients take an "
            f"MCP server block like this (Cursor: ~/.cursor/mcp.json; Gemini CLI: ~/.gemini/settings.json):\n\n{snippet}",
        ),
        Step(
            "Add the standing instruction",
            manual="Paste this where the client reads your own instructions (Cursor: User Rules; "
            "Gemini CLI: ~/.gemini/GEMINI.md):\n\n" + standing_instruction(captured=False),
        ),
    ]


AUTOMATIC = {"claude-desktop": _claude_desktop, "vscode-copilot": _vscode_copilot, "claude-code": _claude_code}


def plan(client: str, ctx: Context) -> list[Step]:
    build = AUTOMATIC.get(client)
    return build(ctx) if build else _manual(ctx, client)
