"""`openmcp setup [client]`: connect an MCP client to OpenMCP (SPEC.md M3.5).

Any MCP client works from one generic snippet: `openmcp serve`, with no client name,
since the server names the client from what it reports when it connects. Setup never
edits another application's files. It uses the vendor's own way of adding an MCP
server where there is one, detected when it runs rather than assumed:

- VS Code: `code --add-mcp <json>` (the CLI that ships with VS Code);
- Cursor: its install link, which opens Cursor's own review prompt;
- Claude Code: `claude mcp add`.

Everywhere else (Claude Desktop included) it prints the snippet; `--config <path>`
merges it into a JSON file the user names. The standing instruction is the short
pointer (openmcp.instructions), put on the clipboard to paste wherever the client
reads the user's own instructions. The only client-specific code that remains is
these installers and the transcript readers (openmcp.capture), and both are optional.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import openmcp
from openmcp import config
from openmcp.instructions import pointer_instruction
from openmcp.models import CAPTURE_SOURCES, CaptureSource

SERVER_NAME = "openmcp"


class SetupError(Exception):
    """A step that can't be done automatically; the message says what to do by hand."""


@dataclass
class Step:
    description: str
    apply: Callable[[], str | None] | None = None
    """Does the step; may return a detail to show."""
    manual: str | None = None
    """What the user does by hand (shown for every step that has it)."""
    note: str | None = None
    """A quiet aside, shown dimmed."""


@dataclass(frozen=True)
class Context:
    base: Path
    """OpenMCP data directory."""
    user_home: Path
    platform: str
    command: tuple[str, ...]
    """How to run `openmcp`, e.g. ('/path/.venv/bin/openmcp',)."""
    which: Callable[[str], str | None] = shutil.which
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run
    copy: Callable[[str], bool] | None = None
    """Puts text on the clipboard, returning whether it worked (default: copy_to_clipboard)."""
    open_link: Callable[[str], bool] | None = None
    """Opens a URL with the system's handler (default: open_link)."""


def openmcp_command() -> tuple[str, ...]:
    """The openmcp executable next to the running interpreter (the venv it is installed in)."""
    candidate = Path(sys.executable).with_name("openmcp.exe" if sys.platform == "win32" else "openmcp")
    if candidate.exists():
        return (str(candidate),)
    found = shutil.which("openmcp")
    if found:
        return (found,)
    return (sys.executable, "-m", "openmcp.cli")


def copy_to_clipboard(
    text: str,
    platform: str = sys.platform,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> bool:
    """Put `text` on the system clipboard with the platform's own tool. False if none works."""
    if platform == "darwin":
        tools = [["pbcopy"]]
    elif platform == "win32":
        tools = [["clip"]]
    else:
        tools = [["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]
    for tool in tools:
        if which(tool[0]) is None:
            continue
        try:
            if run(tool, input=text.encode(), capture_output=True, timeout=5).returncode == 0:
                return True
        except (OSError, subprocess.TimeoutExpired):
            continue
    return False


def _copy_step(ctx: Context, text: str, where: str) -> Step:
    """Copies the instruction for a paste step. Never fails setup: without a clipboard
    tool, the text printed with the next step is the way to get it."""

    def apply() -> str:
        copied = (ctx.copy or copy_to_clipboard)(text)
        return "copied" if copied else "no clipboard tool found: copy it from below"

    return Step(f"Copy the standing instruction to your clipboard, for {where}", apply=apply)


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


def open_link(url: str, platform: str = sys.platform, run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> bool:
    """Open a URL (including app links like cursor://) with the system's handler."""
    if platform == "darwin":
        command = ["open", url]
    elif platform == "win32":
        command = ["cmd", "/c", "start", "", url]
    else:
        command = ["xdg-open", url]
    try:
        return run(command, capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def server_entry(ctx: Context) -> dict[str, Any]:
    """The {command, args, env} block for `openmcp serve`, the same for every client."""
    entry: dict[str, Any] = {"command": ctx.command[0], "args": [*ctx.command[1:], "serve"]}
    if ctx.base != ctx.user_home / ".openmcp":
        entry["env"] = {"OPENMCP_HOME": str(ctx.base)}
    return entry


def snippet(ctx: Context) -> str:
    return json.dumps({"mcpServers": {SERVER_NAME: server_entry(ctx)}}, indent=2)


# --- vendor installers -----------------------------------------------------------------


def _vscode(ctx: Context) -> str:
    code = ctx.which("code")
    spec = json.dumps({"name": SERVER_NAME, **server_entry(ctx)})
    by_hand = f"code --add-mcp '{spec}'"
    if code is None:
        raise SetupError(
            "VS Code's `code` command isn't on PATH (in VS Code: 'Shell Command: Install code command in PATH'). "
            f"Then run:\n{by_hand}\nor add this to its MCP servers:\n{snippet(ctx)}"
        )
    help_text = ctx.run([code, "--help"], capture_output=True, text=True, timeout=60).stdout
    if "--add-mcp" not in help_text:
        raise SetupError(f"This VS Code can't add MCP servers from the command line; add this by hand:\n{snippet(ctx)}")
    done = ctx.run([code, "--add-mcp", spec], capture_output=True, text=True, timeout=60)
    if done.returncode != 0:
        raise SetupError(f"`code --add-mcp` failed: {(done.stderr or done.stdout).strip()}\nAdd this by hand:\n{snippet(ctx)}")
    return done.stdout.strip() or "added"


def cursor_link(ctx: Context) -> str:
    """Cursor's install link: its own extension handles it and asks the user to review."""
    config_b64 = base64.b64encode(json.dumps(server_entry(ctx)).encode()).decode()
    return f"cursor://anysphere.cursor-deeplink/mcp/install?name={SERVER_NAME}&config={quote(config_b64, safe='')}"


def _cursor(ctx: Context) -> str:
    link = cursor_link(ctx)
    if not (ctx.open_link or open_link)(link):
        raise SetupError(f"Couldn't open Cursor's install link. Open it yourself:\n{link}\nor add this to "
                         f"~/.cursor/mcp.json:\n{snippet(ctx)}")
    return "opened Cursor's install prompt: review and confirm it there"


def _claude_code(ctx: Context) -> str:
    entry = server_entry(ctx)
    env = [f"--env={k}={v}" for k, v in entry.get("env", {}).items()]
    add = ["claude", "mcp", "add", "--scope", "user", *env, SERVER_NAME, "--", entry["command"], *entry["args"]]
    # The resolved path, not "claude": on Windows an npm install is claude.cmd, which
    # subprocess can't start by bare name.
    claude = ctx.which("claude")
    if claude is None:
        raise SetupError("The claude command isn't on PATH. Run this once it is:\n" + " ".join(add))
    # Re-running setup replaces the registration instead of failing on a duplicate.
    ctx.run([claude, "mcp", "remove", "--scope", "user", SERVER_NAME], capture_output=True)
    done = ctx.run([claude, *add[1:]], capture_output=True, text=True)
    if done.returncode != 0:
        raise SetupError(f"claude mcp add failed: {(done.stderr or done.stdout).strip()}")
    return " ".join(add)


INSTALLERS: dict[str, tuple[str, Callable[[Context], str]]] = {
    "vscode-copilot": ("VS Code", _vscode),
    "cursor": ("Cursor", _cursor),
    "claude-code": ("Claude Code", _claude_code),
}
"""Clients with a vendor-maintained way to add an MCP server: (display name, installer)."""


# --- steps -----------------------------------------------------------------------------


def _merge_into(ctx: Context, path: Path) -> Step:
    """--config: merge the snippet into a JSON file the user named (with a .bak copy)."""

    def apply() -> str:
        if path.exists():
            text = path.read_text(encoding="utf-8-sig")  # tolerate a BOM (common on Windows)
            try:
                data = json.loads(text) if text.strip() else {}
            except ValueError:
                raise SetupError(f"{path} isn't plain JSON (comments?), so it wasn't changed. Add this by hand:\n"
                                 f"{snippet(ctx)}") from None
            if not isinstance(data, dict):
                raise SetupError(f"{path} doesn't hold a JSON object, so it wasn't changed.")
            shutil.copy2(path, path.with_name(path.name + ".bak"))
        else:
            data = {}
            path.parent.mkdir(parents=True, exist_ok=True)
        data.setdefault("mcpServers", {})[SERVER_NAME] = server_entry(ctx)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return str(path)

    return Step(f"Add the openmcp server to {path}", apply=apply)


def _enable_capture(ctx: Context, source: CaptureSource) -> Step:
    def apply() -> str:
        config.enable_capture(config.config_path(ctx.base), source)
        return str(config.config_path(ctx.base))

    return Step(
        f"Turn on transcript capture for {source} in config.toml, so its conversations are stored "
        "without the model calling record_exchange",
        apply=apply,
        note="Only some projects? List folders under include_folders or exclude_folders in [capture].",
    )


def _copy_pointer(ctx: Context, name: str, captured: bool) -> list[Step]:
    text = pointer_instruction(captured=captured)
    return [
        _copy_step(ctx, text, name),
        Step(
            "Add the standing instruction",
            manual=f"Paste it wherever {name} reads your own instructions (custom instructions, user rules, or a "
            "file like CLAUDE.md), replacing any earlier OpenMCP instruction. It's short on purpose: the details "
            "come from the openmcp server, so it won't need replacing when they change.\n\n" + text,
        ),
    ]


def plan(client: str | None, ctx: Context, config_file: Path | None = None) -> list[Step]:
    """Steps to connect `client` (any name; None for "your MCP client")."""
    installer = INSTALLERS.get(client or "")
    name = installer[0] if installer else (client or "your MCP client")
    captured = client in CAPTURE_SOURCES
    if config_file is not None:
        steps = [_merge_into(ctx, config_file)]
    elif installer:
        steps = [Step(f"Add the openmcp server to {name}, the way {name} provides", apply=lambda: installer[1](ctx))]
    else:
        steps = [Step(
            f"Add the openmcp server to {name}",
            manual="Add this to the client's MCP server config (most clients use this shape; Claude Desktop: "
            "Settings → Developer → Edit Config), or run setup again with --config <that file> to have it merged in. "
            "It's the same for every client: the server works out which one it's talking to.\n\n" + snippet(ctx),
        )]
    if captured:
        steps.append(_enable_capture(ctx, client))  # type: ignore[arg-type]
    steps += _copy_pointer(ctx, name, captured)
    steps.append(Step(f"Restart {name}", manual="Restart or reload it, so it starts the openmcp server."))
    return steps
