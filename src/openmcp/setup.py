"""`openmcp setup <client>`: connect a client to OpenMCP and install its standing instruction.

Steps come from the client registry (openmcp.clients): register the server, put the
standing instruction where the client reads it, record the client's app name, turn on
capture where it exists, restart. Steps with `apply` change a file or run a command;
steps with only `manual` are things the user does by hand, printed for them. File edits
keep a `.bak` copy and refuse to touch a file they can't parse. Entries not yet checked
on a real install are printed unless the user asks to try them; clients with no entry
get the generic snippet, which works in any MCP client.
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
from openmcp.clients import CLIENTS, ClientSpec, ServerConfig
from openmcp.instructions import STANDING_PREFIX, pointer_instruction, standing_instruction
from openmcp.models import CaptureSource
from openmcp.paths import app_data_dir

SERVER_NAME = "openmcp"
BLOCK_START = "<!-- openmcp:start -->"
BLOCK_END = "<!-- openmcp:end -->"
COPILOT_INSTRUCTIONS = "github.copilot.chat.codeGeneration.instructions"
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
    copy: Callable[[str], bool] | None = None
    """Puts text on the clipboard, returning whether it worked (default: copy_to_clipboard)."""


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


def server_entry(ctx: Context, client: str | None) -> dict[str, Any]:
    """The {command, args, env} block most clients use for a stdio MCP server. Without a
    client name, the server identifies the client from what it reports (SPEC.md M3.5)."""
    args = [*ctx.command[1:], "serve", *(["--client", client] if client else [])]
    entry: dict[str, Any] = {"command": ctx.command[0], "args": args}
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


def _edit_toml(path: Path, change: Callable[[dict[str, Any]], None], hint: str) -> str:
    """Like _edit_json, for TOML. A file with comments is left alone: rewriting drops them."""
    import tomllib

    import tomli_w

    if path.exists():
        text = path.read_text(encoding="utf-8")
        if any(line.lstrip().startswith("#") for line in text.splitlines()):
            raise SetupError(f"{path} has comments that rewriting would drop, so it wasn't changed. Add this by hand:\n{hint}")
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            raise SetupError(f"{path} isn't valid TOML, so it wasn't changed. Add this by hand:\n{hint}") from None
        shutil.copy2(path, path.with_name(path.name + ".bak"))
    else:
        data = {}
        path.parent.mkdir(parents=True, exist_ok=True)
    change(data)
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
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


def _nested(data: dict[str, Any], key: tuple[str, ...]) -> dict[str, Any]:
    """data[key[0]][key[1]]..., creating tables on the way."""
    for part in key:
        data = data.setdefault(part, {})
    return data


def _snippet(server: ServerConfig, entry: dict[str, Any]) -> str:
    if server.format == "toml":
        import tomli_w

        data: dict[str, Any] = {}
        _nested(data, server.key)[SERVER_NAME] = entry
        return tomli_w.dumps(data)
    wrapped: dict[str, Any] = {}
    _nested(wrapped, server.key)[SERVER_NAME] = entry
    return json.dumps(wrapped, indent=2)


def _pin_app(ctx: Context, client: str, app: str) -> Step:
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


def _unverified(spec: ClientSpec, what: str, path: Path, text: str) -> Step:
    return Step(
        f"{what} (not checked on a real {spec.display} install yet, so not written)",
        manual=f"Add this to {path}, or run `openmcp setup {spec.name} --try` to let setup write it (it keeps a "
        f".bak copy). Either way, please report whether it worked, so the entry can be marked verified.\n\n{text}",
    )


# --- steps from the registry -----------------------------------------------------------


def claude_desktop_config(ctx: Context) -> Path:
    # Linux has no official Claude Desktop; community builds use the same layout.
    server = CLIENTS["claude-desktop"].server
    assert server is not None
    return server.path(ctx.user_home, ctx.platform)


def _register(ctx: Context, spec: ClientSpec, try_unverified: bool) -> list[Step]:
    """Add the server entry to the client's MCP config file."""
    server = spec.server
    assert server is not None
    path = server.path(ctx.user_home, ctx.platform)
    entry = {**server.extra, **server_entry(ctx, spec.name)}
    snippet = _snippet(server, entry)
    if spec.verified is None and not try_unverified:
        return [_unverified(spec, f"Add the openmcp server to {path}", path, snippet)]

    def change(data: dict[str, Any]) -> None:
        # Exactly at the registry's key: a block nested anywhere else is ignored by the client.
        _nested(data, server.key)[SERVER_NAME] = entry

    if server.format == "toml":
        apply = lambda: _edit_toml(path, change, snippet)  # noqa: E731
    else:
        apply = lambda: _edit_json(path, change, snippet, indent=server.indent)  # noqa: E731
    return [Step(f"Add the openmcp server to {path}", apply=apply)]


def _instruction(ctx: Context, spec: ClientSpec, try_unverified: bool) -> list[Step]:
    target = spec.instruction
    text = standing_instruction(captured=spec.capture is not None) if target.full else pointer_instruction()
    if target.kind == "paste":
        return [
            _copy_step(ctx, text, spec.display),
            Step(
                f"Add the standing instruction to {spec.display}",
                manual=f"Paste it here: {target.where}, replacing any earlier OpenMCP instruction. It's short on "
                "purpose: the details come from the openmcp server, so it won't need replacing when they change."
                f"\n\n{text}",
            ),
        ]
    assert target.kind == "markdown" and target.path is not None
    path = target.path(ctx.user_home, ctx.platform)
    if spec.verified is None and not try_unverified:
        return [_unverified(spec, f"Add the standing instruction to {path}", path, f"{BLOCK_START}\n{text}\n{BLOCK_END}")]
    return [Step(f"Add the standing instruction to {path}", apply=lambda: _set_block(path, text))]


def _vscode_copilot(ctx: Context, spec: ClientSpec) -> list[Step]:
    """VS Code: servers live in settings.json or, from 1.102, mcp.json; its discovery of
    other apps' servers must be off (it ran Claude Desktop's entry as Copilot's); the
    instruction is a list setting."""
    user_dir = vscode_copilot.user_dir(ctx.user_home, ctx.platform)
    settings = user_dir / "settings.json"
    mcp_json = user_dir / "mcp.json"
    entry = {"type": "stdio", **server_entry(ctx, spec.name)}
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
    ]


def _claude_code(ctx: Context, spec: ClientSpec) -> list[Step]:
    """Claude Code registers servers through its own CLI, not a file setup should edit."""
    entry = server_entry(ctx, spec.name)
    env = [f"--env={k}={v}" for k, v in entry.get("env", {}).items()]
    add = ["claude", "mcp", "add", "--scope", "user", *env, SERVER_NAME, "--", entry["command"], *entry["args"]]

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

    return [Step("Register the openmcp server with Claude Code (user scope)", apply=register)]


def _generic(ctx: Context, client: str) -> list[Step]:
    """Tier 0: any MCP client, from one snippet that's the same for every client."""
    snippet = json.dumps({"mcpServers": {SERVER_NAME: server_entry(ctx, None)}}, indent=2)
    return [
        Step(
            f"Add the openmcp server to {client}",
            manual=f"{client} isn't in OpenMCP's client registry (`openmcp clients` lists it), which is fine: any MCP "
            "client works. Add this to its MCP server config; most clients use this shape. The server works "
            "out which client it's talking to from the name the client reports when it connects.\n\n" + snippet,
        ),
        _copy_step(ctx, pointer_instruction(), client),
        Step(
            "Add the standing instruction",
            manual="Paste it wherever the client reads your own instructions or rules:\n\n" + pointer_instruction(),
        ),
    ]


SPECIAL: dict[str, Callable[[Context, ClientSpec], list[Step]]] = {
    "vscode-copilot": _vscode_copilot,
    "claude-code": _claude_code,
}


def plan(client: str, ctx: Context, try_unverified: bool = False) -> list[Step]:
    spec = CLIENTS.get(client)
    if spec is None:
        return _generic(ctx, client)
    if spec.name in SPECIAL:
        steps = SPECIAL[spec.name](ctx, spec)
        if spec.instruction.kind != "json-list":
            steps += _instruction(ctx, spec, try_unverified)
    else:
        steps = _register(ctx, spec, try_unverified) + _instruction(ctx, spec, try_unverified)
    if spec.app:
        steps.append(_pin_app(ctx, spec.name, spec.app))
    if spec.capture:
        steps.append(_enable_capture(ctx, spec.capture))
    steps.append(Step(f"Restart {spec.display}", manual=spec.restart))
    return steps
