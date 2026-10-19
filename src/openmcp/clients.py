"""The client registry (SPEC.md M3.5): what OpenMCP knows about particular MCP clients.

No entry is needed to use OpenMCP (Tier 0): any MCP client can run `openmcp serve`
from one generic snippet, and the server names it from the application name it sends
when the session starts (MCP clientInfo). An entry adds what's specific to a client
(Tier 1): where its config lives, where its standing instruction goes, its clientInfo
name, and whether its transcripts can be captured (Tier 2).

`verified` records what was checked on a real install. Setup writes files only for
verified entries unless asked to try an unverified one (it keeps a .bak), because a
wrong guess about another app's config could break it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from openmcp.models import CaptureSource
from openmcp.paths import app_data_dir

PathFn = Callable[[Path, str], Path]
"""(user home, platform) -> a path."""


def home_path(*parts: str) -> PathFn:
    return lambda home, platform: home.joinpath(*parts)


def app_data_path(*parts: str) -> PathFn:
    return lambda home, platform: app_data_dir(home, platform).joinpath(*parts)


@dataclass(frozen=True)
class ServerConfig:
    """Where a client reads MCP servers from, and the shape of an entry."""

    path: PathFn
    key: tuple[str, ...] = ("mcpServers",)
    """Where entries go: {"mcpServers": {"openmcp": {...}}} for most clients."""
    extra: dict[str, Any] = field(default_factory=dict)
    """Added to each entry, e.g. {"type": "stdio"}."""
    format: Literal["json", "toml"] = "json"
    indent: int = 2


@dataclass(frozen=True)
class InstructionTarget:
    """Where the standing instruction goes."""

    kind: Literal["paste", "markdown", "json-list"]
    where: str
    """For people: where to paste it, or what the file or setting is."""
    path: PathFn | None = None
    key: str | None = None
    """The setting, for json-list."""
    full: bool = False
    """Full guidance (only where setup writes it, so it refreshes on every run) or the
    stable pointer (where the user pastes it)."""


@dataclass(frozen=True)
class ClientSpec:
    name: str
    """The client name: `--client`, [clients.<name>] in config, `openmcp setup <name>`."""
    display: str
    app: str | None
    """What the application calls itself over MCP (clientInfo.name), once confirmed."""
    instruction: InstructionTarget
    server: ServerConfig | None = None
    """None when the client registers servers another way (see setup's special cases)."""
    capture: CaptureSource | None = None
    verified: str | None = None
    """What was checked on a real install, when; None until someone has."""
    restart: str = "Restart the app so it picks up the new server."


CLIENTS: dict[str, ClientSpec] = {spec.name: spec for spec in [
    ClientSpec(
        name="claude-desktop",
        display="Claude Desktop",
        app="claude-ai",
        server=ServerConfig(app_data_path("Claude", "claude_desktop_config.json")),
        instruction=InstructionTarget(
            "paste", "Claude Desktop: Settings → Profile → personal preferences (kept in your account)"
        ),
        verified="macOS, 2026-10-18: config, clientInfo 'claude-ai', instruction followed",
        restart="Quit Claude Desktop completely (macOS: ⌘Q; Windows: right-click its tray icon → Quit) and reopen it.",
    ),
    ClientSpec(
        name="vscode-copilot",
        display="VS Code (Copilot Chat)",
        app="Visual Studio Code",
        instruction=InstructionTarget(
            "json-list", "Copilot's custom instructions in VS Code settings",
            key="github.copilot.chat.codeGeneration.instructions", full=True,
        ),
        capture="vscode-copilot",
        verified="VS Code 1.99.3 on macOS, 2026-10-18: config, discovery off, clientInfo, transcript capture",
        restart="Run 'Developer: Reload Window', then use Copilot Chat in Agent mode with the openmcp tools enabled.",
    ),
    ClientSpec(
        name="claude-code",
        display="Claude Code",
        app="claude-code",
        instruction=InstructionTarget("markdown", "~/.claude/CLAUDE.md", path=home_path(".claude", "CLAUDE.md"), full=True),
        capture="claude-code",
        verified="macOS, 2026-10-18: transcript format, on real sessions (`openmcp setup claude-code` itself "
        "not yet run on a real install)",
        restart="Start a new Claude Code session: running ones don't pick up new MCP servers.",
    ),
    ClientSpec(
        name="cursor",
        display="Cursor",
        # Not its product name ('Cursor'): what its agent reported in the audit log.
        app="cursor-vscode",
        server=ServerConfig(home_path(".cursor", "mcp.json")),
        instruction=InstructionTarget("paste", "Cursor Settings → Rules → User Rules"),
        verified="Cursor 3.25.9 on macOS, 2026-10-18: ~/.cursor/mcp.json, clientInfo 'cursor-vscode', pointer "
        "instruction followed (get_profile, record_exchange reusing the conversation, topic search, lookups "
        "not recorded)",
    ),
    ClientSpec(
        name="windsurf",
        display="Windsurf",
        app=None,
        server=ServerConfig(home_path(".codeium", "windsurf", "mcp_config.json")),
        instruction=InstructionTarget("paste", "Windsurf Settings → Cascade → global rules"),
    ),
    ClientSpec(
        name="gemini-cli",
        display="Gemini CLI",
        app=None,
        server=ServerConfig(home_path(".gemini", "settings.json")),
        instruction=InstructionTarget("markdown", "~/.gemini/GEMINI.md", path=home_path(".gemini", "GEMINI.md"), full=True),
        restart="Start a new Gemini CLI session.",
    ),
    ClientSpec(
        name="cline",
        display="Cline (VS Code extension)",
        app=None,
        server=ServerConfig(app_data_path("Code", "User", "globalStorage", "saoudrizwan.claude-dev", "settings",
                                          "cline_mcp_settings.json")),
        instruction=InstructionTarget("paste", "Cline → Rules (global)"),
        restart="Reload VS Code.",
    ),
    ClientSpec(
        name="lm-studio",
        display="LM Studio",
        app=None,
        server=ServerConfig(home_path(".lmstudio", "mcp.json")),
        instruction=InstructionTarget("paste", "LM Studio: the system prompt of your chat or preset"),
    ),
    ClientSpec(
        name="codex-cli",
        display="Codex CLI",
        app=None,
        server=ServerConfig(home_path(".codex", "config.toml"), key=("mcp_servers",), format="toml"),
        instruction=InstructionTarget("markdown", "~/.codex/AGENTS.md", path=home_path(".codex", "AGENTS.md"), full=True),
        restart="Start a new Codex session.",
    ),
]}


_NOT_NAME = re.compile(r"[^a-z0-9._-]+")


def slug(app: str) -> str:
    """A client name for an application with no entry: 'Some Editor 2' -> 'some-editor-2'."""
    name = _NOT_NAME.sub("-", app.casefold()).strip("-._")[:64]
    return name or "mcp-client"


def client_for_app(app: str | None, pins: dict[str, str]) -> str:
    """The client name for an application, from what it reports over MCP: the user's
    config pins first ({app: client}, from [clients.<name>].app), then the registry,
    then a name derived from the application's own."""
    if not app:
        return "mcp-client"
    if app in pins:
        return pins[app]
    for spec in CLIENTS.values():
        if spec.app == app:
            return spec.name
    return slug(app)
