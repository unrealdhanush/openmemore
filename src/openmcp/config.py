"""Data directory paths, config.toml loading/writing, and client scope resolution."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

import tomli_w
from pydantic import BaseModel, ConfigDict, Field

from openmcp.models import CaptureSource, Scope

DEFAULT_CONFIG: dict[str, Any] = {
    "ignore_scopes": False,
    "defaults": {"read_scopes": ["personal"], "write_scope": "personal"},
    "capture": {"sources": [], "days": 30},
}

CONFIG_HEADER = """\
# OpenMCP configuration.
#
# Scopes are labels on every memory and conversation (e.g. personal, work). Each client
# reads some and writes one, so an assistant can be kept away from part of your data.
# Set ignore_scopes = true to share everything with every client: scopes stay as labels
# but nothing is hidden or locked.
#
# A client is named by what its application reports when it connects ('Some Editor 2'
# -> some-editor-2); `openmcp clients` lists the ones that have connected. Clients
# without a [clients.<name>] table use [defaults]. For example:
#
#   [clients.cursor-vscode]
#   read_scopes = ["work", "personal"]
#   write_scope = "work"
#
# Transcript capture can be limited to some project folders, for apps whose transcripts
# say which folder a chat was in (Claude Code, VS Code). Exclusions win:
#
#   [capture]
#   include_folders = ["~/projects"]
#   exclude_folders = ["~/projects/client-work"]

"""


class ClientScopes(BaseModel):
    """The scopes one client may read from and write to."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    read_scopes: tuple[Scope, ...] | None
    """None: every scope (ignore_scopes), and existing items in any scope can be changed."""
    write_scope: Scope
    """The label new items get."""


class ClientOverride(BaseModel):
    """A [clients.<name>] table; any key left out falls back to [defaults]."""

    model_config = ConfigDict(extra="forbid")

    read_scopes: tuple[Scope, ...] | None = None
    write_scope: Scope | None = None


class CaptureConfig(BaseModel):
    """[capture]: which clients' transcripts are read from disk (SPEC.md M1.5). Off until
    the user enables a source, normally through `openmcp setup <client>`."""

    model_config = ConfigDict(extra="forbid")

    sources: tuple[CaptureSource, ...] = ()
    days: int = Field(default=30, ge=1, le=3650)
    """Only transcripts changed within this many days are read."""
    include_folders: tuple[str, ...] = ()
    """If set, only conversations from these project folders (or inside them) are captured."""
    exclude_folders: tuple[str, ...] = ()
    """Conversations from these project folders (or inside them) are never captured."""


DEFAULT_DAEMON_PORT = 47318


class DaemonConfig(BaseModel):
    """[daemon]: the background service (SPEC.md M6)."""

    model_config = ConfigDict(extra="forbid")

    port: int = Field(default=DEFAULT_DAEMON_PORT, ge=1024, le=65535)
    """Where it listens on 127.0.0.1, for OpenTelemetry exports and health checks."""
    token: str | None = None
    """Exporters must send it (x-openmcp-token header); created on first use."""
    interval: float = Field(default=3.0, ge=0.5, le=3600)
    """Seconds between checks for changed transcripts."""


class ConsolidationConfig(BaseModel):
    """[consolidation]: automatic consolidation with a model OpenMCP calls itself
    (SPEC.md M7). Off until the user sets it up with `openmcp consolidate setup`."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    endpoint: str = ""
    """Base URL of an Open Responses (or Chat Completions) endpoint, e.g. http://127.0.0.1:11434/v1."""
    model: str = ""
    api: Literal["responses", "chat-completions"] = "responses"
    schedule: str = Field(default="03:00", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    """Local time of day for the daemon's daily run."""
    max_conversations: int = Field(default=50, ge=1, le=10_000)
    """Conversations per run; the rest wait for the next."""
    allowed_host: str | None = None
    """The non-local host the user allowed conversations to be sent to (set when asked)."""


class Config(BaseModel):
    """config.toml. Unknown keys are rejected so typos (e.g. `read_scope`) don't pass silently."""

    model_config = ConfigDict(extra="forbid")

    ignore_scopes: bool = False
    """Share everything with every client: scopes stay as labels on new items, but every
    client reads and changes items in any scope."""
    defaults: ClientScopes = ClientScopes(read_scopes=("personal",), write_scope="personal")
    clients: dict[str, ClientOverride] = Field(default_factory=dict)
    capture: CaptureConfig = CaptureConfig()
    daemon: DaemonConfig = DaemonConfig()
    consolidation: ConsolidationConfig = ConsolidationConfig()


def home() -> Path:
    """The data directory: $OPENMCP_HOME if set, else ~/.openmcp."""
    env = os.environ.get("OPENMCP_HOME")
    return Path(env).expanduser() if env else Path.home() / ".openmcp"


def db_path(base: Path | None = None) -> Path:
    return (base or home()) / "openmcp.db"


def config_path(base: Path | None = None) -> Path:
    return (base or home()) / "config.toml"


def write_default_config(path: Path) -> bool:
    """Write the default config if none exists. Returns True if a file was written."""
    if path.exists():
        return False
    path.write_text(CONFIG_HEADER + tomli_w.dumps(DEFAULT_CONFIG), encoding="utf-8")
    return True


def load_config(path: Path) -> Config:
    """Load and validate config.toml, falling back to the built-in defaults if it is missing."""
    if not path.exists():
        return Config.model_validate(DEFAULT_CONFIG)
    with path.open("rb") as f:
        return Config.model_validate(tomllib.load(f))


def save_config(path: Path, config: Config) -> None:
    data = config.model_dump(mode="json", exclude_none=True)
    if not data.get("clients"):
        data.pop("clients", None)  # an empty [clients] table reads as if something were missing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CONFIG_HEADER + tomli_w.dumps(data), encoding="utf-8")


def enable_capture(path: Path, source: CaptureSource) -> bool:
    """Turn on transcript capture for a source. Returns False if it was already on."""
    cfg = load_config(path)
    if source in cfg.capture.sources:
        return False
    cfg.capture.sources = (*cfg.capture.sources, source)
    save_config(path, cfg)
    return True


def resolve_client(config: Config, client: str) -> ClientScopes:
    """Scopes for a client name; unknown clients (or missing keys) fall back to [defaults]."""
    override = config.clients.get(client, ClientOverride())
    if config.ignore_scopes:
        read_scopes = None
    else:
        read_scopes = override.read_scopes if override.read_scopes is not None else config.defaults.read_scopes
    return ClientScopes(
        read_scopes=read_scopes,
        write_scope=override.write_scope if override.write_scope is not None else config.defaults.write_scope,
    )
