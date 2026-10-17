"""Data directory paths, config.toml loading/writing, and client scope resolution."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import BaseModel, ConfigDict, Field

from openmcp.models import Scope

DEFAULT_CONFIG: dict[str, Any] = {
    "defaults": {"read_scopes": ["personal"], "write_scope": "personal"},
    "clients": {
        "claude-desktop": {"read_scopes": ["personal", "work"], "write_scope": "personal"},
        "cursor": {"read_scopes": ["work"], "write_scope": "work"},
    },
}


class ClientScopes(BaseModel):
    """The scopes one client may read from and write to."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    read_scopes: tuple[Scope, ...]
    write_scope: Scope


class ClientOverride(BaseModel):
    """A [clients.<name>] table; any key left out falls back to [defaults]."""

    model_config = ConfigDict(extra="forbid")

    read_scopes: tuple[Scope, ...] | None = None
    write_scope: Scope | None = None


class Config(BaseModel):
    """config.toml. Unknown keys are rejected so typos (e.g. `read_scope`) don't pass silently."""

    model_config = ConfigDict(extra="forbid")

    defaults: ClientScopes = ClientScopes(read_scopes=("personal",), write_scope="personal")
    clients: dict[str, ClientOverride] = Field(default_factory=dict)


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
    path.write_text(tomli_w.dumps(DEFAULT_CONFIG))
    return True


def load_config(path: Path) -> Config:
    """Load and validate config.toml, falling back to the built-in defaults if it is missing."""
    if not path.exists():
        return Config.model_validate(DEFAULT_CONFIG)
    with path.open("rb") as f:
        return Config.model_validate(tomllib.load(f))


def resolve_client(config: Config, client: str) -> ClientScopes:
    """Scopes for a client name; unknown clients (or missing keys) fall back to [defaults]."""
    override = config.clients.get(client, ClientOverride())
    return ClientScopes(
        read_scopes=override.read_scopes if override.read_scopes is not None else config.defaults.read_scopes,
        write_scope=override.write_scope if override.write_scope is not None else config.defaults.write_scope,
    )
