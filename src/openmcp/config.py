"""Data directory paths, config.toml loading/writing, and client scope resolution."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli_w

DEFAULT_CONFIG: dict[str, Any] = {
    "defaults": {"read_scopes": ["personal"], "write_scope": "personal"},
    "clients": {
        "claude-desktop": {"read_scopes": ["personal", "work"], "write_scope": "personal"},
        "cursor": {"read_scopes": ["work"], "write_scope": "work"},
    },
}


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


def load_config(path: Path) -> dict[str, Any]:
    """Load config.toml, falling back to the built-in defaults if it is missing."""
    if not path.exists():
        return DEFAULT_CONFIG
    with path.open("rb") as f:
        return tomllib.load(f)


@dataclass(frozen=True)
class ClientScopes:
    read_scopes: tuple[str, ...]
    write_scope: str


def resolve_client(config: dict[str, Any], client: str) -> ClientScopes:
    """Scopes for a client name; unknown clients (or missing keys) fall back to [defaults]."""
    defaults = {**DEFAULT_CONFIG["defaults"], **config.get("defaults", {})}
    entry = {**defaults, **config.get("clients", {}).get(client, {})}
    return ClientScopes(read_scopes=tuple(entry["read_scopes"]), write_scope=entry["write_scope"])
