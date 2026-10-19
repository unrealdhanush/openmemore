"""Naming clients (SPEC.md M3.5).

OpenMCP keeps no table of MCP clients. A client is named by what its application
reports when the session starts (MCP clientInfo), so any MCP client works without
OpenMCP knowing about it: 'Some Editor 2' becomes the client 'some-editor-2', and
[clients.some-editor-2] in config.toml gives it scopes. To use another name, pin it:
`app = "Some Editor 2"` under [clients.<name>]. (`--client` still overrides, for
configs written before this.) `openmcp clients` lists the applications seen so far.
"""

from __future__ import annotations

import re

_NOT_NAME = re.compile(r"[^a-z0-9._-]+")


def slug(app: str) -> str:
    """A client name from an application's name: 'Some Editor 2' -> 'some-editor-2'."""
    name = _NOT_NAME.sub("-", app.casefold()).strip("-._")[:64]
    return name or "mcp-client"


def client_for_app(app: str | None, pins: dict[str, str]) -> str:
    """The client name for an application: the user's pin ({app: client}, from
    [clients.<name>].app) if there is one, else one made from the application's name."""
    if not app:
        return "mcp-client"
    return pins.get(app) or slug(app)
