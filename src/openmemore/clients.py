"""Naming clients.

A client is named by what its application reports when the session starts (MCP
clientInfo), so any MCP client works without OpenMemore knowing about it: 'Some Editor 2'
becomes the client 'some-editor-2', and [clients.some-editor-2] in config.toml gives it
scopes. There's no table of clients and no way to rename one: `openmemore clients` lists
the names of the applications that have connected.
"""

from __future__ import annotations

import re

_NOT_NAME = re.compile(r"[^a-z0-9._-]+")


def client_name(app: str | None) -> str:
    """The client name for an application: 'Some Editor 2' -> 'some-editor-2'."""
    name = _NOT_NAME.sub("-", (app or "").casefold()).strip("-._")[:64]
    return name or "mcp-client"
