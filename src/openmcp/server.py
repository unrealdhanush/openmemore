"""The MCP server: one instance per `openmcp serve --client <name>` process."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from openmcp import __version__, config, db

INSTRUCTIONS = (
    "OpenMCP is the user's local, user-owned context store, shared across all of their "
    "AI assistants."
)


def build_server(client: str, base: Path) -> MCPServer:
    """Build the server for one client. `base` is the data directory (already initialized)."""
    path = config.db_path(base)
    db.open_db(path).close()
    scopes = config.resolve_client(config.load_config(config.config_path(base)), client)

    def connect() -> closing[sqlite3.Connection]:
        # Sync tools run in worker threads and sqlite3 connections are bound to the thread
        # that created them, so each call opens its own short-lived connection.
        return closing(db.connect(path))

    server = MCPServer(name="openmcp", version=__version__, instructions=INSTRUCTIONS)

    @server.tool()
    def ping() -> dict[str, Any]:
        """Check that the OpenMCP server is running and reachable. Returns the server
        version, the name this client connected as, and the schema version."""
        with connect() as conn:
            version = db.schema_version(conn)
        return {
            "ok": True,
            "version": __version__,
            "client": client,
            "read_scopes": list(scopes.read_scopes),
            "write_scope": scopes.write_scope,
            "schema_version": version,
        }

    return server
