"""The MCP server: one instance per `openmcp serve --client <name>` process."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from openmcp import __version__, audit, config, db
from openmcp.models import PingResult

INSTRUCTIONS = (
    "OpenMCP is the user's local, user-owned context store, shared across all of their "
    "AI assistants."
)


def build_server(client: str, base: Path) -> MCPServer:
    """Build the server for one client. `base` is the data directory (already initialized)."""
    path = config.db_path(base)
    db.open_db(path).close()
    scopes = config.resolve_client(config.load_config(config.config_path(base)), client)

    @contextmanager
    def tool_call(tool: str) -> Iterator[tuple[sqlite3.Connection, audit.AuditCall]]:
        # Sync tools run in worker threads and sqlite3 connections are bound to the thread
        # that created them, so each call opens its own short-lived connection.
        with closing(db.connect(path)) as conn, audit.audited(conn, client, tool) as call:
            yield conn, call

    server = MCPServer(name="openmcp", version=__version__, instructions=INSTRUCTIONS)

    @server.tool()
    def ping() -> PingResult:
        """Check that the OpenMCP server is running and reachable. Returns the server
        version, the name this client connected as, its scopes, and the schema version."""
        with tool_call("ping") as (conn, _):
            return PingResult(
                ok=True,
                version=__version__,
                client=client,
                read_scopes=list(scopes.read_scopes),
                write_scope=scopes.write_scope,
                schema_version=db.schema_version(conn),
            )

    return server
