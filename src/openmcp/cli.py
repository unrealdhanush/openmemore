"""`openmcp` command-line interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer
from pydantic import TypeAdapter, ValidationError
from rich.console import Console

from openmcp import config, db
from openmcp.models import ClientName

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Local-first shared context store for AI assistants.")
console = Console()


@dataclass
class InitResult:
    base: Path
    config_written: bool
    migrations_applied: list[int]


def initialize(base: Path) -> InitResult:
    """Create the data directory, database, and default config. Safe to run repeatedly."""
    base.mkdir(parents=True, exist_ok=True)
    (base / "imports").mkdir(exist_ok=True)
    (base / "logs").mkdir(exist_ok=True)
    conn = db.connect(config.db_path(base))
    try:
        applied = db.migrate(conn)
    finally:
        conn.close()
    written = config.write_default_config(config.config_path(base))
    return InitResult(base=base, config_written=written, migrations_applied=applied)


@app.command()
def init() -> None:
    """Create the data directory, database, and default config."""
    result = initialize(config.home())
    console.print(f"Data directory: [bold]{result.base}[/bold]")
    if result.migrations_applied:
        console.print(f"Database: applied migrations {result.migrations_applied}")
    else:
        console.print("Database: up to date")
    cfg = config.config_path(result.base)
    console.print(f"Config: {'created' if result.config_written else 'kept existing'} {cfg}")


@app.command()
def serve(
    client: str = typer.Option(..., "--client", help="Name of the MCP client launching this server, e.g. claude-desktop."),
) -> None:
    """Run the MCP server over stdio for one client."""
    # stdout carries the MCP protocol: nothing else may be printed to it.
    from openmcp.server import build_server

    try:
        client = TypeAdapter(ClientName).validate_python(client)
    except ValidationError:
        raise typer.BadParameter(
            "use 1-64 letters, digits, '.', '_' or '-', starting with a letter or digit",
            param_hint="--client",
        ) from None
    base = config.home()
    initialize(base)
    build_server(client, base).run(transport="stdio")


if __name__ == "__main__":
    app()
