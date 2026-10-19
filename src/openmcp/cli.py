"""`openmcp` command-line interface."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import typer
from pydantic import BaseModel, TypeAdapter, ValidationError
from rich.console import Console
from rich.table import Table
from rich.text import Text

from openmcp import audit, config, db, embed, repo
from openmcp.models import (
    CAPTURE_SOURCES,
    KINDS,
    AuditFilter,
    CaptureRequest,
    ClientName,
    ConversationFilter,
    MemoryFilter,
)

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

    client = _client_name(client, "--client")
    base = config.home()
    initialize(base)
    build_server(client, base).run(transport="stdio")


def _client_name(value: str, option: str) -> str:
    try:
        return TypeAdapter(ClientName).validate_python(value)
    except ValidationError:
        raise typer.BadParameter(
            "use 1-64 letters, digits, '.', '_' or '-', starting with a letter or digit", param_hint=option
        ) from None


@app.command()
def setup(
    client: str = typer.Argument(
        ..., help="claude-desktop, vscode-copilot, or claude-code (others get printed instructions)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the steps without changing anything."),
    dev: bool = typer.Option(
        False, "--dev", help="Register this development checkout itself: the app then runs whatever branch is checked out."
    ),
) -> None:
    """Connect an AI assistant to OpenMCP and install the standing instruction it needs."""
    from openmcp import setup as client_setup

    client = _client_name(client, "CLIENT")
    checkout = client_setup.development_checkout()
    if checkout is not None and not dev:
        # Registering this checkout would make the app run whatever branch is checked out,
        # and this code would migrate the real database. Hand the whole job to an installed copy.
        installed = client_setup.installed_command(checkout)
        if installed is None:
            console.print(
                f"[yellow]This is a development checkout[/yellow] ({checkout}), and no installed openmcp was "
                "found. Registering the checkout would make the app run whatever branch is checked out, "
                "against your real data. Install openmcp (e.g. `uv tool install --managed-python --python 3.12 "
                f"\"openmcp @ git+<repo URL>@main\"`) and run setup again, or pass --dev to register the checkout."
            )
            raise typer.Exit(2)
        console.print(
            f"This is a development checkout ({checkout}); running setup with the installed openmcp at "
            f"{installed[0]} instead, so the app and your data don't follow your branches. "
            "(--dev registers the checkout.)"
        )
        args = [*installed, "setup", client, *(["--dry-run"] if dry_run else [])]
        raise typer.Exit(subprocess.run(args).returncode)
    base = config.home()
    initialize(base)
    ctx = client_setup.Context(
        base=base, user_home=Path.home(), platform=sys.platform, command=client_setup.openmcp_command()
    )
    incomplete = False
    for number, step in enumerate(client_setup.plan(client, ctx), start=1):
        console.print(f"[bold]{number}. {step.description}[/bold]")
        if step.apply is not None:
            if dry_run:
                console.print("   [dim]dry run: not changed[/dim]")
            else:
                try:
                    detail = step.apply()
                except client_setup.SetupError as e:
                    incomplete = True
                    console.print("   [yellow]Not done automatically:[/yellow]")
                    console.print(textwrap.indent(str(e), "   "), markup=False, highlight=False)
                else:
                    console.print(f"   [green]done[/green] [dim]{detail or ''}[/dim]")
        if step.manual:
            console.print(textwrap.indent(step.manual, "   "), markup=False, highlight=False)
    if not dry_run:
        if client in CAPTURE_SOURCES:
            _run_capture(CaptureRequest(sources=[client]))
        else:
            _run_embed()
    if incomplete:
        raise typer.Exit(1)


# --- helpers -------------------------------------------------------------------------

CLI_CLIENT = "cli"
"""Audit-log client name for actions the user takes through this CLI."""

M = TypeVar("M", bound=BaseModel)


def _validated(model: type[M], options: dict[str, str], **fields: object) -> M:
    """Build a model from CLI input, turning validation errors into usage errors.
    `options` maps model field names to the CLI option that sets them."""
    try:
        return model.model_validate(fields)
    except ValidationError as e:
        err = e.errors()[0]
        option = options.get(str(err["loc"][0])) if err["loc"] else None
        raise typer.BadParameter(err["msg"], param_hint=option) from None


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    base = config.home()
    initialize(base)
    with closing(db.connect(config.db_path(base))) as conn:
        yield conn


def _when(dt: datetime) -> str:
    return dt.astimezone().strftime("%Y-%m-%d %H:%M")


def _one_line(text: str, width: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


# --- memories ------------------------------------------------------------------------

memories_app = typer.Typer(no_args_is_help=True, help="List, inspect, and delete stored memories.")
app.add_typer(memories_app, name="memories")


@memories_app.command("list")
def memories_list(
    kind: list[str] = typer.Option([], "--kind", "-k", help=f"Only these kinds ({', '.join(KINDS)}). Repeatable."),
    project: str | None = typer.Option(None, "--project", "-p", help="Only memories tagged with this project."),
    scope: list[str] = typer.Option([], "--scope", "-s", help="Only these scopes. Repeatable."),
    include_superseded: bool = typer.Option(False, "--all", "-a", help="Include replaced (older) versions."),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum number of memories to show."),
) -> None:
    """List memories, newest first."""
    f = _validated(
        MemoryFilter,
        {"kinds": "--kind", "project": "--project", "scopes": "--scope", "limit": "--limit"},
        kinds=kind, project=project, scopes=scope, include_superseded=include_superseded, limit=limit,
    )
    with _db() as conn, audit.audited(conn, CLI_CLIENT, "memories list") as call:
        found = repo.list_memories(
            conn,
            scopes=f.scopes or None,
            kinds=f.kinds,
            project=f.project,
            include_superseded=f.include_superseded,
            limit=f.limit,
        )
        call.result_count = len(found)

    if not found:
        console.print("No memories found.")
        return
    table = Table(box=None, header_style="bold")
    for column in ("id", "kind", "scope", "project", "content", "source", "updated"):
        table.add_column(column, overflow="fold", no_wrap=column != "content")
    for m in found:
        content = _one_line(m.content)
        if not m.active:
            content = f"[dim]{content} (replaced by {m.superseded_by})[/dim]"
        table.add_row(
            str(m.id), m.kind, m.scope, m.project or "", content, m.source_client, _when(m.updated_at)
        )
    console.print(table)


@memories_app.command("show")
def memories_show(memory_id: int = typer.Argument(..., help="Memory id.")) -> None:
    """Show one memory in full, with its version history."""
    with _db() as conn, audit.audited(conn, CLI_CLIENT, "memories show") as call:
        call.summary = f"id={memory_id}"
        chain = repo.memory_chain(conn, memory_id)
        call.result_count = len(chain)
    memory = next((m for m in chain if m.id == memory_id), None)
    if memory is None:
        console.print(f"[red]Memory {memory_id} not found.[/red]")
        raise typer.Exit(1)

    status = "active" if memory.active else f"replaced by {memory.superseded_by}"
    console.print(f"[bold]Memory {memory.id}[/bold] ({memory.kind}, {status})")
    for label, value in (
        ("subject", memory.subject),
        ("project", memory.project),
        ("scope", memory.scope),
        ("source", memory.source_client),
        ("confidence", memory.confidence),
        ("created", _when(memory.created_at)),
        ("updated", _when(memory.updated_at)),
    ):
        if value is not None:
            console.print(f"  {label + ':':<12}{value}")
    if memory.metadata:
        console.print(f"  {'metadata:':<12}{json.dumps(memory.metadata)}")
    console.print()
    console.print(memory.content, markup=False, highlight=False)
    if len(chain) > 1:
        console.print("\n[bold]History[/bold] (oldest first)")
        for m in chain:
            marker = "→" if m.id == memory.id else " "
            console.print(f" {marker} {m.id:>5}  {_when(m.created_at)}  {m.source_client:<16} {_one_line(m.content, 60)}")


@memories_app.command("rm")
def memories_rm(
    memory_id: int = typer.Argument(..., help="Memory id."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation."),
) -> None:
    """Permanently delete a memory and all of its versions."""
    with _db() as conn:
        chain = repo.memory_chain(conn, memory_id)
        if not any(m.id == memory_id for m in chain):
            console.print(f"[red]Memory {memory_id} not found.[/red]")
            raise typer.Exit(1)
        head = chain[-1]
        console.print(f"[bold]{head.id}[/bold] ({head.kind}, {head.scope}): {_one_line(head.content)}")
        versions = f" and {len(chain) - 1} earlier version(s)" if len(chain) > 1 else ""
        if not yes and not typer.confirm(f"Delete this memory{versions}? This can't be undone.", default=False):
            raise typer.Exit(1)
        with audit.audited(conn, CLI_CLIENT, "memories rm") as call, db.transaction(conn):
            call.summary = f"id={memory_id}"
            call.result_count = repo.delete_memories(conn, [m.id for m in chain])
    console.print(f"Deleted {call.result_count} row(s).")


# --- conversations -------------------------------------------------------------------


conversations_app = typer.Typer(no_args_is_help=True, help="List, read, and delete stored conversations.")
app.add_typer(conversations_app, name="conversations")


@conversations_app.command("list")
def conversations_list(
    source: str | None = typer.Option(None, "--source", help="Only this assistant/provider (partial match)."),
    scope: list[str] = typer.Option([], "--scope", "-s", help="Only these scopes. Repeatable."),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum number of conversations to show."),
) -> None:
    """List conversations, most recently active first."""
    f = _validated(
        ConversationFilter, {"scopes": "--scope", "limit": "--limit"}, source=source, scopes=scope, limit=limit
    )
    with _db() as conn, audit.audited(conn, CLI_CLIENT, "conversations list") as call:
        convs = repo.list_conversations(conn, scopes=f.scopes or None, source=f.source, limit=f.limit)
        counts = repo.message_counts(conn, [c.id for c in convs])
        call.result_count = len(convs)

    if not convs:
        console.print("No conversations found.")
        return
    table = Table(box=None, header_style="bold")
    for column in ("id", "source", "scope", "title", "messages", "updated"):
        table.add_column(column, overflow="fold", no_wrap=column != "title")
    for c in convs:
        table.add_row(
            str(c.id), c.provider, c.scope, Text(c.title or ""), str(counts.get(c.id, 0)),
            _when(c.updated_at) if c.updated_at else "",
        )
    console.print(table)


@conversations_app.command("show")
def conversations_show(conversation_id: int = typer.Argument(..., help="Conversation id.")) -> None:
    """Show a conversation's messages in full."""
    with _db() as conn, audit.audited(conn, CLI_CLIENT, "conversations show") as call:
        call.summary = f"id={conversation_id}"
        conv = repo.get_conversation(conn, conversation_id)
        messages = repo.conversation_messages(conn, conversation_id) if conv else []
        call.result_count = len(messages)
    if conv is None:
        console.print(f"[red]Conversation {conversation_id} not found.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Conversation {conv.id}[/bold]: ", Text(conv.title or "(untitled)"))
    console.print(f"  {conv.provider}, scope {conv.scope}, {len(messages)} messages")
    for m in messages:
        when = f"  {_when(m.created_at)}" if m.created_at else ""
        console.print(f"\n[bold]{m.role}[/bold][dim]{when}[/dim]")
        console.print(m.content, markup=False, highlight=False)


@conversations_app.command("rm")
def conversations_rm(
    conversation_id: int = typer.Argument(..., help="Conversation id."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation."),
) -> None:
    """Permanently delete a conversation and all of its messages."""
    with _db() as conn:
        conv = repo.get_conversation(conn, conversation_id)
        if conv is None:
            console.print(f"[red]Conversation {conversation_id} not found.[/red]")
            raise typer.Exit(1)
        count = repo.message_counts(conn, [conv.id]).get(conv.id, 0)
        console.print(f"[bold]{conv.id}[/bold] ({conv.provider}, {count} messages): ", Text(conv.title or ""))
        if not yes and not typer.confirm("Delete this conversation? This can't be undone.", default=False):
            raise typer.Exit(1)
        with audit.audited(conn, CLI_CLIENT, "conversations rm") as call, db.transaction(conn):
            call.summary = f"id={conversation_id}"
            repo.delete_conversation(conn, conversation_id)
            call.result_count = count
    console.print(f"Deleted conversation {conversation_id} ({count} messages).")


# --- capture -------------------------------------------------------------------------


def _run_capture(request: CaptureRequest) -> None:
    from openmcp.capture import capture

    cfg = config.load_config(config.config_path(config.home()))
    sources = request.sources or list(cfg.capture.sources)
    if not sources:
        console.print("Transcript capture is off. Turn it on with: openmcp setup claude-code (or vscode-copilot)")
        return
    days = request.days or cfg.capture.days
    with _db() as conn, audit.audited(conn, CLI_CLIENT, "capture") as call:
        call.summary = f"{', '.join(sources)}, {days} days"
        report = capture(conn, cfg, sources=sources, days=days)
        call.result_count = report.messages
    console.print(
        f"Captured from {', '.join(sources)} (last {days} days): read {report.files_read} transcript files, "
        f"{report.messages} messages added or updated in {report.conversations} conversations."
    )
    for error in report.errors:
        console.print(f"[yellow]Skipped[/yellow] {error}", markup=True, highlight=False)
    _run_embed()


@app.command("capture")
def capture_command(
    source: list[str] = typer.Option(
        [], "--source", help=f"Only this source ({', '.join(CAPTURE_SOURCES)}). Repeatable."
    ),
    days: int | None = typer.Option(None, "--days", help="Read transcripts changed in this many days (default: config)."),
) -> None:
    """Store new conversations from the transcripts of clients with capture turned on.

    Servers already do this before each search; run it to catch up on older history."""
    _run_capture(_validated(CaptureRequest, {"sources": "--source", "days": "--days"}, sources=source, days=days))


# --- embeddings ----------------------------------------------------------------------


def _run_embed() -> bool:
    """Download the model if needed and embed every pending memory and chunk. Returns
    False (after saying so) if the model couldn't be loaded."""
    from rich.progress import Progress

    with _db() as conn:
        pending = repo.count_pending_embeddings(conn)
        with console.status(f"Loading the embedding model ({embed.MODEL}; downloads ~70MB the first time)..."):
            embedder = embed.get_embedder(allow_download=True)
        if embedder is None:
            console.print(
                "[yellow]Semantic search unavailable:[/yellow] the embedding model couldn't be loaded "
                f"into {embed.model_dir()}. Keyword search still works; run `openmcp embed` to retry."
            )
            return False
        with audit.audited(conn, CLI_CLIENT, "embed") as call, Progress(console=console, transient=True) as bar:
            task = bar.add_task("Embedding", total=pending)
            report = embed.embed_pending(conn, embedder, progress=lambda n: bar.advance(task, n))
            call.result_count = report.memories + report.chunks
    console.print(
        f"Semantic search ready: embedded {report.memories} memories and {report.chunks} conversation chunks."
    )
    return True


@app.command("embed")
def embed_command() -> None:
    """Download the embedding model (once) and embed memories and conversations that don't have
    embeddings yet, so search can match by meaning. Servers never download the model themselves."""
    if not _run_embed():
        raise typer.Exit(1)


# --- audit ---------------------------------------------------------------------------


@app.command("audit")
def audit_log(
    client: str | None = typer.Option(None, "--client", "-c", help="Only calls from this client."),
    tool: str | None = typer.Option(None, "--tool", "-t", help="Only calls to this tool."),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum number of entries to show."),
    show_stats: bool = typer.Option(False, "--stats", help="Show call counts and p50/p95 latency per tool instead."),
) -> None:
    """Show the log of every tool call made by every client."""
    f = _validated(AuditFilter, {"limit": "--limit"}, client=client, tool=tool, limit=limit)
    with _db() as conn:
        if show_stats:
            rows = audit.stats(repo.audit_durations(conn))
        else:
            entries = repo.list_audit(conn, f)

    if show_stats:
        if not rows:
            console.print("No tool calls logged yet.")
            return
        table = Table(box=None, header_style="bold")
        for column in ("tool", "calls", "p50 ms", "p95 ms", "max ms"):
            table.add_column(column, justify="left" if column == "tool" else "right")
        for s in rows:
            table.add_row(s.tool, str(s.calls), f"{s.p50_ms:.1f}", f"{s.p95_ms:.1f}", f"{s.max_ms:.1f}")
        console.print(table)
        return

    if not entries:
        console.print("No tool calls logged yet.")
        return
    table = Table(box=None, header_style="bold")
    for column in ("time", "client", "app", "tool", "summary", "results", "ms"):
        table.add_column(
            column, justify="right" if column in ("results", "ms") else "left", no_wrap=column != "summary"
        )
    for e in entries:
        table.add_row(
            _when(e.ts),
            e.client,
            Text(e.app or ""),
            e.tool,
            Text(e.summary or ""),
            "" if e.result_count is None else str(e.result_count),
            f"{e.duration_ms:.1f}",
        )
    console.print(table)


if __name__ == "__main__":
    app()
