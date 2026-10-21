"""`openmemore` command-line interface."""

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
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from openmemore import audit, config, db, embed, repo
from openmemore.models import (
    CAPTURE_SOURCES,
    KINDS,
    AuditFilter,
    CaptureRequest,
    ClientName,
    ConversationFilter,
    ImportRequest,
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
def serve() -> None:
    """Run the MCP server over stdio for whichever MCP client starts it (it names the
    client from what it reports when it connects)."""
    # stdout carries the MCP protocol: nothing else may be printed to it.
    from openmemore.server import build_server

    base = config.home()
    initialize(base)
    build_server(base).run(transport="stdio")


def _client_name(value: str, option: str) -> str:
    try:
        return TypeAdapter(ClientName).validate_python(value)
    except ValidationError:
        raise typer.BadParameter(
            "use 1-64 letters, digits, '.', '_' or '-', starting with a letter or digit", param_hint=option
        ) from None


@app.command()
def setup(
    client: str | None = typer.Argument(
        None,
        help="The client: vscode-copilot, cursor, or claude-code use the app's own install command; "
        "any other name (or none) gets the snippet that works in every MCP client.",
    ),
    config_file: Path | None = typer.Option(
        None, "--config", help="Merge the server into this JSON config file (its mcpServers; keeps a .bak copy)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the steps without changing anything."),
    dev: bool = typer.Option(
        False, "--dev", help="Register this development checkout itself: the app then runs whatever branch is checked out."
    ),
) -> None:
    """Connect an MCP client to OpenMemore and put the standing instruction on your clipboard."""
    from openmemore import setup as client_setup

    if client is not None:
        client = _client_name(client, "CLIENT")
    checkout = client_setup.development_checkout()
    if checkout is not None and not dev:
        # Registering this checkout would make the app run whatever branch is checked out,
        # and this code would migrate the real database. Hand the whole job to an installed copy.
        installed = client_setup.installed_command(checkout)
        if installed is None:
            console.print(
                f"[yellow]This is a development checkout[/yellow] ({checkout}), and no installed openmemore was "
                "found. Registering the checkout would make the app run whatever branch is checked out, "
                "against your real data. Install openmemore (e.g. `uv tool install --managed-python --python 3.12 "
                f"\"openmemore @ git+<repo URL>@main\"`) and run setup again, or pass --dev to register the checkout."
            )
            raise typer.Exit(2)
        console.print(
            f"This is a development checkout ({checkout}); running setup with the installed openmemore at "
            f"{installed[0]} instead, so the app and your data don't follow your branches. "
            "(--dev registers the checkout.)"
        )
        args = [*installed, "setup", *([client] if client else []),
                *(["--config", str(config_file)] if config_file else []), *(["--dry-run"] if dry_run else [])]
        raise typer.Exit(subprocess.run(args).returncode)
    base = config.home()
    initialize(base)
    ctx = client_setup.Context(
        base=base, user_home=Path.home(), platform=sys.platform, command=client_setup.openmemore_command()
    )
    incomplete = False
    for number, step in enumerate(client_setup.plan(client, ctx, config_file), start=1):
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
        if step.note:
            console.print(f"   [dim]{escape(step.note)}[/dim]", highlight=False)
    if not dry_run:
        if client in CAPTURE_SOURCES:
            _run_capture(CaptureRequest(sources=[client]))
        else:
            _run_embed()
        _offer_daemon()
    if incomplete:
        raise typer.Exit(1)


def _offer_daemon() -> None:
    """After setup: if there's something to capture in the background and the daemon
    isn't running, ask whether it may run (SPEC.md M6)."""
    from openmemore import daemon, service

    cfg = config.load_config(config.config_path(config.home()))
    if not cfg.capture.sources or daemon.healthy(cfg.daemon.port):
        return
    try:
        if service.installed(_service()):
            return
    except service.ServiceError:
        return
    console.print("\n[bold]Capture in the background?[/bold] Without it, chats are captured when an assistant next looks.")
    _install_daemon(yes=False)


@app.command("clients")
def clients_command(
    show_all: bool = typer.Option(False, "--all", "-a", help="Also list names from earlier versions' audit history."),
) -> None:
    """List the MCP clients that have connected, and the scopes each one gets."""
    from openmemore.capture import captured_source
    from openmemore.clients import client_name

    cfg = config.load_config(config.config_path(config.home()))
    with _db() as conn:
        seen = repo.seen_clients(conn)
    current = [row for row in seen if row[1] is not None and row[0] == client_name(row[1])]
    earlier = len(seen) - len(current)
    seen = seen if show_all else current
    if not seen:
        console.print("No MCP client has connected yet. `openmemore setup` connects one; any MCP client works.")
        return
    table = Table(box=None, header_style="bold")
    for column in ("client", "application", "reads", "writes", "capture", "calls", "last seen"):
        table.add_column(column, no_wrap=column != "application")
    for client, app, calls, last in seen:
        scopes = config.resolve_client(cfg, client)
        source = captured_source(app, cfg.capture.sources)
        table.add_row(
            client, Text(app or "(unknown)"),
            "all (ignore_scopes)" if scopes.read_scopes is None else ", ".join(scopes.read_scopes), scopes.write_scope,
            f"transcripts ({source})" if source else "record_exchange", str(calls), _when(last),
        )
    console.print(table)
    if earlier and not show_all:
        console.print(f"({earlier} earlier names in the audit history: --all lists them)")
    console.print(
        "\nA client's name is what its application reports. Give it scopes with read_scopes and write_scope "
        "under [clients.<name>] in config.toml.",
        markup=False,
    )


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
    from openmemore.capture import capture

    cfg = config.load_config(config.config_path(config.home()))
    sources = request.sources or list(cfg.capture.sources)
    if not sources:
        console.print("Transcript capture is off. Turn it on with: openmemore setup claude-code (or vscode-copilot)")
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
    if report.filtered:
        console.print(f"[dim]Not captured: {report.filtered} from folders outside [capture] include_folders / "
                      "exclude_folders.[/dim]")
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


# --- import ---------------------------------------------------------------------------

def _format_name(importer_name: str) -> str:
    from openmemore.ingest.registry import importer_named

    importer = importer_named(importer_name)
    return importer.display if importer else importer_name


@app.command("import")
def import_command(
    path: Path = typer.Argument(..., help="The export: the .zip you downloaded, its conversations.json, or an unpacked folder."),
    scope: str | None = typer.Option(
        None, "--scope", "-s",
        help="Scope to store everything in. Default: an OpenMemore export's own scopes; others go to config's default write scope.",
    ),
) -> None:
    """Import a ChatGPT or Claude data export (or an OpenMemore export), newest conversations first.

    Safe to interrupt: running the same command again resumes where it stopped, and
    importing a file again does nothing once it's done."""
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

    from openmemore.ingest.detect import ExportError
    from openmemore.ingest.jobs import import_export

    request = _validated(ImportRequest, {"scope": "--scope"}, scope=scope)
    cfg = config.load_config(config.config_path(config.home()))
    target_scope = request.scope or cfg.defaults.write_scope
    with _db() as conn:
        with console.status(f"Loading the embedding model ({embed.MODEL})..."):
            embedder = embed.get_embedder(allow_download=True)
        if embedder is None:
            console.print("[yellow]Semantic search unavailable:[/yellow] importing for keyword search only; "
                          "run `openmemore embed` later to add the rest.")
        columns = (TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TextColumn("conversations"),
                   TimeRemainingColumn())
        try:
            with audit.audited(conn, CLI_CLIENT, "import") as call, Progress(*columns, console=console) as bar:
                call.summary = f"{path.name}, scope={target_scope}"
                storing = bar.add_task("Importing", total=None)
                embedding = bar.add_task("Embedding", total=None, visible=False)
                report = import_export(
                    conn, path, scope=target_scope, keep_scopes=request.scope is None, embedder=embedder,
                    progress=lambda done, total: bar.update(storing, completed=done, total=total),
                    embed_progress=lambda done, total: bar.update(embedding, completed=done, total=total, visible=True),
                )
                call.result_count = report.messages
        except ExportError as e:
            console.print(f"[red]Can't import {path}:[/red] {e}")
            raise typer.Exit(2) from None
        except KeyboardInterrupt:
            console.print(
                "\nInterrupted. Run the same command again to resume where it stopped. (If it was already "
                "embedding, everything is stored and searchable by keyword; `openmemore embed` finishes the rest.)"
            )
            raise typer.Exit(130) from None

    name = _format_name(report.provider)
    if report.already_done:
        console.print(f"Already imported: this {name} export ({report.total} conversations) was imported before.")
        return
    resumed = f", resuming at {report.resumed_from}" if report.resumed_from else ""
    what = name if report.provider == "openmemore" else f"{name} export"  # "OpenMemore export" already says it
    where = f"into scope '{target_scope}'" if request.scope or report.provider != "openmemore" else "keeping its scopes"
    console.print(
        f"Imported the {what} {where}{resumed}: {report.conversations} conversations "
        f"with new messages ({report.messages} messages) of {report.total}."
    )
    if report.memories or report.memories_present:
        console.print(f"Memories: {report.memories} added, {report.memories_present} already here.")
    if report.skipped:
        console.print(f"Skipped {report.skipped} with nothing to keep, or that you deleted before.")
    for error in report.errors:
        console.print(f"[yellow]Skipped[/yellow] {error}", highlight=False)


@app.command("export")
def export_command(
    output: str | None = typer.Option(
        None, "--output", "-o", help="File to write (default: openmemore-export-<date>.json here), or '-' for stdout."
    ),
    scope: list[str] = typer.Option([], "--scope", "-s", help="Only these scopes (default: everything). Repeatable."),
) -> None:
    """Export every memory and conversation, in the format `openmemore import` reads back."""
    from openmemore.export import export_document

    with _db() as conn, audit.audited(conn, CLI_CLIENT, "export") as call:
        doc = export_document(conn, scopes=scope or None)
        call.summary = f"{len(doc['memories'])} memories, {len(doc['conversations'])} conversations"
        call.result_count = len(doc["memories"]) + len(doc["conversations"])
    text = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    if output == "-":
        sys.stdout.write(text)
        return
    path = Path(output) if output else Path(f"openmemore-export-{datetime.now():%Y%m%d-%H%M%S}.json")
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)  # everything you've stored: readable by you only
    messages = sum(len(c["messages"]) for c in doc["conversations"])
    current = sum(1 for m in doc["memories"] if m["superseded_by"] is None)
    console.print(
        f"Exported {current} memories ({len(doc['memories']) - current} earlier versions) and "
        f"{len(doc['conversations'])} conversations ({messages} messages) to {path}"
    )


def _local_moment(value: str) -> datetime:
    """'2026-10-20' (midnight, local time) or a full ISO time; without a zone, local time."""
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        raise typer.BadParameter("use a date like 2026-10-20 or a time like 2026-10-20T09:30",
                                 param_hint="--before") from None
    return moment if moment.tzinfo else moment.astimezone()


def _password_ok() -> bool:
    """Ask for the login password the way sudo does, through sudo itself.

    -k ignores sudo's cached credentials, so it always asks and leaves none behind.
    Windows has no sudo: the typed confirmation stands alone there.
    """
    if sys.platform == "win32":
        return True
    try:
        return subprocess.run(["sudo", "-k", "-v", "-p", "Password for %u, to wipe: "]).returncode == 0
    except FileNotFoundError:
        console.print("[red]wipe asks for your password through sudo, which isn't installed.[/red]")
        return False


@app.command("wipe")
def wipe_command(
    before: str | None = typer.Option(
        None, "--before", help="Delete what wasn't touched since this date (local time), e.g. 2026-10-20."
    ),
    everything: bool = typer.Option(False, "--all", help="Delete everything stored: a fresh start."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask to type 'wipe'. Your password is still asked."),
    no_backup: bool = typer.Option(False, "--no-backup", help="Don't keep a copy of the database first."),
) -> None:
    """Delete in bulk: everything not touched since a date (--before), or everything (--all).

    Like sudo, it asks for your login password first (not on Windows).

    Whole conversations and memories go, with their versions, proposals, the audit log
    and finished import and consolidation records from that time. Deleted conversations
    stay deleted: capture won't bring them back. Config, the embedding model and the
    daemon are untouched. The deleted text is then erased from the database file.
    """
    from openmemore import wipe

    if (before is None) == (not everything):
        console.print("[red]Choose one: --before DATE, or --all.[/red]")
        raise typer.Exit(2)
    cutoff = _local_moment(before) if before is not None else None
    with _db() as conn:
        p = wipe.plan(conn, cutoff)
        what = f"not touched since {_when(cutoff)}" if cutoff else "everything"
        if p.empty:
            console.print(f"Nothing to delete ({what}).")
            return
        console.print(f"[bold]This deletes {what}:[/bold]")
        for count, label in [(len(p.memory_ids), "memories, counting earlier versions"),
                             (len(p.conversation_ids), f"conversations ({p.messages} messages)"),
                             (p.proposals, "proposals"), (p.audit_entries, "audit log entries"),
                             (p.consolidation_runs, "consolidation runs"), (p.import_jobs, "finished imports")]:
            if count:
                console.print(f"  {count:>6}  {label}")
        typed = yes or typer.prompt("It can't be undone. Type 'wipe' to continue", default="") == "wipe"
        if not typed or not _password_ok():
            console.print("Nothing deleted.")
            raise typer.Exit(1)
        copy = None
        if not no_backup:
            path = config.db_path(config.home())
            copy = wipe.backup(conn, path.with_name(f"{path.name}.before-wipe-{datetime.now():%Y%m%d-%H%M%S}"))
        with audit.audited(conn, CLI_CLIENT, "wipe") as call:
            call.summary = f"before {repo.to_iso(cutoff)}" if cutoff else "all"
            wipe.apply(conn, p)
            call.result_count = len(p.memory_ids) + len(p.conversation_ids)
    console.print(f"Deleted {len(p.memory_ids)} memories and {len(p.conversation_ids)} conversations.")
    if copy:
        console.print(f"A copy from before the wipe is at {copy}. Delete it too to erase the data for good.")


@app.command("status")
def status_command() -> None:
    """Show import jobs and what's stored."""
    with _db() as conn:
        jobs = repo.list_jobs(conn)
        counts = repo.store_counts(conn)
    if jobs:
        table = Table(box=None, header_style="bold")
        for column in ("job", "export", "file", "status", "progress", "started", "finished"):
            table.add_column(column, no_wrap=column != "file")
        for j in jobs:
            status = j.status if not j.error else f"{j.status}: {j.error}"
            table.add_row(
                str(j.id), _format_name(j.provider), Text(Path(j.file_path).name), Text(status),
                f"{j.processed}/{j.total if j.total is not None else '?'}",
                _when(j.started_at) if j.started_at else "", _when(j.finished_at) if j.finished_at else "",
            )
        console.print(table)
        console.print()
    else:
        console.print("No imports yet. Import a ChatGPT or Claude data export with `openmemore import <file>`.\n")
    conversations = counts["conversations"]
    by_source = ", ".join(f"{n} from {p}" for p, n in conversations.items()) or "none"
    console.print(f"Memories: {counts['memories']}")
    console.print(f"Conversations: {sum(conversations.values())} ({by_source}), {counts['messages']} messages")
    pending = counts["pending_embeddings"]
    console.print(f"Waiting for embeddings: {pending}" + (" (run `openmemore embed`)" if pending else ""))
    proposals = counts["pending_proposals"]
    console.print(f"Proposals waiting for review: {proposals}" + (" (run `openmemore review`)" if proposals else ""))
    console.print(f"Conversations not yet consolidated: {counts['due_for_consolidation']}")
    if counts["conflicts"]:
        console.print(f"Conflicts to settle: {counts['conflicts']} (run `openmemore review`)")
    console.print(f"Daemon: {_daemon_line()}", markup=False, soft_wrap=True)
    console.print(f"Automatic consolidation: {_consolidation_line()}", markup=False, soft_wrap=True)


# --- daemon --------------------------------------------------------------------------

daemon_app = typer.Typer(
    no_args_is_help=True,
    help="Capture in the background: transcripts as they change, and clients' OpenTelemetry exports.",
)
app.add_typer(daemon_app, name="daemon")

LOG_ROTATE_BYTES = 5 * 1024 * 1024


def _handoff_from_checkout(args: list[str], dev: bool) -> None:
    """From a development checkout, run `args` with the installed openmemore instead, so a
    long-lived service doesn't run whatever branch is checked out (as `setup` does)."""
    from openmemore import setup as client_setup

    checkout = client_setup.development_checkout()
    if checkout is None or dev:
        return
    installed = client_setup.installed_command(checkout)
    if installed is None:
        console.print(
            f"[yellow]This is a development checkout[/yellow] ({checkout}), and no installed openmemore was found. "
            "Install openmemore and run this again, or pass --dev to run the checkout as the service."
        )
        raise typer.Exit(2)
    console.print(f"This is a development checkout; using the installed openmemore at {installed[0]}. (--dev uses the checkout.)")
    raise typer.Exit(subprocess.run([*installed, *args]).returncode)


def _service():
    from openmemore import service
    from openmemore import setup as client_setup

    return service.describe(sys.platform, client_setup.openmemore_command(), config.home(), Path.home())


@daemon_app.command("run")
def daemon_run(
    home: Path | None = typer.Option(None, "--home", help="Data directory (default: $OPENMEMORE_HOME or ~/.openmemore)."),
    log_file: Path | None = typer.Option(None, "--log-file", help="Append output here instead of the terminal."),
) -> None:
    """Run the daemon in the foreground (the installed service runs this)."""
    import os

    from openmemore import daemon

    if home is not None:
        os.environ["OPENMEMORE_HOME"] = str(home)
    base = config.home()
    initialize(base)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if log_file.exists() and log_file.stat().st_size > LOG_ROTATE_BYTES:
            log_file.replace(log_file.with_suffix(log_file.suffix + ".1"))
        sys.stderr = sys.stdout = log_file.open("a", encoding="utf-8", buffering=1)
    try:
        daemon.run(base, sys.stderr)
    except RuntimeError as e:
        print(f"openmemore daemon: {e}", file=sys.stderr, flush=True)
        raise typer.Exit(1) from None


def _install_daemon(*, yes: bool, dry_run: bool = False) -> bool:
    """Install and start the daemon service, after asking (SPEC.md M6). Returns whether
    it's installed."""
    from openmemore import consent, daemon, service

    base = config.home()
    initialize(base)
    try:
        svc = _service()
    except service.ServiceError as e:
        console.print(str(e))
        return False
    if dry_run:
        console.print(f"Would install the openmemore daemon as a {svc.name}, after asking you.")
        if svc.file is not None:
            console.print(f"Service file: {svc.file}", markup=False, soft_wrap=True)
        for cmd in svc.install:
            console.print("  " + (subprocess.list2cmdline(cmd) if sys.platform == "win32" else " ".join(cmd)),
                          markup=False, soft_wrap=True)
        if svc.content:
            console.print(svc.content, markup=False, soft_wrap=True)
        console.print("[dim]dry run: nothing changed[/dim]")
        return False
    answer = "allowed" if yes else consent.ask()
    if answer != "allowed":
        console.print(
            "Couldn't ask: there's no desktop session or terminal here. To allow it, run "
            "`openmemore daemon install --yes`." if answer == "unavailable"
            else "Not running in the background. `openmemore daemon install` asks again whenever you like."
        )
        return False
    daemon.ensure_token(base)
    with _db() as conn, audit.audited(conn, CLI_CLIENT, "daemon install") as call:
        call.summary = "allowed with --yes" if yes else "allowed when asked"
        try:
            service.install(svc)
        except service.ServiceError as e:
            call.summary += ", failed"
            console.print(f"Couldn't install the {svc.name}: {e}", markup=False, soft_wrap=True)
            return False
    port = config.load_config(config.config_path(base)).daemon.port
    if answer := daemon.wait_until_healthy(port):
        console.print(f"Running in the background (pid {answer['pid']}, 127.0.0.1:{port}); it starts when you log in.")
    else:
        console.print(f"Installed, but it isn't answering on port {port} yet. Its log: {base / service.LOG_FILE}",
                      markup=False, soft_wrap=True)
    if sys.platform == "darwin":
        console.print("macOS lists it under System Settings → General → Login Items, where you can also switch it off.")
    for note in svc.notes:
        console.print(note)
    return True


@daemon_app.command("install")
def daemon_install(
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask: you allow it to run in the background."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be installed without changing anything."),
    dev: bool = typer.Option(False, "--dev", help="Run this development checkout as the service."),
) -> None:
    """Run OpenMemore in the background, starting at login, once you allow it."""
    _handoff_from_checkout(["daemon", "install", *(["--yes"] if yes else []), *(["--dry-run"] if dry_run else [])], dev)
    if not _install_daemon(yes=yes, dry_run=dry_run) and not dry_run:
        raise typer.Exit(1)


@daemon_app.command("uninstall")
def daemon_uninstall() -> None:
    """Stop the daemon and remove its service."""
    from openmemore import service

    try:
        existed = service.uninstall(_service())
    except service.ServiceError as e:
        console.print(str(e))
        raise typer.Exit(1) from None
    console.print("Stopped and removed the openmemore daemon." if existed else "The daemon service wasn't installed.")


def _daemon_line() -> str:
    from openmemore import daemon, service

    base = config.home()
    port = config.load_config(config.config_path(base)).daemon.port
    if answer := daemon.healthy(port):
        state = daemon.read_state(base) or {}
        since = f", since {state['started_at']}" if state.get("started_at") else ""
        return f"running (pid {answer['pid']}, port {port}{since})"
    try:
        installed = service.installed(_service())
    except service.ServiceError:
        installed = False
    if installed:
        return f"installed but not running; its log: {base / service.LOG_FILE}"
    return "not installed (`openmemore daemon install` captures in the background)"


@daemon_app.command("status")
def daemon_status() -> None:
    """Whether the daemon is installed and running."""
    console.print(f"Daemon: {_daemon_line()}", markup=False, soft_wrap=True)


@daemon_app.command("telemetry")
def daemon_telemetry() -> None:
    """Print the OpenTelemetry settings that send an app's conversations to the daemon."""
    from openmemore import daemon

    base = config.home()
    initialize(base)
    token = daemon.ensure_token(base)
    cfg = config.load_config(config.config_path(base))
    console.print(
        "Any app or library that records its conversations with OpenTelemetry's GenAI conventions "
        "(gen_ai.input.messages / gen_ai.output.messages) can send them to the daemon. Give it these standard "
        "OTLP settings, as environment variables:\n",
        markup=False, soft_wrap=True,
    )
    for key, value in (
        ("OTEL_EXPORTER_OTLP_ENDPOINT", daemon.endpoint(cfg)),
        ("OTEL_EXPORTER_OTLP_HEADERS", f"{daemon.TOKEN_HEADER}={token}"),
    ):
        console.print(f"  {key}={value}", markup=False, soft_wrap=True)
    console.print(
        "\nJSON and protobuf both work. Message content is opt-in in most instrumentation; in OpenTelemetry's own "
        "GenAI instrumentations it's:\n\n"
        "  OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental\n"
        "  OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY\n\n"
        "Conversations are filed under the app's service.name (OTEL_SERVICE_NAME), which `openmemore clients` scopes "
        "like any client. If the app already sends telemetry elsewhere, these settings replace that destination.",
        markup=False, soft_wrap=True,
    )


# --- automatic consolidation ------------------------------------------------------------

consolidate_app = typer.Typer(
    no_args_is_help=True,
    help="Consolidate conversations into proposed memories with a model OpenMemore calls itself (off until set up).",
)
app.add_typer(consolidate_app, name="consolidate")


def _consolidation_line() -> str:
    from openmemore import consolidate, llm

    base = config.home()
    cfg = config.load_config(config.config_path(base)).consolidation
    if not cfg.enabled:
        return "off (`openmemore consolidate setup` turns it on)"
    with _db() as conn:
        last = consolidate.last_run(conn)
    line = f"daily at {cfg.schedule} with {cfg.model} at {llm.host(cfg.endpoint)}"
    if last:
        line += (f"; last run {last['started_at'][:16].replace('T', ' ')} UTC: {last['conversations']} conversations, "
                 f"{last['proposals']} proposals" + (f", error: {last['error']}" if last["error"] else ""))
    return line


@consolidate_app.command("setup")
def consolidate_setup(
    endpoint: str = typer.Option(..., "--endpoint", help="Base URL of an Open Responses or Chat Completions endpoint, "
                                 "e.g. http://127.0.0.1:11434/v1 or https://api.openai.com/v1."),
    model: str = typer.Option(..., "--model", help="The model's name at that endpoint."),
    api: str = typer.Option("responses", "--api", help="responses (Open Responses, the default) or chat-completions."),
    schedule: str = typer.Option("03:00", "--schedule", help="Local time of the daily run (HH:MM)."),
    key: bool = typer.Option(False, "--key", help="Prompt for an API key, kept in a file only you can read."),
    check: bool = typer.Option(True, "--check/--no-check", help="Send a one-line test request first."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask before allowing a non-local endpoint."),
) -> None:
    """Set up automatic consolidation (off until you do): which model to use, and when."""
    from urllib.parse import urlparse

    from openmemore import consent, consolidate, llm

    if urlparse(endpoint).scheme not in ("http", "https") or not urlparse(endpoint).hostname:
        console.print("The endpoint must be an http(s) URL, e.g. http://127.0.0.1:11434/v1")
        raise typer.Exit(2)
    if api not in ("responses", "chat-completions"):
        console.print("--api must be responses or chat-completions")
        raise typer.Exit(2)
    base = config.home()
    initialize(base)
    path = config.config_path(base)
    cfg = config.load_config(path)
    host = llm.host(endpoint)
    if not llm.is_local(endpoint) and cfg.consolidation.allowed_host != host:
        answer = "allowed" if yes else consent.ask(
            f"Send your conversations to {host}?",
            f"Automatic consolidation sends each conversation's new messages, and related memories, to {host} "
            f"for {model} to read. They're handled under that provider's terms. Nothing becomes a memory until "
            "you approve it with openmemore review.\n\nTurn it off any time with: openmemore consolidate off",
        )
        if answer != "allowed":
            console.print(f"Not set up: conversations won't be sent to {host}." if answer == "denied" else
                          "Couldn't ask (no desktop or terminal); pass --yes to allow it.")
            raise typer.Exit(1)
    if key:
        consolidate.save_key(base, typer.prompt(f"API key for {host}", hide_input=True))
    try:
        cfg.consolidation = config.ConsolidationConfig(
            enabled=False, endpoint=endpoint, model=model, api=api, schedule=schedule,
            max_conversations=cfg.consolidation.max_conversations,
            allowed_host=None if llm.is_local(endpoint) else host,
        )
    except ValidationError as e:
        console.print(f"Invalid setting: {e.errors()[0]['msg']}")
        raise typer.Exit(2) from None
    if check:
        try:
            reply = llm.complete(consolidate.model_from(base, cfg), "Answer with the single word OK.", "Are you there?")
        except llm.ModelError as e:
            config.save_config(path, cfg)  # kept, but off until it works
            console.print(f"[red]The test request failed:[/red] {escape(str(e))}\nSaved but left off; fix it and run "
                          "setup again (or pass --no-check).")
            raise typer.Exit(1) from None
        console.print(f"Test request answered: {reply.strip()[:40]!r}", markup=False)
    cfg.consolidation.enabled = True
    config.save_config(path, cfg)
    with _db() as conn, audit.audited(conn, CLI_CLIENT, "consolidate setup") as call:
        call.summary = f"{model} at {host}" + (
            "" if llm.is_local(endpoint) else (", allowed with --yes" if yes else ", allowed when asked"))
    console.print(f"Automatic consolidation: {_consolidation_line()}", markup=False, soft_wrap=True)
    if not _daemon_line().startswith("running"):
        console.print("It runs in the daemon, which isn't running: `openmemore daemon install`. "
                      "`openmemore consolidate run` runs it now.")


@consolidate_app.command("run")
def consolidate_run(
    limit: int | None = typer.Option(None, "--limit", "-n", min=1, help="At most this many conversations."),
) -> None:
    """Consolidate now, instead of waiting for the daily run."""
    from openmemore import consolidate, llm

    base = config.home()
    initialize(base)
    cfg = config.load_config(config.config_path(base))
    model = consolidate.model_from(base, cfg)
    if model is None:
        console.print("Not set up yet: `openmemore consolidate setup --endpoint URL --model NAME`.")
        raise typer.Exit(2)
    if not consolidate.permitted(cfg):
        console.print(f"Sending conversations to {llm.host(model.endpoint)} hasn't been allowed; run setup again.")
        raise typer.Exit(2)

    def progress(report) -> None:
        console.print(f"  {report.conversations} conversations, {report.proposals} proposals so far")

    with _db() as conn, audit.audited(conn, CLI_CLIENT, "consolidate") as call:
        report = consolidate.run(conn, cfg, model, limit=limit, progress=progress)
        call.summary = (f"{report.conversations} conversations, {report.proposals} proposals, {model.model} at "
                        f"{llm.host(model.endpoint)}")
        call.result_count = report.proposals
    console.print(f"Consolidated {report.conversations} conversations into {report.proposals} proposals"
                  + (f" ({report.dropped} unusable items dropped)" if report.dropped else "") + ".")
    for error in report.errors:
        console.print(f"[yellow]{escape(error)}[/yellow]")
    if report.proposals:
        console.print("Review them with `openmemore review`: nothing is saved until you approve it.")
    if report.errors and not report.conversations:
        raise typer.Exit(1)


@consolidate_app.command("status")
def consolidate_status() -> None:
    """Whether automatic consolidation is on, and how the last run went."""
    console.print(f"Automatic consolidation: {_consolidation_line()}", markup=False, soft_wrap=True)
    with _db() as conn:
        due = repo.count_due_conversations(conn, None)
    console.print(f"Conversations not yet consolidated: {due}")


@consolidate_app.command("off")
def consolidate_off(
    forget: bool = typer.Option(False, "--forget", help="Also delete the saved API key and the allowed host."),
) -> None:
    """Turn automatic consolidation off."""
    from openmemore import consolidate

    base = config.home()
    path = config.config_path(base)
    cfg = config.load_config(path)
    cfg.consolidation.enabled = False
    if forget:
        cfg.consolidation.allowed_host = None
        consolidate.key_path(base).unlink(missing_ok=True)
    config.save_config(path, cfg)
    console.print("Automatic consolidation is off." + (" The API key and allowed host are forgotten." if forget else ""))


# --- review --------------------------------------------------------------------------


@app.command("review")
def review_command(
    no_browser: bool = typer.Option(False, "--no-browser", help="Print the link instead of opening a browser."),
    port: int = typer.Option(0, "--port", min=0, max=65535, help="Port to listen on (default: any free one)."),
) -> None:
    """Review proposed memories, and browse, edit, and delete memories, in your browser."""
    import threading
    import time
    import webbrowser

    import uvicorn

    from openmemore import proposals, review

    base = config.home()
    with _db() as conn, audit.audited(conn, CLI_CLIENT, "find duplicates") as call:
        found = proposals.find_duplicates(conn)
        call.result_count = found
    port = port or review.free_port()
    token = review.new_token()
    url = f"http://127.0.0.1:{port}/?token={token}"
    server = uvicorn.Server(uvicorn.Config(
        review.build_app(base, token, port), host="127.0.0.1", port=port, log_level="warning",
    ))

    def when_listening() -> None:
        while not server.started and not server.should_exit:
            time.sleep(0.05)
        if server.started:
            if found:
                console.print(f"Found {found} new groups of near-duplicate memories to merge.")
            console.print(f"Review at {url}\nPress Ctrl+C to stop.", markup=False)
            if not no_browser:
                webbrowser.open(url)

    threading.Thread(target=when_listening, daemon=True).start()
    server.run()


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
                f"into {embed.model_dir()}. Keyword search still works; run `openmemore embed` to retry."
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
