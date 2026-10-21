"""Transcript capture: store the conversations that clients already save
on the user's disk, so capture doesn't depend on the model calling record_exchange.

Each enabled source lists its transcript files and parses one file at a time. Files
are tracked in `capture_files`: unchanged ones are skipped and growing ones re-read
from where their last turn starts, so a refresh before each read is cheap. Messages
are upserted by the transcript's own ids, and secrets are redacted as in
record_exchange (openmemore.ingest.store_conversation, shared with imports).
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from openmemore import db, repo
from openmemore.capture import claude_code, vscode_copilot
from openmemore.clients import client_name
from openmemore.config import Config, resolve_client
from openmemore.ingest import store_conversation
from openmemore.models import CaptureFile, CaptureSource, ParsedConversation


@dataclass(frozen=True)
class Source:
    """A transcript reader: the only client-specific code OpenMemore needs, and optional
    (record_exchange works for every client). It names the application it reads, as
    that application reports itself over MCP."""

    files: Callable[[Path], Iterable[Path]]
    """User home -> this source's transcript files."""
    parse: Callable[[Path, int], tuple[ParsedConversation | None, int]]
    """(file, byte offset) -> (conversation, offset to resume from next time)."""
    app: str
    """The application's MCP clientInfo name, confirmed from a real session."""


SOURCES: dict[CaptureSource, Source] = {
    "claude-code": Source(claude_code.transcript_files, claude_code.parse, app="claude-code"),
    "vscode-copilot": Source(vscode_copilot.transcript_files, vscode_copilot.parse, app="Visual Studio Code"),
}


def _normal(folder: str) -> str:
    path = os.path.normpath(os.path.expanduser(folder))
    # macOS and Windows file systems ignore case by default.
    return path.casefold() if sys.platform in ("darwin", "win32") else path


def _within(folder: str, roots: Iterable[str]) -> bool:
    f = _normal(folder)
    return any(f == r or f.startswith(r.rstrip(os.sep) + os.sep) for r in map(_normal, roots))


def folder_captured(folder: str | None, cfg: Config) -> bool:
    """Whether a conversation from this project folder is captured: not inside an
    excluded folder, and inside an included one if any are listed. A conversation with
    no known folder is captured unless include_folders is set."""
    include, exclude = cfg.capture.include_folders, cfg.capture.exclude_folders
    if folder is None:
        return not include
    if _within(folder, exclude):
        return False
    return not include or _within(folder, include)


def captured_source(app: str | None, enabled: Iterable[str]) -> str | None:
    """The enabled transcript reader for this application, if any."""
    for name in enabled:
        if app is not None and SOURCES[name].app == app:
            return name
    return None


@dataclass
class CaptureReport:
    files_read: int = 0
    conversations: int = 0
    """Conversations that gained or changed messages."""
    messages: int = 0
    """Messages added or changed."""
    deferred: int = 0
    """Changed files left for next time because the time budget ran out."""
    filtered: int = 0
    """Changed files not read: their folder isn't captured ([capture] include/exclude_folders)."""
    errors: list[str] = field(default_factory=list)


def capture(
    conn: sqlite3.Connection,
    cfg: Config,
    *,
    sources: Sequence[CaptureSource] | None = None,
    user_home: Path | None = None,
    days: int | None = None,
    budget_ms: float | None = None,
) -> CaptureReport:
    """Capture new and changed transcripts from `sources` (default: the enabled ones),
    newest first. With `budget_ms`, stop starting new files once that much time has passed."""
    start = time.perf_counter()
    home = user_home or Path.home()
    cutoff = (datetime.now(UTC) - timedelta(days=days or cfg.capture.days)).timestamp() * 1e9
    report = CaptureReport()

    candidates: list[tuple[int, CaptureSource, Path, int]] = []
    for name in cfg.capture.sources if sources is None else sources:
        for path in SOURCES[name].files(home):
            try:
                st = path.stat()
            except OSError:
                continue
            if st.st_mtime_ns >= cutoff:
                candidates.append((st.st_mtime_ns, name, path, st.st_size))
    candidates.sort(key=lambda c: c[0], reverse=True)

    for mtime_ns, name, path, size in candidates:
        state = repo.get_capture_file(conn, str(path))
        if state is not None and state.ignored:
            continue
        changed = state is None or (state.mtime_ns, state.size) != (mtime_ns, size)
        if state is not None and not folder_captured(state.folder, cfg):
            # Its folder isn't captured: noted, not read (a file with no known folder is
            # read again only when it changes, in case it now says where it belongs).
            if changed and state.folder is not None:
                repo.save_capture_file(conn, state.model_copy(update={
                    "mtime_ns": mtime_ns, "size": size, "filtered": True, "captured_at": datetime.now(UTC)}))
                report.filtered += 1
                continue
            if not changed or state.folder is not None:
                continue
        elif not changed and not state.filtered:
            continue
        if budget_ms is not None and (time.perf_counter() - start) * 1000 > budget_ms:
            report.deferred += 1
            continue
        # A file that shrank was rewritten, not appended to; one skipped before for its
        # folder was never stored: read either from the top.
        whole = state is None or state.filtered or size < state.size
        offset = 0 if whole else state.resume_offset
        conversation_id = state.conversation_id if state is not None else None
        try:
            parsed, resume = SOURCES[name].parse(path, offset)
        except Exception as e:  # noqa: BLE001 - one unreadable transcript mustn't stop the others
            report.errors.append(f"{path}: {type(e).__name__}: {e}")
            parsed, resume = None, offset
        folder = parsed.metadata.get("folder") if parsed is not None else (state.folder if state else None)
        folder = folder if isinstance(folder, str) else None
        allowed = folder_captured(folder, cfg)
        with db.transaction(conn):
            if parsed is not None and allowed:
                # Stored under the reader's descriptive name ('vscode-copilot'), in the scope of
                # the application it reads, the same client its live calls come from.
                scope = resolve_client(cfg, client_name(SOURCES[name].app)).write_scope
                conversation_id, stored = store_conversation(conn, parsed, scope)
                if stored:
                    report.conversations += 1
                    report.messages += stored
            elif parsed is not None:
                report.filtered += 1
            repo.save_capture_file(
                conn,
                CaptureFile(
                    path=str(path),
                    source=name,
                    mtime_ns=mtime_ns,
                    size=size,
                    resume_offset=resume,
                    conversation_id=conversation_id,
                    folder=folder,
                    filtered=not allowed,
                    captured_at=datetime.now(UTC),
                ),
            )
        report.files_read += 1
    return report
