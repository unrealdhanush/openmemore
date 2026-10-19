"""Transcript capture (SPEC.md M1.5): store the conversations that clients already save
on the user's disk, so capture doesn't depend on the model calling record_exchange.

Each enabled source lists its transcript files and parses one file at a time. Files
are tracked in `capture_files`: unchanged ones are skipped and growing ones re-read
from where their last turn starts, so a refresh before each read is cheap. Messages
are upserted by the transcript's own ids, and secrets are redacted as in
record_exchange.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from openmcp import db, repo
from openmcp.capture import claude_code, vscode_copilot
from openmcp.config import Config, resolve_client
from openmcp.models import CaptureFile, CaptureSource, NewConversation, NewMessage, ParsedConversation
from openmcp.safety.secrets import redact
from openmcp.tools.conversations import chunk_text, default_title, message_hash


@dataclass(frozen=True)
class Source:
    files: Callable[[Path], Iterable[Path]]
    """User home -> this source's transcript files."""
    parse: Callable[[Path, int], tuple[ParsedConversation | None, int]]
    """(file, byte offset) -> (conversation, offset to resume from next time)."""


SOURCES: dict[CaptureSource, Source] = {
    "claude-code": Source(claude_code.transcript_files, claude_code.parse),
    "vscode-copilot": Source(vscode_copilot.transcript_files, vscode_copilot.parse),
}


@dataclass
class CaptureReport:
    files_read: int = 0
    conversations: int = 0
    """Conversations that gained or changed messages."""
    messages: int = 0
    """Messages added or changed."""
    deferred: int = 0
    """Changed files left for next time because the time budget ran out."""
    errors: list[str] = field(default_factory=list)


def _store(conn: sqlite3.Connection, parsed: ParsedConversation, scope: str) -> tuple[int, int]:
    """Upsert one parsed conversation. Returns (conversation id, messages added or changed)."""
    title = default_title(redact(parsed.title)[0]) if parsed.title else None
    conversation_id = repo.upsert_conversation(
        conn,
        NewConversation(
            provider=parsed.provider,
            external_id=parsed.external_id,
            title=title,
            scope=scope,
            metadata=parsed.metadata,
            created_at=parsed.created_at,
            updated_at=parsed.updated_at,
        ),
    )
    changed = 0
    for m in parsed.messages:
        content, _ = redact(m.content)
        message_id, did_change = repo.upsert_message(
            conn,
            NewMessage(
                conversation_id=conversation_id,
                role=m.role,
                content=content,
                created_at=m.created_at,
                content_hash=message_hash(m.role, content, m.created_at),
                external_id=m.external_id,
                recall=m.recall,
            ),
        )
        if did_change:
            repo.replace_chunks(conn, message_id, chunk_text(content))
            changed += 1
    return conversation_id, changed


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
        if state is not None and (state.ignored or (state.mtime_ns, state.size) == (mtime_ns, size)):
            continue
        if budget_ms is not None and (time.perf_counter() - start) * 1000 > budget_ms:
            report.deferred += 1
            continue
        # A file that shrank was rewritten, not appended to: read it from the top.
        offset = state.resume_offset if state is not None and size >= state.size else 0
        conversation_id = state.conversation_id if state is not None else None
        try:
            parsed, resume = SOURCES[name].parse(path, offset)
        except (OSError, ValueError) as e:
            report.errors.append(f"{path}: {type(e).__name__}: {e}")
            parsed, resume = None, offset
        with db.transaction(conn):
            if parsed is not None:
                conversation_id, changed = _store(conn, parsed, resolve_client(cfg, name).write_scope)
                if changed:
                    report.conversations += 1
                    report.messages += changed
            repo.save_capture_file(
                conn,
                CaptureFile(
                    path=str(path),
                    source=name,
                    mtime_ns=mtime_ns,
                    size=size,
                    resume_offset=resume,
                    conversation_id=conversation_id,
                    captured_at=datetime.now(UTC),
                ),
            )
        report.files_read += 1
    return report
