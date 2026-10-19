"""`openmcp import`: a resumable job per export file (SPEC.md Section 10).

Two phases. First every conversation is stored, newest first, 50 per transaction, with
the job's cursor advancing in the same transaction as the batch it covers: storing is
fast (~10k messages a second), so the whole export is keyword-searchable within
seconds, and an interrupted import resumes exactly where it stopped. Then the export
is embedded, again newest first (~70 messages a second with bge-small on a laptop); if
that's interrupted, `openmcp embed` finishes it. Embedding inside each batch instead
would hold the oldest conversations back from even keyword search for as long as
embedding everything takes.

Re-running a finished import of the same file is a no-op; a newer export of the same
account adds what's new (messages are upserted by the provider's ids). This runs in
the CLI, never inside `serve`.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from openmcp import db, embed, repo
from openmcp.ingest import store_conversation
from openmcp.ingest.detect import PARSERS, Export, load
from openmcp.models import ExportProvider, IngestJob

BATCH = 50


@dataclass
class ImportReport:
    job_id: int
    provider: ExportProvider
    total: int
    already_done: bool = False
    resumed_from: int = 0
    conversations: int = 0
    """Conversations that gained or changed messages."""
    messages: int = 0
    skipped: int = 0
    """Conversations with nothing to keep (e.g. only system messages), or deleted by the user."""
    errors: list[str] = field(default_factory=list)
    embedded: int = 0
    """Chunks embedded in this run."""


def import_export(
    conn: sqlite3.Connection,
    path: Path,
    *,
    scope: str,
    embedder: embed.Embedder | None = None,
    progress: Callable[[int, int], None] | None = None,
    embed_progress: Callable[[int, int], None] | None = None,
) -> ImportReport:
    """Import (or resume importing) an export file. `progress(done, total)` is called
    after each batch of conversations commits, `embed_progress(done, total)` after each
    batch of chunks is embedded."""
    export = load(path)
    parse = PARSERS[export.provider]
    order = sorted(
        range(len(export.conversations)),
        key=lambda i: (-parse.updated(export.conversations[i]), parse.external_id(export.conversations[i]) or "", i),
    )
    total = len(order)

    job = repo.job_by_hash(conn, export.file_hash)
    if job is not None and job.status == "done":
        return ImportReport(job_id=job.id, provider=export.provider, total=total, already_done=True)
    now = datetime.now(UTC)
    with db.transaction(conn):
        if job is None:
            job_id = repo.insert_job(conn, IngestJob(
                provider=export.provider, file_path=str(path), file_hash=export.file_hash, status="queued",
                total=total, cursor="0",
            ))
            start = 0
        else:
            job_id, start = job.id, min(int(job.cursor or 0), total)
        repo.start_job(conn, job_id, file_path=str(path), total=total, now=now)
    report = ImportReport(job_id=job_id, provider=export.provider, total=total, resumed_from=start)

    try:
        for batch_start in range(start, total, BATCH):
            batch_end = min(batch_start + BATCH, total)
            with db.transaction(conn):
                for i in order[batch_start:batch_end]:
                    try:
                        parsed = parse.conversation(export.conversations[i])
                    except Exception as e:  # noqa: BLE001 - one malformed conversation mustn't stop the import
                        report.errors.append(f"conversation {i}: {type(e).__name__}: {e}")
                        parsed = None
                    stored = store_conversation(conn, parsed, scope) if parsed else (None, 0)
                    if stored[0] is None:
                        report.skipped += 1
                    elif stored[1]:
                        report.conversations += 1
                        report.messages += stored[1]
                repo.advance_job(conn, job_id, batch_end)
            if progress:
                progress(batch_end, total)
    except BaseException as e:
        with db.transaction(conn):
            reason = "interrupted" if isinstance(e, KeyboardInterrupt) else f"{type(e).__name__}: {e}"
            repo.finish_job(conn, job_id, "failed", datetime.now(UTC), reason)
        raise
    with db.transaction(conn):
        repo.finish_job(conn, job_id, "done", datetime.now(UTC))
    if embedder is not None:
        report.embedded = _embed_newest_first(conn, embedder, export, order, embed_progress)
    return report


LOOKUP = 500
"""External ids resolved per query: well under SQLite's limit on query parameters."""


def _embed_newest_first(
    conn: sqlite3.Connection,
    embedder: embed.Embedder,
    export: Export,
    order: list[int],
    progress: Callable[[int, int], None] | None,
) -> int:
    """Embed the export's chunks, newest conversations first, BATCH conversations at a
    time. (Plain embed_pending goes by row id, and here the newest conversations got the
    lowest ids.) Progress counts conversations, like the store phase."""
    parse = PARSERS[export.provider]
    external = [e for e in (parse.external_id(export.conversations[i]) for i in order) if e]
    conversations: list[int] = []
    for start in range(0, len(external), LOOKUP):
        conversations += repo.conversation_ids(conn, export.provider, external[start:start + LOOKUP])
    done = 0
    for start in range(0, len(conversations), BATCH):
        chunks = repo.pending_chunks_in(conn, conversations[start:start + BATCH])
        for first in range(0, len(chunks), embed.BATCH):
            batch = chunks[first:first + embed.BATCH]
            vectors = embedder.passages([text for _, text in batch])
            with db.transaction(conn):
                done += repo.store_vectors(conn, "chunks", [(cid, v) for (cid, _), v in zip(batch, vectors)])
        if progress:
            progress(min(start + BATCH, len(conversations)), len(conversations))
    return done
