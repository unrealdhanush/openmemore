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

from openmcp import db, embed, relations, repo
from openmcp.ingest import store_conversation
from openmcp.ingest.detect import Export, load
from openmcp.models import IngestJob, NewMemory
from openmcp.safety.secrets import find_secret
from openmcp.tools.memories import content_hash

BATCH = 50


@dataclass
class ImportReport:
    job_id: int
    provider: str
    """The importer's name, e.g. 'chatgpt'."""
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
    memories: int = 0
    """Memories added (OpenMCP exports)."""
    memories_present: int = 0
    """Memories skipped because the same one was already stored in that scope."""


def import_export(
    conn: sqlite3.Connection,
    path: Path,
    *,
    scope: str,
    keep_scopes: bool = True,
    embedder: embed.Embedder | None = None,
    progress: Callable[[int, int], None] | None = None,
    embed_progress: Callable[[int, int], None] | None = None,
) -> ImportReport:
    """Import (or resume importing) an export file. Items keep the scope the export gives
    them (OpenMCP exports do) unless `keep_scopes` is off; others go to `scope`.
    `progress(done, total)` is called after each batch of conversations commits,
    `embed_progress(done, total)` after each batch of chunks is embedded."""
    export = load(path)
    parse = export.importer
    order = sorted(
        range(len(export.conversations)),
        key=lambda i: (-parse.updated(export.conversations[i]), parse.identity(export.conversations[i]) or ("", ""), i),
    )
    total = len(order)

    job = repo.job_by_hash(conn, export.file_hash)
    if job is not None and job.status == "done":
        return ImportReport(job_id=job.id, provider=parse.name, total=total, already_done=True)
    now = datetime.now(UTC)
    with db.transaction(conn):
        if job is None:
            job_id = repo.insert_job(conn, IngestJob(
                provider=parse.name, file_path=str(path), file_hash=export.file_hash, status="queued",
                total=total, cursor="0",
            ))
            start = 0
        else:
            job_id, start = job.id, min(int(job.cursor or 0), total)
        repo.start_job(conn, job_id, file_path=str(path), total=total, now=now)
    report = ImportReport(job_id=job_id, provider=parse.name, total=total, resumed_from=start)

    try:
        for batch_start in range(start, total, BATCH):
            batch_end = min(batch_start + BATCH, total)
            with db.transaction(conn):
                for i in order[batch_start:batch_end]:
                    raw = export.conversations[i]
                    try:
                        parsed = parse.conversation(raw)
                    except Exception as e:  # noqa: BLE001 - one malformed conversation mustn't stop the import
                        report.errors.append(f"conversation {i}: {type(e).__name__}: {e}")
                        parsed = None
                    if parsed is not None and parsed.updated_at is None and parse.updated(raw) > 0:
                        # Listings order by last activity: use the importer's sort key if the parse had none.
                        parsed = parsed.model_copy(update={"updated_at": datetime.fromtimestamp(parse.updated(raw), UTC)})
                    target = parsed.scope if parsed and keep_scopes and parsed.scope else scope
                    stored = store_conversation(conn, parsed, target) if parsed else (None, 0)
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
    _import_memories(conn, export, scope, keep_scopes, report)
    with db.transaction(conn):
        repo.finish_job(conn, job_id, "done", datetime.now(UTC))
    if embedder is not None:
        embed.embed_pending(conn, embedder, tables=("memories",))
        report.embedded = _embed_newest_first(conn, embedder, export, order, embed_progress)
    return report


def _import_memories(conn: sqlite3.Connection, export: Export, scope: str, keep_scopes: bool, report: ImportReport) -> None:
    """Current memories from an OpenMCP export. The same statement already stored in that
    scope is skipped (so importing twice adds nothing), and the secret filter applies as
    it does to save_memories."""
    report.errors += export.memory_errors
    if not export.memories:
        return
    with db.transaction(conn):
        for m in export.memories:
            if reason := next((find_secret(t) for t in (m.content, m.subject, m.project) if t and find_secret(t)), None):
                report.errors.append(f"memory not imported: {reason}")
                continue
            target = m.scope if keep_scopes and m.scope else scope
            digest = content_hash(m.content)
            if repo.find_active_duplicate(conn, target, digest):
                report.memories_present += 1
                continue
            memory_id = repo.insert_memory(conn, NewMemory(
                kind=m.kind, content=m.content, subject=m.subject, project=m.project, scope=target,
                source_client=m.source_client, confidence=m.confidence, content_hash=digest, metadata=m.metadata,
                created_at=m.created_at, updated_at=m.updated_at,
                valid_from=m.valid_from, valid_to=m.valid_to,
            ))
            relations.derived_from(conn, memory_id, [
                cid for provider, external_id in m.derived_from
                if (cid := repo.find_conversation(conn, provider, external_id)) is not None
            ], datetime.now(UTC), source="import")
            report.memories += 1


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
    # One export can hold several providers' conversations (the OpenMCP format), so
    # conversations are looked up by (provider, id), a provider at a time.
    identities = [k for k in (export.importer.identity(export.conversations[i]) for i in order) if k]
    by_provider: dict[str, list[str]] = {}
    for provider, external_id in identities:
        by_provider.setdefault(provider, []).append(external_id)
    stored: dict[tuple[str, str], int] = {}
    for provider, external_ids in by_provider.items():
        for start in range(0, len(external_ids), LOOKUP):
            found = repo.conversation_id_map(conn, provider, external_ids[start:start + LOOKUP])
            stored.update({(provider, e): cid for e, cid in found.items()})
    conversations = [stored[k] for k in identities if k in stored]
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
