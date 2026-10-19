"""save_memories, update_memory, forget."""

from __future__ import annotations

import hashlib
import sqlite3
import unicodedata
from datetime import UTC, datetime

from mcp.server.mcpserver.exceptions import ToolError

from openmcp import db, embed, repo
from openmcp.models import (
    ForgetResult,
    Kind,
    MemoryInput,
    NewMemory,
    PossibleDuplicate,
    Rejection,
    SaveMemoriesResult,
    UpdateMemoryResult,
)
from openmcp.safety.secrets import find_secret
from openmcp.safety.wrap import wrap
from openmcp.tools import Caller, require_writable

NEAR_DUPLICATE_SIMILARITY = 0.92
"""SPEC.md Section 7.1: this similar and the same kind is reported as a possible duplicate."""
NEAR_DUPLICATE_CANDIDATES = 10


def content_hash(text: str) -> str:
    """Hash of the normalized text: case, whitespace, Unicode form, and trailing
    punctuation don't make two statements different."""
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split()).rstrip(".!")
    return hashlib.sha256(normalized.encode()).hexdigest()


def secret_in(*fields: str | None) -> str | None:
    for text in fields:
        if text and (reason := find_secret(text)):
            return reason
    return None


def embed_texts(texts: list[str]) -> list[bytes | None]:
    """Vectors for new memories, computed before the write transaction opens so it stays
    short. Without a model, Nones: the rows stay pending for `openmcp embed`."""
    embedder = embed.get_embedder()
    if embedder is None or not texts:
        return [None] * len(texts)
    return list(embedder.passages(texts))


def _near_duplicate(
    conn: sqlite3.Connection, caller: Caller, kind: Kind, vector: bytes
) -> tuple[int, str, str] | None:
    """The most similar active memory of the same kind that this client can read, if it
    is at least NEAR_DUPLICATE_SIMILARITY similar: (id, content, source)."""
    nearest = repo.knn_memories(
        conn, vector, k=NEAR_DUPLICATE_CANDIDATES, scopes=caller.scopes.read_scopes, kinds=[kind]
    )
    if nearest and embed.similarity(nearest[0][1]) >= NEAR_DUPLICATE_SIMILARITY:
        memory = nearest[0][0]
        return memory.id, memory.content, memory.source_client
    return None


def save_memories(
    conn: sqlite3.Connection, caller: Caller, items: list[MemoryInput], scope: str | None = None
) -> SaveMemoriesResult:
    scope = scope or caller.scopes.write_scope
    if not caller.can_write(scope):
        raise ToolError(f"this client can't write to scope '{scope}' (it writes to '{caller.scopes.write_scope}')")

    result = SaveMemoriesResult()
    accepted: list[tuple[int, MemoryInput]] = []
    for index, item in enumerate(items):
        if reason := secret_in(item.content, item.subject, item.project):
            result.rejected.append(Rejection(index=index, reason=reason))
        else:
            accepted.append((index, item))
    vectors = embed_texts([embed.memory_text(item.subject, item.content) for _, item in accepted])

    now = datetime.now(UTC)
    with db.transaction(conn):
        for (index, item), vector in zip(accepted, vectors):
            digest = content_hash(item.content)
            if repo.find_active_duplicate(conn, scope, digest):
                result.skipped_duplicates += 1
                continue
            # Near-duplicates are saved anyway and reported, so the calling model can follow
            # up with update_memory or forget. Earlier items in this call count too.
            if vector is not None and (near := _near_duplicate(conn, caller, item.kind, vector)):
                existing_id, existing_content, source = near
                result.possible_duplicates.append(
                    PossibleDuplicate(
                        new_index=index,
                        existing_id=existing_id,
                        existing_content=wrap(existing_content, source=source, id=existing_id),
                    )
                )
            memory = NewMemory(
                **item.model_dump(),
                scope=scope,
                source_client=caller.client,
                content_hash=digest,
                created_at=now,
                updated_at=now,
            )
            memory_id = repo.insert_memory(conn, memory)
            if vector is not None:
                repo.store_vectors(conn, "memories", [(memory_id, vector)])
            result.saved.append(memory_id)
    return result


def update_memory(
    conn: sqlite3.Connection, caller: Caller, memory_id: int, content: str, subject: str | None = None
) -> UpdateMemoryResult:
    if reason := secret_in(content, subject):
        raise ToolError(f"not saved: {reason}")
    # The subject may come from the old version, which needs a read first; embedding
    # needs the final text. Read, embed outside the transaction, then write.
    current = require_writable(caller, repo.get_memory(conn, memory_id), memory_id)
    [vector] = embed_texts([embed.memory_text(subject if subject is not None else current.subject, content)])
    now = datetime.now(UTC)
    with db.transaction(conn):
        old = require_writable(caller, repo.get_memory(conn, memory_id), memory_id)
        if not old.active:
            raise ToolError(
                f"memory {memory_id} was already replaced by memory {old.superseded_by}; update that one instead"
            )
        if subject is None and old.subject != current.subject:
            vector = None  # changed meanwhile: leave it to `openmcp embed`
        new = NewMemory(
            **old.model_dump(include={"kind", "project", "scope", "confidence", "metadata"}),
            content=content,
            subject=subject if subject is not None else old.subject,
            source_client=caller.client,
            content_hash=content_hash(content),
            created_at=now,
            updated_at=now,
        )
        new_id = repo.insert_memory(conn, new)
        repo.mark_superseded(conn, old.id, new_id, now)
        if vector is not None:
            repo.store_vectors(conn, "memories", [(new_id, vector)])
    return UpdateMemoryResult(new_id=new_id, previous_id=old.id)


def forget(conn: sqlite3.Connection, caller: Caller, memory_id: int) -> ForgetResult:
    """Hard-delete a memory and every version in its history."""
    with db.transaction(conn):
        require_writable(caller, repo.get_memory(conn, memory_id), memory_id)
        chain = repo.memory_chain(conn, memory_id)
        deleted = repo.delete_memories(conn, [m.id for m in chain])
    return ForgetResult(deleted=deleted)
