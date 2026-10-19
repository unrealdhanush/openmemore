"""save_memories, update_memory, forget."""

from __future__ import annotations

import hashlib
import sqlite3
import unicodedata
from datetime import UTC, datetime

from mcp.server.mcpserver.exceptions import ToolError

from openmcp import db, repo
from openmcp.models import (
    ForgetResult,
    MemoryInput,
    NewMemory,
    Rejection,
    SaveMemoriesResult,
    UpdateMemoryResult,
)
from openmcp.safety.secrets import find_secret
from openmcp.tools import Caller, require_writable


def content_hash(text: str) -> str:
    """Hash of the normalized text: case, whitespace, Unicode form, and trailing
    punctuation don't make two statements different."""
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split()).rstrip(".!")
    return hashlib.sha256(normalized.encode()).hexdigest()


def _secret_in(*fields: str | None) -> str | None:
    for text in fields:
        if text and (reason := find_secret(text)):
            return reason
    return None


def save_memories(
    conn: sqlite3.Connection, caller: Caller, items: list[MemoryInput], scope: str | None = None
) -> SaveMemoriesResult:
    scope = scope or caller.scopes.write_scope
    if not caller.can_write(scope):
        raise ToolError(f"this client can't write to scope '{scope}' (it writes to '{caller.scopes.write_scope}')")

    result = SaveMemoriesResult()
    now = datetime.now(UTC)
    with db.transaction(conn):
        for index, item in enumerate(items):
            if reason := _secret_in(item.content, item.subject, item.project):
                result.rejected.append(Rejection(index=index, reason=reason))
                continue
            digest = content_hash(item.content)
            if repo.find_active_duplicate(conn, scope, digest):
                result.skipped_duplicates += 1
                continue
            memory = NewMemory(
                **item.model_dump(),
                scope=scope,
                source_client=caller.client,
                content_hash=digest,
                created_at=now,
                updated_at=now,
            )
            result.saved.append(repo.insert_memory(conn, memory))
    return result


def update_memory(
    conn: sqlite3.Connection, caller: Caller, memory_id: int, content: str, subject: str | None = None
) -> UpdateMemoryResult:
    if reason := _secret_in(content, subject):
        raise ToolError(f"not saved: {reason}")
    now = datetime.now(UTC)
    with db.transaction(conn):
        old = require_writable(caller, repo.get_memory(conn, memory_id), memory_id)
        if not old.active:
            raise ToolError(
                f"memory {memory_id} was already replaced by memory {old.superseded_by}; update that one instead"
            )
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
    return UpdateMemoryResult(new_id=new_id, previous_id=old.id)


def forget(conn: sqlite3.Connection, caller: Caller, memory_id: int) -> ForgetResult:
    """Hard-delete a memory and every version in its history."""
    with db.transaction(conn):
        require_writable(caller, repo.get_memory(conn, memory_id), memory_id)
        chain = repo.memory_chain(conn, memory_id)
        deleted = repo.delete_memories(conn, [m.id for m in chain])
    return ForgetResult(deleted=deleted)
