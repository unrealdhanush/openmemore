"""Storing conversations that come from outside a tool call: captured transcripts
(SPEC.md M1.5) and imported exports (M3). Both arrive as ParsedConversation and are
stored the same way: secrets redacted, messages upserted by the source's own ids (so
re-reading a source never duplicates), chunked for search.
"""

from __future__ import annotations

import sqlite3

from openmemore import repo
from openmemore.models import NewConversation, NewMessage, ParsedConversation
from openmemore.safety.secrets import redact
from openmemore.tools.conversations import chunk_text, default_title, message_hash


def store_conversation(conn: sqlite3.Connection, parsed: ParsedConversation, scope: str) -> tuple[int | None, int]:
    """Upsert one conversation. Returns (conversation id, messages added or changed);
    (None, 0) if the user deleted this conversation before, which is respected."""
    if repo.was_deleted(conn, parsed.provider, parsed.external_id):
        return None, 0
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
