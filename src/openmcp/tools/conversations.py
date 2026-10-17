"""record_exchange and recent_conversations: live capture of what the user asked
each assistant, so another assistant can pick up where it left off."""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from datetime import UTC, datetime

from mcp.server.mcpserver.exceptions import ToolError

from openmcp import db, repo
from openmcp.models import (
    RECENT_MESSAGES_PER_CONVERSATION,
    ConversationMessage,
    ConversationView,
    NewConversation,
    NewMessage,
    RecentConversations,
    RecordExchangeResult,
    Role,
)
from openmcp.safety.secrets import redact
from openmcp.safety.wrap import wrap
from openmcp.tools import Caller

CHUNK_CHARS = 1000
TITLE_CHARS = 80
SINGLE_CONVERSATION_MESSAGES = 50


def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Split text into pieces of at most `size` chars, preferring paragraph, then
    sentence, then word boundaries."""
    text = text.strip()
    chunks = []
    while len(text) > size:
        window = text[:size]
        for sep in ("\n\n", "\n", ". ", " "):
            cut = window.rfind(sep)
            if cut > size // 2:
                cut += len(sep)
                break
        else:
            cut = size
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks


def message_hash(role: Role, content: str, created_at: datetime | None) -> str:
    stamp = repo.to_iso(created_at) if created_at else ""
    return hashlib.sha256(f"{role}\n{stamp}\n{content}".encode()).hexdigest()


def _default_title(question: str) -> str:
    first_line = question.strip().splitlines()[0]
    return first_line if len(first_line) <= TITLE_CHARS else first_line[: TITLE_CHARS - 1] + "…"


def _add_message(conn: sqlite3.Connection, conversation_id: int, role: Role, content: str, now: datetime) -> int:
    """Same role, content, and timestamp in the same conversation is the same message
    (e.g. a retried call), so it is stored once."""
    message_id, inserted = repo.insert_message(
        conn,
        NewMessage(
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=now,
            content_hash=message_hash(role, content, now),
        ),
    )
    if inserted:
        repo.insert_chunks(conn, message_id, chunk_text(content))
    return message_id


def record_exchange(
    conn: sqlite3.Connection,
    caller: Caller,
    question: str,
    answer_summary: str,
    conversation_id: int | None = None,
    title: str | None = None,
) -> RecordExchangeResult:
    question, q_reasons = redact(question)
    answer_summary, a_reasons = redact(answer_summary)
    reasons = list(dict.fromkeys(q_reasons + a_reasons))
    now = datetime.now(UTC)

    with db.transaction(conn):
        if conversation_id is not None:
            conv = repo.get_conversation(conn, conversation_id)
            # A client may only append to conversations it started itself.
            if conv is None or conv.provider != caller.client or not caller.can_write(conv.scope):
                raise ToolError(
                    f"conversation {conversation_id} not found for this assistant; "
                    "omit conversation_id to start a new one"
                )
            repo.touch_conversation(conn, conv.id, now)
        else:
            conversation_id = repo.insert_conversation(
                conn,
                NewConversation(
                    provider=caller.client,
                    external_id=uuid.uuid4().hex,
                    title=title or _default_title(question),
                    scope=caller.scopes.write_scope,
                    metadata={"capture": "record_exchange"},
                    created_at=now,
                    updated_at=now,
                ),
            )
        ids = [
            _add_message(conn, conversation_id, "user", question, now),
            _add_message(conn, conversation_id, "assistant", answer_summary, now),
        ]
    return RecordExchangeResult(conversation_id=conversation_id, message_ids=ids, redacted=reasons)


def recent_conversations(
    conn: sqlite3.Connection,
    caller: Caller,
    source: str | None = None,
    conversation_id: int | None = None,
    limit: int = 5,
) -> RecentConversations:
    convs = repo.list_conversations(
        conn, scopes=caller.scopes.read_scopes, source=source, conversation_id=conversation_id, limit=limit
    )
    if conversation_id is not None and not convs:
        raise ToolError(f"conversation {conversation_id} not found")
    counts = repo.message_counts(conn, [c.id for c in convs])
    # A single requested conversation shows much more of itself; listings show the latest turns.
    last = SINGLE_CONVERSATION_MESSAGES if conversation_id is not None else RECENT_MESSAGES_PER_CONVERSATION
    return RecentConversations(
        conversations=[
            ConversationView(
                id=c.id,
                title=c.title,
                source=c.provider,
                started_at=c.created_at,
                updated_at=c.updated_at,
                message_count=counts.get(c.id, 0),
                messages=[
                    ConversationMessage(
                        role=m.role, content=wrap(m.content, source=c.provider, id=m.id), created_at=m.created_at
                    )
                    for m in repo.conversation_messages(conn, c.id, last=last)
                ],
            )
            for c in convs
        ]
    )
