"""record_exchange and recent_conversations: live capture of what the user asked
each assistant, so another assistant can pick up where it left off."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from mcp.server.mcpserver.exceptions import ToolError

from openmemore import db, embed, repo
from openmemore.models import (
    MAX_CODE_CHARS,
    RECENT_MESSAGES_PER_CONVERSATION,
    CodeBlock,
    ConversationMessage,
    ConversationView,
    NewConversation,
    NewMessage,
    RecentConversations,
    RecordExchangeResult,
    Role,
)
from openmemore.safety.secrets import redact
from openmemore.safety.wrap import wrap
from openmemore.tools import Caller

CHUNK_CHARS = 1000
TITLE_CHARS = 80
SINGLE_CONVERSATION_MESSAGES = 50
FULL_MESSAGE_CHARS = MAX_CODE_CHARS + 5000
"""Per-message cap when one conversation is read in full: room for a code block and its summary."""


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


def with_code(summary: str, code: Sequence[CodeBlock]) -> str:
    """The answer summary followed by each code block, fenced so it renders as code."""
    parts = [summary]
    for block in code:
        longest = max((len(run) for run in re.findall(r"`+", block.content)), default=0)
        fence = "`" * max(3, longest + 1)
        label = f"`{block.path}`:\n" if block.path else ""
        body = block.content.rstrip("\n")
        parts.append(f"{label}{fence}{block.language or ''}\n{body}\n{fence}")
    return "\n\n".join(parts)


def default_title(question: str) -> str:
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


def not_recorded(caller: Caller) -> str:
    """Why an exchange wasn't stored, for a client whose conversations aren't kept."""
    then = " Save lasting facts and decisions with save_memories instead." if caller.scopes.saves_memories else ""
    return (f"Not stored: this assistant's conversations aren't kept (writes = \"{caller.scopes.writes}\" in "
            f"OpenMemore's config.toml), so record_exchange isn't needed here.{then}")


def record_exchange(
    conn: sqlite3.Connection,
    caller: Caller,
    question: str,
    answer_summary: str,
    conversation_id: int | None = None,
    title: str | None = None,
    code: Sequence[CodeBlock] = (),
) -> RecordExchangeResult:
    if not caller.scopes.records:
        return RecordExchangeResult(conversation_id=None, note=not_recorded(caller))
    question, q_reasons = redact(question)
    answer, a_reasons = redact(with_code(answer_summary, code))
    title, t_reasons = redact(title) if title else (None, [])
    reasons = list(dict.fromkeys(q_reasons + a_reasons + t_reasons))
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
                    title=title or default_title(question),
                    scope=caller.scopes.write_scope,
                    metadata={"capture": "record_exchange"},
                    created_at=now,
                    updated_at=now,
                ),
            )
        ids = [
            _add_message(conn, conversation_id, "user", question, now),
            _add_message(conn, conversation_id, "assistant", answer, now),
        ]
    # After the write, so the transaction stays short; newest first, so this exchange's
    # chunks are embedded before any backlog.
    embed.embed_pending(conn, embed.get_embedder(), budget_ms=embed.WRITE_BUDGET_MS)
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
    # A single requested conversation shows much more of itself, in full; listings show
    # the latest turns, each cut to the default length.
    single = conversation_id is not None
    last = SINGLE_CONVERSATION_MESSAGES if single else RECENT_MESSAGES_PER_CONVERSATION
    wrap_options = {"max_chars": FULL_MESSAGE_CHARS} if single else {}
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
                        role=m.role,
                        content=wrap(m.content, source=c.provider, id=m.id, **wrap_options),
                        created_at=m.created_at,
                    )
                    for m in repo.conversation_messages(conn, c.id, last=last)
                ],
            )
            for c in convs
        ]
    )
