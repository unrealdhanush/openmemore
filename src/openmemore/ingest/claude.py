"""Claude data export: conversations.json, a list of conversations.

Each conversation has `uuid`, `name`, ISO timestamps, and `chat_messages` in order;
each message has `sender` ('human' or 'assistant'), `created_at`, and either `content`
blocks or a plain `text`. Text blocks are kept; tool use, tool results, and thinking
are skipped. Unknown fields are ignored.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from openmemore.ingest.registry import Importer
from openmemore.models import ParsedConversation, ParsedMessage

PROVIDER = "claude"
ROLES = {"human": "user", "assistant": "assistant"}


def _when(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return when if when.tzinfo else None


def updated(raw: Any) -> float:
    """Sort key: last activity, in seconds (0 if unknown)."""
    if not isinstance(raw, dict):
        return 0.0
    when = _when(raw.get("updated_at")) or _when(raw.get("created_at"))
    return when.timestamp() if when else 0.0


def external_id(raw: Any) -> str | None:
    value = raw.get("uuid") if isinstance(raw, dict) else None
    return value if isinstance(value, str) and value else None


def _text(message: dict[str, Any]) -> str | None:
    content = message.get("content")
    if isinstance(content, list) and content:
        texts = [
            b["text"] for b in content
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
        ]
        text = "\n\n".join(t for t in texts if t.strip())
    else:
        text = message.get("text") if isinstance(message.get("text"), str) else ""
    return text.strip() or None


def conversation(raw: Any) -> ParsedConversation | None:
    """One exported conversation, or None if it has no id or no messages worth keeping."""
    conv_id = external_id(raw)
    chat = raw.get("chat_messages") if conv_id else None
    if not isinstance(chat, list):
        return None
    messages = []
    for m in chat:
        if not isinstance(m, dict) or m.get("sender") not in ROLES or not isinstance(m.get("uuid"), str):
            continue
        text = _text(m)
        if text is None:
            continue
        messages.append(
            ParsedMessage(external_id=m["uuid"], role=ROLES[m["sender"]], content=text, created_at=_when(m.get("created_at")))
        )
    if not messages:
        return None
    name = raw.get("name")
    return ParsedConversation(
        provider=PROVIDER,
        external_id=conv_id,
        title=name if isinstance(name, str) and name.strip() else messages[0].content,
        created_at=_when(raw.get("created_at")) or messages[0].created_at,
        updated_at=_when(raw.get("updated_at")) or messages[-1].created_at,
        metadata={"capture": "import"},
        messages=messages,
    )


def _identity(raw: Any) -> tuple[str, str] | None:
    conv_id = external_id(raw)
    return (PROVIDER, conv_id) if conv_id else None


# Lambdas look the functions up at call time, so tests can replace them.
IMPORTER = Importer(
    name=PROVIDER, display="Claude",
    matches=lambda doc: isinstance(doc, list) and any(isinstance(c, dict) and "chat_messages" in c for c in doc),
    items=lambda doc: doc, updated=lambda raw: updated(raw), identity=lambda raw: _identity(raw),
    conversation=lambda raw: conversation(raw),
)
