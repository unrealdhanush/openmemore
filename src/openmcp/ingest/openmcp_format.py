"""OpenMCP's own conversation format: a provider-neutral way in for any assistant's
history. Write a small converter from any export (Gemini via Google Takeout,
Perplexity, ...) to this, and `openmcp import` handles the rest. `openmcp export`
(M4) writes the same format, so exports round-trip.

    {
      "format": "openmcp.conversations",
      "version": 1,
      "conversations": [
        {
          "provider": "gemini",              // a short name: letters, digits, '.', '_', '-'
          "id": "the provider's conversation id",
          "title": "optional",
          "created_at": "2026-10-01T10:00:00Z",   // optional, ISO 8601 with a time zone
          "updated_at": "2026-10-01T10:05:00Z",   // optional
          "messages": [
            {"id": "...", "role": "user" | "assistant", "content": "...", "created_at": "..."}
          ]
        }
      ]
    }
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints

from openmcp.ingest.registry import Importer
from openmcp.models import ParsedConversation, ParsedMessage

FORMAT = "openmcp.conversations"
VERSION = 1

Name = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")]
Text = Annotated[str, StringConstraints(min_length=1)]


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: Text
    role: Literal["user", "assistant"]
    content: str
    created_at: AwareDatetime | None = None


class Conversation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: Name
    id: Text
    title: str | None = None
    created_at: AwareDatetime | None = None
    updated_at: AwareDatetime | None = None
    messages: list[Message] = Field(default_factory=list)


def matches(doc: Any) -> bool:
    return isinstance(doc, dict) and doc.get("format") == FORMAT


def items(doc: Any) -> list[Any]:
    if doc.get("version") != VERSION:
        raise ValueError(f"{FORMAT} version {doc.get('version')!r} isn't supported (this version reads {VERSION})")
    conversations = doc.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError(f"{FORMAT} needs a 'conversations' list")
    return conversations


def _seconds(value: Any) -> float:
    if not isinstance(value, str):
        return 0.0
    try:
        when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return when.timestamp() if when.tzinfo else 0.0


def updated(raw: Any) -> float:
    if not isinstance(raw, dict):
        return 0.0
    return _seconds(raw.get("updated_at")) or _seconds(raw.get("created_at"))


def identity(raw: Any) -> tuple[str, str] | None:
    if isinstance(raw, dict) and isinstance(raw.get("provider"), str) and isinstance(raw.get("id"), str):
        return raw["provider"], raw["id"]
    return None


def conversation(raw: Any) -> ParsedConversation | None:
    """Raises ValueError (with Pydantic's explanation) for a conversation that doesn't fit
    the format, so the import reports it."""
    conv = Conversation.model_validate(raw)
    messages = [
        ParsedMessage(external_id=m.id, role=m.role, content=m.content.strip(), created_at=m.created_at)
        for m in conv.messages
        if m.content.strip()
    ]
    if not messages:
        return None
    return ParsedConversation(
        provider=conv.provider,
        external_id=conv.id,
        title=conv.title.strip() if conv.title and conv.title.strip() else messages[0].content,
        created_at=conv.created_at or messages[0].created_at,
        updated_at=conv.updated_at or messages[-1].created_at,
        metadata={"capture": "import"},
        messages=messages,
    )


IMPORTER = Importer(
    name="openmcp", display="OpenMCP format", matches=matches, items=items,
    updated=lambda raw: updated(raw), identity=lambda raw: identity(raw), conversation=lambda raw: conversation(raw),
)
