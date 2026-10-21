"""OpenMemore's own format (SPEC.md M3.5, M4): what `openmemore export` writes and `openmemore
import` reads back, and a provider-neutral way in for any assistant's history: write a
small converter from any export (Gemini via Google Takeout, Perplexity, ...) to this.

    {
      "format": "openmemore.export",
      "version": 1,
      "exported_at": "2026-10-19T10:00:00Z",       // optional
      "memories": [                                // optional
        {"id": 3, "kind": "preference", "content": "...", "subject": null, "project": null,
         "scope": "personal", "source": "claude-ai", "confidence": null, "metadata": {},
         "created_at": "...", "updated_at": "...", "superseded_by": null,
         "valid_from": "...", "valid_to": null, "superseded_at": null,  // when it was true (optional)
         "derived_from": [{"provider": "...", "id": "..."}]}          // its source conversations (optional)
      ],
      "conversations": [
        {
          "provider": "gemini",              // a short name: letters, digits, '.', '_', '-'
          "id": "the provider's conversation id",
          "title": "optional",
          "scope": "personal",                       // optional
          "created_at": "2026-10-01T10:00:00Z",      // optional, ISO 8601 with a time zone
          "updated_at": "2026-10-01T10:05:00Z",      // optional
          "metadata": {},                            // optional
          "messages": [
            {"id": "...", "role": "user" | "assistant", "content": "...", "created_at": "...", "recall": false}
          ]
        }
      ]
    }

Superseded memories (earlier versions, `superseded_by` set) are exported as history and
not imported: importing restores what's current.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from openmemore.ingest.registry import Importer
from openmemore.models import Kind, ParsedConversation, ParsedMemory, ParsedMessage

FORMAT = "openmemore.export"
VERSION = 1

Name = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")]
Text = Annotated[str, StringConstraints(min_length=1)]
ScopeName = Annotated[str, StringConstraints(min_length=1, max_length=64)]


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: Text
    role: Literal["user", "assistant"]
    content: str
    created_at: AwareDatetime | None = None
    recall: bool = False


class Conversation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: Name
    id: Text
    title: str | None = None
    scope: ScopeName | None = None
    created_at: AwareDatetime | None = None
    updated_at: AwareDatetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    messages: list[Message] = Field(default_factory=list)


class Memory(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Kind
    content: Text
    subject: str | None = None
    project: str | None = None
    scope: ScopeName | None = None
    source: Text = "import:openmemore"
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: AwareDatetime
    updated_at: AwareDatetime
    superseded_by: int | None = None
    valid_from: AwareDatetime | None = None
    valid_to: AwareDatetime | None = None
    derived_from: list[dict[str, Any]] = Field(default_factory=list)


def matches(doc: Any) -> bool:
    return isinstance(doc, dict) and doc.get("format") == FORMAT


def _check_version(doc: Any) -> None:
    if doc.get("version") != VERSION:
        raise ValueError(f"{FORMAT} version {doc.get('version')!r} isn't supported (this version reads {VERSION})")


def items(doc: Any) -> list[Any]:
    _check_version(doc)
    conversations = doc.get("conversations", [])
    if not isinstance(conversations, list):
        raise ValueError(f"{FORMAT} needs 'conversations' to be a list")
    return conversations


def memories(doc: Any) -> tuple[list[ParsedMemory], list[str]]:
    """Current memories (not earlier versions), and why any were skipped."""
    _check_version(doc)
    raw = doc.get("memories", [])
    if not isinstance(raw, list):
        raise ValueError(f"{FORMAT} needs 'memories' to be a list")
    kept, errors = [], []
    for n, item in enumerate(raw):
        try:
            m = Memory.model_validate(item)
            if m.superseded_by is not None:
                continue
            kept.append(ParsedMemory(
                kind=m.kind, content=m.content, subject=m.subject, project=m.project, scope=m.scope,
                source_client=m.source, confidence=m.confidence, metadata=m.metadata,
                created_at=m.created_at, updated_at=m.updated_at,
                valid_from=m.valid_from, valid_to=m.valid_to,
                derived_from=[(str(d["provider"]), str(d["id"])) for d in m.derived_from
                              if isinstance(d, dict) and d.get("provider") and d.get("id")],
            ))
        except ValidationError as e:
            errors.append(f"memory {n}: {e.errors()[0]['loc']}: {e.errors()[0]['msg']}")
    return kept, errors


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
        ParsedMessage(external_id=m.id, role=m.role, content=m.content.strip(), created_at=m.created_at, recall=m.recall)
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
        metadata=conv.metadata or {"capture": "import"},
        messages=messages,
        scope=conv.scope,
    )


IMPORTER = Importer(
    name="openmemore", display="OpenMemore export", matches=matches, items=items,
    updated=lambda raw: updated(raw), identity=lambda raw: identity(raw), conversation=lambda raw: conversation(raw),
    memories=lambda doc: memories(doc),
)
