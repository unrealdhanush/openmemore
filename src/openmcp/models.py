"""Pydantic models for every boundary: MCP tool arguments/results, CLI inputs, stored rows.

Nothing reaches the repository (repo.py) without passing through one of these.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints

from openmcp.safety.wrap import NOTICE

ClientName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"),
]
"""Name a client connects as (`openmcp serve --client <name>`); recorded in the audit log."""

Scope = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]

Kind = Literal["fact", "preference", "project", "decision", "note"]
KINDS: tuple[Kind, ...] = ("fact", "preference", "project", "decision", "note")

Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
"""Short free-text labels: subject and project."""

MemoryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]

MemoryId = Annotated[int, Field(ge=1)]

Role = Literal["user", "assistant", "system", "tool"]

SUMMARY_MAX = 200
MAX_SAVE_ITEMS = 100
MAX_SEARCH_LIMIT = 25
PROFILE_GROUP_CAP = 15
ACTIVE_PROJECT_DAYS = 30
MAX_QUESTION_CHARS = 8000
MAX_ANSWER_CHARS = 4000
MAX_RECENT_CONVERSATIONS = 20
RECENT_MESSAGES_PER_CONVERSATION = 6


# --- Stored rows ---------------------------------------------------------------------


class AuditEntry(BaseModel):
    """One row of audit_log: a single tool call by a single client."""

    model_config = ConfigDict(frozen=True)

    id: int | None = None
    ts: AwareDatetime
    client: str
    tool: str
    summary: str | None = Field(default=None, max_length=SUMMARY_MAX)
    result_count: int | None = Field(default=None, ge=0)
    duration_ms: float = Field(ge=0)


class NewMemory(BaseModel):
    """A memory ready to insert: validated, secret-filtered, hashed."""

    kind: Kind
    content: MemoryText
    subject: Label | None = None
    project: Label | None = None
    scope: Scope
    source_client: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: AwareDatetime
    updated_at: AwareDatetime


class Memory(NewMemory):
    """A row of `memories`."""

    id: int
    superseded_by: int | None = None

    @property
    def active(self) -> bool:
        return self.superseded_by is None


class NewConversation(BaseModel):
    """A conversation to insert. For exchanges captured live, `provider` is the client name."""

    provider: str
    external_id: str
    title: str | None = None
    scope: Scope
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: AwareDatetime | None = None
    updated_at: AwareDatetime | None = None


class Conversation(NewConversation):
    """A row of `conversations`."""

    id: int


class NewMessage(BaseModel):
    conversation_id: int
    role: Role
    content: str = Field(min_length=1)
    created_at: AwareDatetime | None = None
    content_hash: str


class Message(NewMessage):
    """A row of `messages`."""

    id: int


class ChunkMatch(BaseModel):
    """A chunk found by full-text search, with the conversation it belongs to."""

    chunk_id: int
    text: str
    role: Role
    created_at: datetime | None
    conversation_id: int
    title: str | None
    provider: str
    rank: float
    """bm25: lower is better."""


# --- MCP tool arguments --------------------------------------------------------------


class MemoryInput(BaseModel):
    """One item passed to `save_memories`."""

    model_config = ConfigDict(extra="forbid")

    kind: Kind = Field(
        description="fact: something true about the user. preference: how they like things done. "
        "project: an ongoing project and its current state. decision: a choice that was made and why. "
        "note: anything else worth keeping."
    )
    content: MemoryText = Field(
        description="One self-contained statement that makes sense with no other context, "
        "e.g. 'Prefers Python for scripting and terse answers.'"
    )
    subject: Label | None = Field(default=None, description="Short label, e.g. 'coding style'.")
    project: Label | None = Field(default=None, description="Project name, if this belongs to one.")
    confidence: float | None = Field(default=None, ge=0, le=1, description="How sure you are, 0 to 1.")


# --- MCP tool results ----------------------------------------------------------------


class ReadResult(BaseModel):
    """Base for every result that returns stored content. `notice` serializes first."""

    notice: str = NOTICE


class PingResult(BaseModel):
    """Result of the `ping` tool."""

    ok: bool
    version: str
    client: str
    read_scopes: list[str]
    write_scope: str
    schema_version: int


class PossibleDuplicate(BaseModel):
    new_index: int
    existing_id: int
    existing_content: str


class Rejection(BaseModel):
    index: int
    reason: str


class SaveMemoriesResult(BaseModel):
    saved: list[int] = Field(default_factory=list)
    skipped_duplicates: int = 0
    possible_duplicates: list[PossibleDuplicate] = Field(default_factory=list)
    rejected: list[Rejection] = Field(default_factory=list)


class UpdateMemoryResult(BaseModel):
    new_id: int
    previous_id: int


class ForgetResult(BaseModel):
    deleted: int


class RecordExchangeResult(BaseModel):
    conversation_id: int
    """Pass this back on later record_exchange calls in the same chat."""
    message_ids: list[int]
    redacted: list[str] = Field(default_factory=list)
    """Why parts of the exchange were replaced with [REDACTED], if any were."""


class SearchHit(BaseModel):
    type: Literal["memory", "chat"]
    id: int
    """Memory id, or chunk id for chat results."""
    content: str
    """Wrapped per SPEC.md Section 8."""
    kind: Kind | None = None
    subject: str | None = None
    project: str | None = None
    conversation_id: int | None = None
    title: str | None = None
    role: Role | None = None
    source: str
    """Client that wrote it (memories, captured chats) or provider it was imported from."""
    created_at: datetime | None
    score: float


class SearchResult(ReadResult):
    results: list[SearchHit] = Field(default_factory=list)


class ProfileItem(BaseModel):
    id: int
    content: str
    """Wrapped per SPEC.md Section 8."""
    subject: str | None = None
    project: str | None = None
    source: str
    updated_at: datetime


class Profile(ReadResult):
    preferences: list[ProfileItem] = Field(default_factory=list)
    facts: list[ProfileItem] = Field(default_factory=list)
    active_projects: list[ProfileItem] = Field(default_factory=list)
    recent_decisions: list[ProfileItem] = Field(default_factory=list)


class ConversationMessage(BaseModel):
    role: Role
    content: str
    """Wrapped per SPEC.md Section 8."""
    created_at: datetime | None


class ConversationView(BaseModel):
    id: int
    title: str | None
    source: str
    started_at: datetime | None
    updated_at: datetime | None
    message_count: int
    messages: list[ConversationMessage]
    """The most recent messages, oldest first."""


class RecentConversations(ReadResult):
    conversations: list[ConversationView] = Field(default_factory=list)


# --- CLI -----------------------------------------------------------------------------


class MemoryFilter(BaseModel):
    """Filters for `openmcp memories list`."""

    model_config = ConfigDict(extra="forbid")

    kinds: list[Kind] = Field(default_factory=list)
    project: Label | None = None
    scopes: list[Scope] = Field(default_factory=list)
    include_superseded: bool = False
    limit: int = Field(default=50, ge=1, le=10_000)


class ConversationFilter(BaseModel):
    """Filters for `openmcp conversations list`."""

    model_config = ConfigDict(extra="forbid")

    source: str | None = None
    scopes: list[Scope] = Field(default_factory=list)
    limit: int = Field(default=50, ge=1, le=10_000)


class AuditFilter(BaseModel):
    """Filters for `openmcp audit`."""

    model_config = ConfigDict(extra="forbid")

    client: str | None = None
    tool: str | None = None
    limit: int = Field(default=50, ge=1, le=10_000)
