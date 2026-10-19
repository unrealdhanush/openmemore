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
"""A client name, as typed for `openmcp setup <client>`."""

Scope = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]

Kind = Literal["fact", "preference", "project", "decision", "note"]
KINDS: tuple[Kind, ...] = ("fact", "preference", "project", "decision", "note")

Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
"""Short free-text labels: subject and project."""

MemoryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]

MemoryId = Annotated[int, Field(ge=1)]

Role = Literal["user", "assistant", "system", "tool"]

VectorTable = Literal["memories", "chunks"]
"""Tables whose rows get embeddings (SPEC.md M2); each has a `<table>_vec` companion."""
VECTOR_TABLES: tuple[VectorTable, ...] = ("memories", "chunks")

JobStatus = Literal["queued", "running", "done", "failed"]

CaptureSource = Literal["claude-code", "vscode-copilot"]
"""Clients whose transcripts OpenMCP can read from disk (SPEC.md M1.5). Each name is
also the client name it connects with, and the `provider` of what it captures."""
CAPTURE_SOURCES: tuple[CaptureSource, ...] = ("claude-code", "vscode-copilot")

SUMMARY_MAX = 200
MAX_SAVE_ITEMS = 100
MAX_SEARCH_LIMIT = 25
PROFILE_GROUP_CAP = 15
ACTIVE_PROJECT_DAYS = 30
MAX_QUESTION_CHARS = 8000
MAX_ANSWER_CHARS = 4000
MAX_CODE_BLOCKS = 10
MAX_CODE_CHARS = 20_000
MAX_RECENT_CONVERSATIONS = 20
RECENT_MESSAGES_PER_CONVERSATION = 6

CHAT_ORDER_NOTE = (
    "Messages within a conversation are in the order they were written. Where a later message "
    "contradicts an earlier one (for example a decision made after several options were discussed), "
    "the later one is current."
)


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
    app: str | None = Field(default=None, max_length=SUMMARY_MAX)
    """The application that made the call (MCP clientInfo.name), when it says."""


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
    consolidated_through: int | None = None
    """The last message read for consolidation (SPEC.md M5)."""


class NewMessage(BaseModel):
    conversation_id: int
    role: Role
    content: str = Field(min_length=1)
    created_at: AwareDatetime | None = None
    content_hash: str
    external_id: str | None = None
    """The source's id for this message (captured transcripts), so re-reading a growing
    transcript updates the message instead of adding another copy."""
    recall: bool = False
    """Part of a turn that only looked up earlier conversations (SPEC.md 7.2)."""


class Message(NewMessage):
    """A row of `messages`."""

    id: int
    recall: bool = False


class ChunkMatch(BaseModel):
    """A chunk found by search, with the conversation it belongs to."""

    chunk_id: int
    message_id: int
    text: str
    role: Role
    recall: bool = False
    created_at: datetime | None
    conversation_id: int
    title: str | None
    provider: str
    rank: float
    """Lower is better: bm25 for keyword matches, cosine distance for vector matches."""


class CaptureFile(BaseModel):
    """A row of `capture_files`: a transcript file already read, and where to resume."""

    path: str
    source: CaptureSource
    mtime_ns: int
    size: int
    resume_offset: int = 0
    conversation_id: int | None = None
    ignored: bool = False
    """The user deleted the conversation, so changes to the file are no longer captured."""
    captured_at: AwareDatetime


class IngestJob(BaseModel):
    """A row of `ingest_jobs`: one export file being imported, resumable from `cursor`."""

    id: int | None = None
    provider: str
    """The importer that read it (openmcp.ingest.registry), e.g. 'chatgpt'."""
    file_path: str
    file_hash: str
    """sha256 of the file: importing the same file again is a no-op once it's done."""
    status: JobStatus
    total: int | None = None
    processed: int = 0
    cursor: str | None = None
    """Where to resume: how many conversations (newest first) are stored."""
    error: str | None = None
    started_at: AwareDatetime | None = None
    finished_at: AwareDatetime | None = None


# --- Conversations read from transcripts ---------------------------------------------


class ParsedMessage(BaseModel):
    external_id: str
    role: Role
    content: str = Field(min_length=1)
    created_at: AwareDatetime | None = None
    recall: bool = False
    """The turn called OpenMCP's recall tools (recent_conversations, search_context)."""


class ParsedConversation(BaseModel):
    """One conversation as read from a client's transcript, before it is stored."""

    provider: str
    external_id: str
    title: str | None = None
    """None keeps the stored title (e.g. when only the end of a transcript was re-read)."""
    created_at: AwareDatetime | None = None
    updated_at: AwareDatetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    messages: list[ParsedMessage] = Field(default_factory=list)
    scope: Scope | None = None
    """The scope it had where it came from (an OpenMCP export); None for provider exports."""


class ParsedMemory(BaseModel):
    """A memory read from an OpenMCP export, before it is stored."""

    kind: Kind
    content: MemoryText
    subject: Label | None = None
    project: Label | None = None
    scope: Scope | None = None
    source_client: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: AwareDatetime
    updated_at: AwareDatetime


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


class CodeBlock(BaseModel):
    """Code passed to `record_exchange`, stored with the answer so another assistant can continue from it."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=MAX_CODE_CHARS, description="The code, in full.")
    path: str | None = Field(
        default=None, pattern=r"^[^\n\r`]{1,300}$", description="The file it belongs in, if any, e.g. 'src/client.py'."
    )
    language: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9_+#.-]{1,30}$", description="Language name, e.g. 'python'."
    )


# --- MCP tool results ----------------------------------------------------------------


class ReadResult(BaseModel):
    """Base for every result that returns stored content. `notice` serializes first."""

    notice: str = NOTICE


class PingResult(BaseModel):
    """Result of the `ping` tool."""

    ok: bool
    version: str
    client: str
    """The name this client gets its scopes under: what its application reports, made config-friendly."""
    read_scopes: list[str] | None
    """None: every scope (ignore_scopes is on)."""
    write_scope: str
    schema_version: int
    semantic_search: str
    """'ready (<model>)', or why only keyword search is available."""
    connected_app: str | None = None
    """What the application on the other end calls itself (MCP clientInfo.name)."""


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
    conversation_id: int | None
    """Pass this back on later record_exchange calls in the same chat (None if not stored)."""
    message_ids: list[int] = Field(default_factory=list)
    redacted: list[str] = Field(default_factory=list)
    """Why parts of the exchange were replaced with [REDACTED], if any were."""
    note: str | None = None
    """Why nothing was stored, when nothing was."""


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
    recall: bool = False
    """A recap of earlier conversations, ranked below them: prefer the original it points to."""


class SearchResult(ReadResult):
    note: str | None = None
    """CHAT_ORDER_NOTE when the results include chat messages."""
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
    project_note: str | None = None
    """When `project` was asked for: which stored project it was taken to mean, or that
    none matched (then the profile is unfiltered)."""
    known_projects: list[str] | None = None
    """The projects that have memories, when the one asked for matched none of them."""


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
    note: str = CHAT_ORDER_NOTE
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


class ImportRequest(BaseModel):
    """Options for `openmcp import`."""

    model_config = ConfigDict(extra="forbid")

    scope: Scope | None = None


class CaptureRequest(BaseModel):
    """Options for `openmcp capture`."""

    model_config = ConfigDict(extra="forbid")

    sources: list[CaptureSource] = Field(default_factory=list)
    days: int | None = Field(default=None, ge=1, le=3650)


# --- Consolidation and review (SPEC.md M5) -------------------------------------------

MAX_BATCH_CONVERSATIONS = 20
BATCH_CHARS = 40_000
"""Message text in one consolidation batch; the rest waits for the next."""
CONVERSATION_BATCH_CHARS = 12_000
"""Message text from one conversation in one batch; a longer one continues in the next."""
MAX_PROPOSALS = 100
MAX_PROPOSAL_SOURCES = 20
MAX_REASON_CHARS = 500

ProposalAction = Literal["add", "merge"]
ProposalStatus = Literal["pending", "approved", "rejected"]


class ProposalInput(MemoryInput):
    """One item passed to `propose_memories`."""

    conversation_ids: list[Annotated[int, Field(ge=1)]] = Field(
        default_factory=list,
        max_length=MAX_PROPOSAL_SOURCES,
        description="The conversations this comes from (ids from consolidation_batch).",
    )
    reason: str | None = Field(
        default=None, max_length=MAX_REASON_CHARS, description="Why this is worth remembering, in a few words."
    )


class DoneConversation(BaseModel):
    """A conversation the model finished reading, as consolidation_batch returned it."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: int = Field(ge=1)
    through_message_id: int = Field(ge=1, description="Copy it from consolidation_batch.")


class BatchMessage(BaseModel):
    id: int
    role: Role
    content: str
    """Wrapped per SPEC.md Section 8."""
    created_at: datetime | None


class BatchConversation(BaseModel):
    id: int
    title: str | None
    source: str
    updated_at: datetime | None
    through_message_id: int
    """The last message in this batch: pass it back in propose_memories' `done`."""
    earlier_messages: int
    """Messages already read in an earlier consolidation (not repeated here)."""
    more_in_next_batch: bool
    messages: list[BatchMessage]


class ConsolidationBatch(ReadResult):
    conversations: list[BatchConversation] = Field(default_factory=list)
    remaining: int = 0
    """Conversations still due after this batch is marked done."""
    note: str


class ProposedSimilar(BaseModel):
    index: int
    proposal_id: int
    memory_id: int
    memory_content: str
    """Wrapped per SPEC.md Section 8."""


class ProposeResult(BaseModel):
    proposed: list[int] = Field(default_factory=list)
    skipped_duplicates: int = 0
    """Already a memory, or already proposed or rejected, in that scope."""
    similar_to_memories: list[ProposedSimilar] = Field(default_factory=list)
    rejected: list[Rejection] = Field(default_factory=list)
    conversations_done: int = 0
    note: str


class NewProposal(BaseModel):
    """A proposal ready to insert: validated, secret-filtered, hashed."""

    action: ProposalAction
    kind: Kind
    content: MemoryText
    subject: Label | None = None
    project: Label | None = None
    scope: Scope
    reason: str | None = None
    source_client: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    content_hash: str
    similar_memory_id: int | None = None
    created_at: AwareDatetime
    conversation_ids: list[int] = Field(default_factory=list)
    memory_ids: list[int] = Field(default_factory=list)
    """For a merge: the memories it replaces."""


class Proposal(NewProposal):
    """A row of `proposals`, with its sources."""

    id: int
    status: ProposalStatus
    decided_at: AwareDatetime | None = None
    memory_id: int | None = None
    """Once approved: the memory it became."""
