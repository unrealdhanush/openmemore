"""consolidation_batch, propose_memories (SPEC.md M5): the connected assistant reads
conversations not yet consolidated and proposes memories from them. Nothing becomes a
memory until the user approves it with `openmcp review`."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from mcp.server.mcpserver.exceptions import ToolError

from openmcp import db, embed, repo
from openmcp.models import (
    BATCH_CHARS,
    CONVERSATION_BATCH_CHARS,
    BatchConversation,
    BatchMessage,
    ConsolidationBatch,
    Conversation,
    DoneConversation,
    Kind,
    Memory,
    NewProposal,
    ProposalInput,
    ProposedSimilar,
    ProposeResult,
    Rejection,
)
from openmcp.safety.wrap import wrap
from openmcp.tools import Caller
from openmcp.tools.memories import NEAR_DUPLICATE_CANDIDATES, NEAR_DUPLICATE_SIMILARITY, content_hash, embed_texts, secret_in

REVIEW_HINT = "Tell the user to review the proposals with `openmcp review`: nothing is saved until they approve it."


def consolidation_batch(conn: sqlite3.Connection, caller: Caller, limit: int) -> ConsolidationBatch:
    """The next conversations with messages not yet read for consolidation, newest
    first, up to BATCH_CHARS of message text (CONVERSATION_BATCH_CHARS from any one)."""
    scopes = caller.scopes.read_scopes
    budget = BATCH_CHARS
    batch: list[BatchConversation] = []
    for c in repo.due_conversations(conn, scopes, limit):
        if budget <= 0:
            break
        cap = min(CONVERSATION_BATCH_CHARS, budget)
        messages = repo.unconsolidated_messages(conn, c.id, c.consolidated_through)
        taken, used = [], 0
        for m in messages:
            if taken and used + len(m.content) > cap:
                break
            taken.append(m)
            used += min(len(m.content), cap)
        budget -= used
        batch.append(BatchConversation(
            id=c.id,
            title=c.title,
            source=c.provider,
            updated_at=c.updated_at,
            through_message_id=taken[-1].id,
            earlier_messages=repo.count_consolidated_messages(conn, c.id, c.consolidated_through),
            more_in_next_batch=len(taken) < len(messages),
            messages=[
                BatchMessage(
                    id=m.id, role=m.role, created_at=m.created_at,
                    content=wrap(m.content, source=c.provider, id=f"message {m.id}", max_chars=cap),
                )
                for m in taken
            ],
        ))
    finished = sum(1 for c in batch if not c.more_in_next_batch)
    remaining = repo.count_due_conversations(conn, scopes) - finished
    if not batch:
        note = "Nothing left to consolidate. " + REVIEW_HINT
    else:
        note = (
            "Read these conversations and call propose_memories with the durable things they show about the "
            "user (preferences, facts, decisions and why, projects and their current state), each citing the "
            "conversation_ids it comes from. Skip one-off questions, small talk, and anything already in their "
            "profile; where something updates a memory in their profile, set `replaces` to that memory's id. In `done`, list every conversation above with its through_message_id, including ones that "
            "gave nothing, so they aren't read again. "
            + (f"{remaining} more conversations are due after this batch: call consolidation_batch again."
               if remaining else "This is the last batch. " + REVIEW_HINT)
        )
    return ConsolidationBatch(conversations=batch, remaining=remaining, note=note)


def _readable(conn: sqlite3.Connection, caller: Caller, conversation_id: int) -> Conversation:
    """The conversation, if this client can read it. Others are reported as not found,
    so their existence doesn't leak."""
    conversation = repo.get_conversation(conn, conversation_id)
    if conversation is None or not caller.can_read(conversation.scope):
        raise ToolError(f"conversation {conversation_id} not found")
    return conversation


def _replaced(conn: sqlite3.Connection, caller: Caller, memory_id: int) -> Memory:
    """The current memory a proposal says it updates, if this client can read it."""
    memory = repo.get_memory(conn, memory_id)
    if memory is None or not caller.can_read(memory.scope):
        raise ToolError(f"memory {memory_id} not found")
    if not memory.active:
        raise ToolError(f"memory {memory_id} was already replaced by memory {memory.superseded_by}; name that one")
    return memory


def _similar_memory(conn: sqlite3.Connection, scope: str, kind: Kind, vector: bytes) -> Memory | None:
    """The most similar current memory of the same kind in `scope`, if it's a near-duplicate."""
    nearest = repo.knn_memories(conn, vector, k=NEAR_DUPLICATE_CANDIDATES, scopes=[scope], kinds=[kind])
    if nearest and embed.similarity(nearest[0][1]) >= NEAR_DUPLICATE_SIMILARITY:
        return nearest[0][0]
    return None


def propose_memories(
    conn: sqlite3.Connection, caller: Caller, items: list[ProposalInput], done: list[DoneConversation]
) -> ProposeResult:
    """Queue memories for the user's review. A proposal takes the scope of the
    conversations it came from and of the memory it replaces (which this client must be
    able to read, and which must agree), or this client's write scope when it cites none."""
    result = ProposeResult(note=REVIEW_HINT)
    accepted: list[tuple[int, ProposalInput, str]] = []
    for index, item in enumerate(items):
        sources = [_readable(conn, caller, cid) for cid in item.conversation_ids]
        scopes = {c.scope for c in sources}
        for named in (item.replaces, item.contradicts):
            if named is not None:
                scopes.add(_replaced(conn, caller, named).scope)
        if len(scopes) > 1:
            raise ToolError(
                f"item {index} cites conversations or a memory in different scopes ({', '.join(sorted(scopes))}); "
                "propose it separately for each"
            )
        if reason := secret_in(item.content, item.subject, item.project, item.reason):
            result.rejected.append(Rejection(index=index, reason=reason))
            continue
        # True since the model says, or since the conversation it came from (not since approval).
        said = max((c.updated_at for c in sources if c.updated_at), default=None)
        accepted.append((index, item.model_copy(update={"valid_from": item.valid_from or said}),
                         scopes.pop() if scopes else caller.scopes.write_scope))
    for d in done:
        _readable(conn, caller, d.conversation_id)
    vectors = embed_texts([embed.memory_text(item.subject, item.content) for _, item, _ in accepted])

    now = datetime.now(UTC)
    with db.transaction(conn):
        for (index, item, scope), vector in zip(accepted, vectors):
            digest = content_hash(item.content)
            # Earlier items in this call count too: they're pending by now.
            if repo.find_active_duplicate(conn, scope, digest) or repo.proposed_or_rejected(conn, scope, digest):
                result.skipped_duplicates += 1
                continue
            similar = _similar_memory(conn, scope, item.kind, vector) if vector is not None else None
            if similar and similar.id == item.replaces:
                similar = None  # already named
            proposal_id = repo.insert_proposal(conn, NewProposal(
                action="add",
                **item.model_dump(include={"kind", "content", "subject", "project", "confidence", "reason"}),
                scope=scope,
                source_client=caller.client,
                content_hash=digest,
                similar_memory_id=similar.id if similar else None,
                replaces_memory_id=item.replaces,
                contradicts_memory_id=item.contradicts if item.contradicts != item.replaces else None,
                valid_from=item.valid_from,
                valid_to=item.valid_to,
                created_at=now,
                conversation_ids=item.conversation_ids,
            ))
            result.proposed.append(proposal_id)
            if similar:
                result.similar_to_memories.append(ProposedSimilar(
                    index=index, proposal_id=proposal_id, memory_id=similar.id,
                    memory_content=wrap(similar.content, source=similar.source_client, id=similar.id),
                ))
        for d in done:
            result.conversations_done += repo.mark_consolidated(conn, d.conversation_id, d.through_message_id)
    return result
