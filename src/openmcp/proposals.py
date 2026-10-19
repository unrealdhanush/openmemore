"""The owner's side of consolidation (SPEC.md M5): approving and rejecting proposals,
and finding near-duplicate memories to merge. `openmcp review` calls these; clients
can only propose."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import UTC, datetime

from typing import Literal

from pydantic import BaseModel, ConfigDict

from openmcp import db, embed, relations, repo
from openmcp.models import Kind, Label, Memory, MemoryText, NewMemory, NewProposal, Proposal
from openmcp.tools.memories import NEAR_DUPLICATE_SIMILARITY, content_hash, embed_texts, secret_in

DUPLICATE_FINDER = "openmcp"
REVIEW = "review"
"""Source of links made by approving: the user's decision on the review page."""
"""source_client of merge proposals: found by comparing vectors, not by a model."""


class ProposalError(Exception):
    """Why a proposal can't be decided as asked; the message is for the user."""


class Approval(BaseModel):
    """Edits made while approving. Fields left out keep the proposal's values."""

    model_config = ConfigDict(extra="forbid")

    content: MemoryText | None = None
    kind: Kind | None = None
    subject: Label | None = None
    project: Label | None = None
    replace_similar: bool = False
    """Replace the similar memory (a new version of it) instead of adding another."""
    keep_replaced: bool = False
    """Keep the memory the proposal says it updates, instead of replacing it."""
    on_conflict: Literal["flag", "replace", "both"] = "flag"
    """For a proposal that contradicts a memory: replace it (this one is right), both are
    true, or flag the conflict to settle later (both kept out of current answers)."""


def _pending(conn: sqlite3.Connection, proposal_id: int) -> Proposal:
    proposal = repo.get_proposal(conn, proposal_id)
    if proposal is None:
        raise ProposalError(f"proposal {proposal_id} not found")
    if proposal.status != "pending":
        raise ProposalError(f"proposal {proposal_id} was already {proposal.status}")
    return proposal


def approve(conn: sqlite3.Connection, proposal_id: int, edits: Approval | None = None) -> int:
    """Turn a proposal into a memory and return its id. An add becomes a new memory (or a
    new version of the similar one, with replace_similar); a merge becomes one memory
    that supersedes every memory in the group."""
    edits = edits or Approval()
    proposal = _pending(conn, proposal_id)
    changed = edits.model_dump(exclude_unset=True, exclude={"replace_similar", "keep_replaced", "on_conflict"})
    fields = {**proposal.model_dump(include={"kind", "content", "subject", "project"}), **changed}
    if reason := secret_in(fields["content"], fields["subject"], fields["project"]):
        raise ProposalError(f"not saved: {reason}")
    [vector] = embed_texts([embed.memory_text(fields["subject"], fields["content"])])

    now = datetime.now(UTC)
    with db.transaction(conn):
        proposal = _pending(conn, proposal_id)
        replaced: list[Memory] = []
        if proposal.action == "merge":
            replaced = [m for mid in proposal.memory_ids if (m := repo.get_memory(conn, mid))]
            if len(replaced) != len(proposal.memory_ids) or not all(m.active for m in replaced):
                raise ProposalError("these memories changed since the merge was proposed; find duplicates again")
        else:
            if proposal.replaces_memory_id and not edits.keep_replaced:
                named = repo.get_memory(conn, proposal.replaces_memory_id)
                if named is None or not named.active:
                    raise ProposalError(
                        f"memory {proposal.replaces_memory_id} was changed since this was proposed; keep both, or reject this"
                    )
                replaced.append(named)
            if proposal.contradicts_memory_id and edits.on_conflict == "replace":
                disputed = repo.get_memory(conn, proposal.contradicts_memory_id)
                if disputed is None or not disputed.active:
                    raise ProposalError(f"memory {proposal.contradicts_memory_id} was changed since; approve "
                                        "this with both kept, or reject it")
                if disputed not in replaced:
                    replaced.append(disputed)
            if edits.replace_similar:
                similar = repo.get_memory(conn, proposal.similar_memory_id) if proposal.similar_memory_id else None
                if similar is None or not similar.active:
                    raise ProposalError("the similar memory was changed or deleted since; approve this as a new memory")
                if similar not in replaced:
                    replaced.append(similar)

        digest = content_hash(fields["content"])
        existing = repo.find_active_duplicate(conn, proposal.scope, digest)
        if existing and not replaced:  # saved meanwhile, by a client or an earlier approval
            repo.decide_proposal(conn, proposal.id, "approved", now, existing.id)
            return existing.id
        # True since the statement was made (or, for a merge, since the earliest version).
        valid_from = min((m.valid_from for m in replaced if m.valid_from), default=None) \
            if proposal.action == "merge" else proposal.valid_from
        memory_id = repo.insert_memory(conn, NewMemory(
            **fields,
            valid_from=valid_from or now,
            valid_to=proposal.valid_to,
            scope=proposal.scope,
            source_client=proposal.source_client,
            confidence=proposal.confidence,
            content_hash=digest,
            metadata={"proposal": proposal.id},
            created_at=now,
            updated_at=now,
        ))
        for old in replaced:
            relations.supersede(conn, memory_id, old.id, now, source=REVIEW,
                                valid_until=None if proposal.action == "merge" else valid_from or now)
        relations.derived_from(conn, memory_id, proposal.conversation_ids, now, source=proposal.source_client)
        if proposal.contradicts_memory_id and edits.on_conflict == "flag":
            disputed = repo.get_memory(conn, proposal.contradicts_memory_id)
            if disputed is not None and disputed.active:
                relations.contradict(conn, memory_id, disputed.id, now, source=REVIEW)
        if vector is not None:
            repo.store_vectors(conn, "memories", [(memory_id, vector)])
        repo.decide_proposal(conn, proposal.id, "approved", now, memory_id)
    return memory_id


def reject(conn: sqlite3.Connection, proposal_id: int) -> None:
    """Rejected proposals are kept, so the same statement (or merge) isn't proposed again."""
    with db.transaction(conn):
        _pending(conn, proposal_id)
        repo.decide_proposal(conn, proposal_id, "rejected", datetime.now(UTC))


def _groups(pairs: list[tuple[int, int]]) -> list[set[int]]:
    """Connected groups of ids from similar pairs (union-find)."""
    parent: dict[int, int] = {}

    def root(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[root(a)] = root(b)
    groups: dict[int, set[int]] = defaultdict(set)
    for x in parent:
        groups[root(x)].add(x)
    return list(groups.values())


def find_duplicates(conn: sqlite3.Connection) -> int:
    """Propose merging groups of current memories of the same kind and scope that are at
    least NEAR_DUPLICATE_SIMILARITY similar. A group already waiting for review, or
    rejected, isn't proposed again; nor is one that overlaps a pending merge. Returns how
    many merges were proposed. Memories not embedded yet aren't compared."""
    import numpy as np  # only here: numpy costs ~35ms to import (SPEC.md Section 4.3)

    by_group: dict[tuple[str, str], list[tuple[Memory, bytes]]] = defaultdict(list)
    for memory, vector in repo.active_memory_vectors(conn):
        by_group[(memory.scope, memory.kind)].append((memory, vector))
    pairs: list[tuple[int, int]] = []
    memories: dict[int, Memory] = {}
    for rows in by_group.values():
        if len(rows) < 2:
            continue
        ids = [m.id for m, _ in rows]
        memories.update((m.id, m) for m, _ in rows)
        matrix = np.stack([np.frombuffer(v, dtype=np.float32) for _, v in rows])
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        for start in range(0, len(rows), 1000):  # bounded memory for large stores
            block = matrix[start:start + 1000] @ matrix.T
            for i, j in zip(*np.nonzero(block >= NEAR_DUPLICATE_SIMILARITY)):
                if start + i < j:
                    pairs.append((ids[start + i], ids[j]))

    known = repo.merge_groups(conn)
    pending = {mid for p in repo.list_proposals(conn, status="pending") for mid in p.memory_ids}
    now = datetime.now(UTC)
    proposed = 0
    with db.transaction(conn):
        for group in sorted(_groups(pairs), key=min):
            if frozenset(group) in known or group & pending:
                continue
            members = sorted((memories[i] for i in group), key=lambda m: (m.updated_at, m.id))
            newest = members[-1]
            repo.insert_proposal(conn, NewProposal(
                action="merge",
                kind=newest.kind,
                content=newest.content,
                subject=newest.subject,
                project=newest.project,
                scope=newest.scope,
                reason=f"{len(members)} memories say nearly the same thing",
                source_client=DUPLICATE_FINDER,
                content_hash=content_hash(newest.content),
                created_at=now,
                memory_ids=[m.id for m in members],
            ))
            proposed += 1
    return proposed
