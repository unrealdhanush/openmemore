"""SPEC.md M7.7 part 5: support raises confidence. A memory stated again by an independent
source (another client, another conversation) is confirmed, not just skipped."""

from datetime import UTC, datetime, timedelta

import pytest

from conftest import caller
from openmcp import relations, repo
from openmcp.ingest import store_conversation
from openmcp.models import MemoryInput, ParsedConversation, ParsedMessage, ProposalInput
from openmcp.tools.consolidation import propose_memories
from openmcp.tools.memories import save_memories
from openmcp.tools.profile import get_profile
from openmcp.tools.search import search_context

CLAUDE = caller("claude-ai", read=["personal"], write="personal")
CURSOR = caller("cursor-vscode", read=["personal"], write="personal")
NOW = datetime(2026, 10, 19, 12, tzinfo=UTC)


def save(conn, who, content="Prefers metric units."):
    return save_memories(conn, who, [MemoryInput(kind="preference", content=content)])


def confidence(conn, memory_id):
    return repo.get_memory(conn, memory_id).confidence


def chat(conn, external_id):
    cid, _ = store_conversation(conn, ParsedConversation(provider="vscode-copilot", external_id=external_id,
                                messages=[ParsedMessage(external_id="1", role="user", content="Metric, please.")]),
                                "personal")
    return cid


def test_the_formula_adds_evidence_in_log_odds():
    assert relations.confirmed(None) == pytest.approx(0.75)  # 0.6 -> odds 1.5 -> 3.0
    assert relations.confirmed(0.75) == pytest.approx(6 / 7, abs=1e-4)
    c = None
    for _ in range(20):
        c = relations.confirmed(c)
    assert c == relations.CONFIDENCE_CAP  # never certainty
    assert relations.confirmed(0.6, reliability=0.5) < relations.confirmed(0.6)


def test_another_client_saying_it_again_confirms_it(conn):
    [memory_id] = save(conn, CLAUDE).saved
    assert confidence(conn, memory_id) is None  # stated once (0.6 when shown)
    result = save(conn, CURSOR, "prefers metric units")
    assert (result.saved, result.skipped_duplicates, result.confirmed) == ([], 1, [memory_id])
    assert confidence(conn, memory_id) == pytest.approx(0.75)
    [link] = relations.links(conn, memory_id, "supports")
    assert (link.source, link.to_memory, link.to_conversation) == ("cursor-vscode", memory_id, None)


def test_repeats_don_t_count_twice(conn):
    [memory_id] = save(conn, CLAUDE).saved
    assert save(conn, CLAUDE).confirmed == []  # the same client restating what it just saved
    save(conn, CURSOR)
    assert save(conn, CURSOR).confirmed == []  # the same source again within a day
    assert confidence(conn, memory_id) == pytest.approx(0.75)
    later = NOW + timedelta(days=2)
    assert relations.support(conn, memory_id, later, source="cursor-vscode")  # a day later it counts again


def test_conversations_confirm_once_each(conn):
    first, second = chat(conn, "a"), chat(conn, "b")
    [pid] = propose_memories(conn, CLAUDE, [ProposalInput(kind="preference", content="Prefers metric units.",
                                                          conversation_ids=[first])], []).proposed
    from openmcp.proposals import approve

    memory_id = approve(conn, pid)  # derived from conversation a
    again = propose_memories(conn, CLAUDE, [ProposalInput(kind="preference", content="Prefers metric units.",
                                                          conversation_ids=[first, second])], [])
    assert (again.proposed, again.confirmed) == ([], [memory_id])  # a was its source; only b confirms
    assert confidence(conn, memory_id) == pytest.approx(0.75)
    assert propose_memories(conn, CLAUDE, [ProposalInput(kind="preference", content="Prefers metric units.",
                                                         conversation_ids=[second])], []).confirmed == []


def test_confidence_shows_and_breaks_ties(conn):
    weak = save(conn, CLAUDE, "Prefers dark mode in the editor.").saved[0]
    strong = save(conn, CLAUDE, "Prefers dark mode in the terminal.").saved[0]
    for who in ("cursor-vscode", "gemini-cli", "vscode-copilot"):
        relations.support(conn, strong, NOW, source=who)
    hits = {h.id: h for h in search_context(conn, CLAUDE, "prefers dark mode").results}
    assert hits[weak].confidence == relations.DEFAULT_CONFIDENCE and hits[strong].confidence > 0.9
    assert [h.id for h in search_context(conn, CLAUDE, "prefers dark mode").results][0] == strong
    assert {i.id: i.confidence for i in get_profile(conn, CLAUDE).preferences}[strong] > 0.9
