"""Recall modes and labels. Current answers stay current; history
and as_of answer questions about the past, with every memory labelled."""

from datetime import UTC, datetime

import pytest

from conftest import caller, connect
from openmemore import db, repo
from openmemore.cli import initialize
from openmemore.ingest import store_conversation
from openmemore.models import MemoryInput, ParsedConversation, ParsedMessage
from openmemore.tools.memories import save_memories, update_memory
from openmemore.tools.search import search_context

ME = caller("claude-ai", read=["personal"], write="personal")


def save(conn, content, **when):
    return save_memories(conn, ME, [MemoryInput(kind="preference", content=content, **when)]).saved[0]


@pytest.fixture
def drinks(conn):
    """Tea from 2024, coffee since June 2026; Paris until 2020."""
    tea = save(conn, "Prefers tea in the morning.", valid_from="2024")
    coffee = update_memory(conn, ME, tea, "Prefers coffee in the morning.",
                           valid_from=datetime(2026, 6, 1, tzinfo=UTC)).new_id
    paris = save(conn, "Lived in Paris.", valid_from="2015", valid_to="2020")
    return tea, coffee, paris


def found(result):
    return [(h.id, h.status, h.replaced_by) for h in result.results if h.type == "memory"]


def test_current_mode_answers_with_what_is_true_now(conn, drinks):
    tea, coffee, paris = drinks
    assert found(search_context(conn, ME, "morning tea coffee")) == [(coffee, "current", None)]
    assert found(search_context(conn, ME, "Paris")) == []


def test_history_mode_includes_old_versions_labelled(conn, drinks):
    tea, coffee, paris = drinks
    result = search_context(conn, ME, "morning tea coffee", mode="history")
    assert sorted(found(result)) == sorted([(tea, "superseded", coffee), (coffee, "current", None)])
    [(_, status, _)] = found(search_context(conn, ME, "Paris", mode="history"))
    assert status == "ended"
    hit = next(h for h in result.results if h.id == tea)
    assert (hit.valid_from, hit.valid_to) == (datetime(2024, 1, 1, tzinfo=UTC), datetime(2026, 6, 1, tzinfo=UTC))


def test_as_of_gives_what_was_true_then(conn, drinks):
    tea, coffee, paris = drinks
    assert found(search_context(conn, ME, "morning tea coffee", as_of=datetime(2025, 3, 1, tzinfo=UTC))) == [
        (tea, "superseded", coffee)]
    assert [i for i, _, _ in found(search_context(conn, ME, "morning tea coffee",
                                                  as_of=datetime(2026, 9, 1, tzinfo=UTC)))] == [coffee]
    assert [i for i, _, _ in found(search_context(conn, ME, "Paris", as_of=datetime(2018, 1, 1, tzinfo=UTC)))] == [paris]


def test_history_is_found_by_meaning_too(conn, fake_embedder):
    fake_embedder.vectors = {"Prefers green tea.": [1.0, 0.0], "Prefers black coffee.": [0.0, 1.0],
                             "what hot drink did I like": [1.0, 0.0]}
    tea = save(conn, "Prefers green tea.")
    update_memory(conn, ME, tea, "Prefers black coffee.")
    assert conn.execute("SELECT COUNT(*) FROM memories_vec WHERE rowid = ?", (tea,)).fetchone()[0] == 1
    assert tea in [h.id for h in search_context(conn, ME, "what hot drink did I like", mode="history").results]
    assert tea not in [h.id for h in search_context(conn, ME, "what hot drink did I like").results]


def test_chat_results_are_labelled(conn):
    store_conversation(conn, ParsedConversation(provider="vscode-copilot", external_id="c", messages=[
        ParsedMessage(external_id="1", role="user", content="Plan a vegetable garden."),
        ParsedMessage(external_id="2", role="user", content="What garden did I ask Claude about?", recall=True),
    ]), "personal")
    assert sorted(h.status for h in search_context(conn, ME, "garden").results) == ["chat", "recap"]


def test_replaced_versions_are_queued_for_vectors_when_upgrading(tmp_path, monkeypatch):
    path = tmp_path / "t.db"
    every = db.migrations()
    monkeypatch.setattr(db, "migrations", lambda: [m for m in every if m[0] < 18])
    conn = db.open_db(path)
    t = "2026-01-01T00:00:00.000Z"
    for i in (1, 2):
        conn.execute("INSERT INTO memories (id, kind, content, scope, source_client, content_hash, created_at, "
                     "updated_at, embedded) VALUES (?, 'fact', ?, 'personal', 'x', ?, ?, ?, 1)", (i, f"m{i}", f"h{i}", t, t))
    conn.execute("UPDATE memories SET superseded_by = 2 WHERE id = 1")
    monkeypatch.undo()
    db.migrate(conn)
    assert dict(conn.execute("SELECT id, embedded FROM memories")) == {1: 0, 2: 1}
    assert [name for (name,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name = 'memories_vec_superseded'")] == []


@pytest.mark.anyio
async def test_the_tool_and_the_library(home):
    from openmemore import Memory

    initialize(home)
    memory = Memory("claude-ai", home=home)
    tea = memory.save("Prefers tea in the morning.", kind="preference", valid_from="2024")
    memory.update(tea, "Prefers coffee in the morning.", valid_from="2026-06")
    assert [(h.status, h.replaced_by is not None) for h in memory.search("morning tea coffee", chats=False,
                                                                         as_of="2025-03")] == [("superseded", True)]
    async with connect(home, "claude-ai") as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert {"mode", "as_of"} <= set(tools["search_context"].input_schema["properties"])
        result = (await client.call_tool("search_context", {"query": "morning tea coffee", "mode": "history"})).structured_content
    assert sorted(h["status"] for h in result["results"]) == ["current", "superseded"]
