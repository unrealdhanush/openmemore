"""SPEC.md M7.7 part 6: explanations and token budgets."""

import pytest

from conftest import caller, connect, insert
from openmemore.cli import initialize
from openmemore.models import PROFILE_GROUP_CAP
from openmemore.ingest import store_conversation
from openmemore.models import ParsedConversation, ParsedMessage
from openmemore.tools.profile import get_profile
from openmemore.tools.search import estimate_tokens, search_context

ME = caller("claude-ai", read=["personal"], write="personal")


def test_explain_says_what_made_each_score(conn):
    insert(conn, "Prefers green tea.", kind="preference")
    store_conversation(conn, ParsedConversation(provider="vscode-copilot", external_id="c", messages=[
        ParsedMessage(external_id="1", role="user", content="Which green tea is best?"),
        ParsedMessage(external_id="2", role="user", content="What tea did I ask Claude about?", recall=True),
    ]), "personal")
    result = search_context(conn, ME, "green tea", explain=True)
    by_type = {}
    for hit in result.results:
        by_type.setdefault(hit.status, hit.explain)
    assert {"memory_keyword_rank", "fusion", "activation", "confidence"} <= set(by_type["current"])
    assert {"chat_keyword_rank", "fusion", "activation"} <= set(by_type["chat"]) and "confidence" not in by_type["chat"]
    assert by_type["recap"]["recap_weight"] < 1
    assert result.trace["mode"] == "current" and result.trace["words"] == ["green", "tea"]
    assert result.trace["candidates"]["memory_keyword"] == 1 and result.trace["meaning_search"] is False
    assert search_context(conn, ME, "green tea").results[0].explain is None


def test_explain_with_meaning_search(conn, fake_embedder):
    from openmemore.models import MemoryInput
    from openmemore.tools.memories import save_memories

    save_memories(conn, ME, [MemoryInput(kind="preference", content="Prefers green tea.")])
    [hit] = search_context(conn, ME, "green tea", explain=True, include_chats=False).results
    assert hit.explain["memory_keyword_rank"] == 1 and hit.explain["memory_meaning_rank"] == 1


def test_a_budget_keeps_the_best_results_that_fit(conn):
    big = insert(conn, "Prefers tea, " + "very " * 200 + "strongly.", kind="preference")
    small = insert(conn, "Prefers tea with honey.", kind="preference")
    everything = search_context(conn, ME, "prefers tea")
    assert {h.id for h in everything.results} == {big, small}
    budget = estimate_tokens(next(h.content for h in everything.results if h.id == small)) + 5
    result = search_context(conn, ME, "prefers tea", budget_tokens=budget)
    assert [h.id for h in result.results] == [small] and result.dropped == 1  # the big one didn't fit
    assert result.tokens_used <= budget


def test_a_profile_budget_doesn_t_let_one_group_crowd_out_the_rest(conn):
    for i in range(10):
        insert(conn, f"Preference number {i}: " + "detail " * 20, kind="preference")
    project = insert(conn, "Building a garden planner app.", kind="project")
    full = get_profile(conn, ME)
    assert len(full.preferences) == min(10, PROFILE_GROUP_CAP)
    trimmed = get_profile(conn, ME, budget_tokens=150)
    assert [i.id for i in trimmed.active_projects] == [project]  # not crowded out by preferences
    assert trimmed.tokens_used <= 150 and trimmed.dropped > 0 and len(trimmed.preferences) < len(full.preferences)


@pytest.mark.anyio
async def test_through_mcp_and_the_library(home):
    from openmemore import Memory

    initialize(home)
    memory = Memory("claude-ai", home=home)
    for i in range(8):
        memory.save(f"Prefers tea variety {i}, " + "notes " * 30, kind="preference")
    async with connect(home, "claude-ai") as client:
        result = (await client.call_tool("search_context", {"query": "prefers tea variety", "explain": True,
                                                            "budget_tokens": 200})).structured_content
        profile = (await client.call_tool("get_profile", {"budget_tokens": 100})).structured_content
    assert result["tokens_used"] <= 200 and result["dropped"] > 0 and result["trace"]["candidates"]
    assert all(h["explain"] for h in result["results"])
    assert profile["tokens_used"] <= 100
    block = memory.context("tea variety", profile=True, budget_tokens=300, limit=10)
    assert estimate_tokens(block) <= 300 + estimate_tokens(block.split("\n")[0]) + 5  # the notice line on top
