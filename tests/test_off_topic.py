"""Keyword hits that meaning search judged off-topic rank below everything it found, by
meaning, without memory boosts. Found in a manual stress test: a memory about another
project matched one shared word (an assistant's name) and came back near the top of every
search, so the assistants kept bringing that project up."""

from datetime import UTC, datetime

from conftest import caller, insert
from openmemore import embed, relations
from openmemore.ingest import store_conversation
from openmemore.models import ParsedConversation, ParsedMessage
from openmemore.tools import search
from openmemore.tools.search import search_context

ME = caller()
QUERY = "Copilot pricing complaints"
ON_TOPIC = "Copilot pricing went up again."
SAME_WORD_CHAT = "Is Copilot worth it?"
OTHER_PROJECT = "Planning a vegetable garden; Copilot drafted the bed layout."


def chat(conn, external_id, text):
    store_conversation(conn, ParsedConversation(provider="vscode-copilot", external_id=external_id, messages=[
        ParsedMessage(external_id="1", role="user", content=text)]), "personal")


def store(conn, fake_embedder):
    fake_embedder.vectors = {QUERY: [1.0, 0.0, 0.0], ON_TOPIC: [0.95, 0.31, 0.0],
                             SAME_WORD_CHAT: [0.6, 0.8, 0.0], OTHER_PROJECT: [0.3, 0.0, 0.95]}
    garden = insert(conn, OTHER_PROJECT, kind="project")
    for source in ("cursor-vscode", "gemini-cli"):  # confirmed, so it carries a confidence boost
        relations.support(conn, garden, datetime.now(UTC), source=source)
    chat(conn, "a", ON_TOPIC)
    chat(conn, "b", SAME_WORD_CHAT)
    embed.embed_pending(conn, fake_embedder)
    return garden


def test_an_off_topic_memory_ranks_below_on_topic_chats(conn, fake_embedder):
    garden = store(conn, fake_embedder)
    hits = search_context(conn, ME, QUERY, explain=True).results
    assert [h.type for h in hits] == ["chat", "chat", "memory"] and hits[-1].id == garden
    assert hits[-1].explain["off_topic_weight"] == search.OFF_TOPIC_WEIGHT
    assert "confidence" not in hits[-1].explain  # boosts order matches, and this isn't one
    assert "off_topic_weight" not in hits[0].explain


def test_off_topic_memories_are_ranked_by_meaning_not_by_keyword_rank(conn, fake_embedder):
    store(conn, fake_embedder)
    terse = insert(conn, "Copilot.")  # the better keyword match, further in meaning
    fake_embedder.vectors["Copilot."] = [0.1, 0.0, 0.99]
    embed.embed_pending(conn, fake_embedder)
    hits = search_context(conn, ME, QUERY, explain=True).results
    assert [h.id for h in hits if h.type == "memory"][-1] == terse


def test_chat_keyword_hits_keep_their_rank(conn, fake_embedder):
    """Chats that share words but not meaning are often the evidence (demoting them lowered
    recall on both benchmarks, bench/RESULTS.md)."""
    store(conn, fake_embedder)
    hits = search_context(conn, ME, QUERY, explain=True).results
    assert "chat_keyword_rank" in hits[1].explain and "off_topic_weight" not in hits[1].explain


def test_a_keyword_match_alone_is_still_returned(conn, fake_embedder):
    """Exact names and codes the model doesn't know aren't hidden."""
    fake_embedder.vectors = {"ERR_4417": [1.0, 0.0], "Build failed with ERR_4417 on the runner.": [0.0, 1.0]}
    insert(conn, "Build failed with ERR_4417 on the runner.")
    embed.embed_pending(conn, fake_embedder)
    assert [h.type for h in search_context(conn, ME, "ERR_4417").results] == ["memory"]


def test_items_without_vectors_keep_their_keyword_rank(conn, fake_embedder):
    store(conn, fake_embedder)
    late = insert(conn, "Copilot pricing tiers for teams.")  # not embedded yet
    hits = search_context(conn, ME, QUERY, explain=True).results
    [hit] = [h for h in hits if h.type == "memory" and h.id == late]
    assert "off_topic_weight" not in hit.explain


def test_regression_real_model_keeps_another_project_out_of_a_banter_search(conn, real_embedder):
    """The stress test's shape with synthetic content: two assistants trading jokes about
    each other, and a confirmed memory about an unrelated project naming both."""
    plan = insert(conn, "Both Gemini and Copilot recommended the meal-planner app as the stronger "
                        "pick for the spring hackathon.", kind="decision")
    for source in ("cursor-vscode", "gemini-cli"):
        relations.support(conn, plan, datetime.now(UTC), source=source)
    for i, text in enumerate([
        "Roast Copilot for me.", "Gave a playful roast of Copilot: ghost text everywhere, confident guesses.",
        "Gemini says you talk too much.", "Fired back at Gemini's jokes about long answers, point by point.",
        "Copilot wants a rematch.", "Another round of jokes about Gemini's overlong answers.",
    ]):
        chat(conn, f"banter-{i}", text)
    embed.embed_pending(conn, real_embedder)
    hits = search_context(conn, ME, "Gemini Copilot jokes rematch").results
    assert hits and hits[0].type == "chat"
    assert all(h.type == "chat" for h in hits[:6])
