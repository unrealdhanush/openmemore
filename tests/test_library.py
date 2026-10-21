"""SPEC.md M7.5: OpenMemore as a library. Programs share memory with assistants in-process;
the decorators record and recall without ever breaking the decorated function."""

import asyncio
import logging
import subprocess
import sys

import pytest

from conftest import connect
from openmemore import Memory, OpenMemoreError, config, db
from openmemore.cli import initialize
from openmemore.safety.wrap import NOTICE, unwrap


@pytest.fixture
def memory(home):
    initialize(home)
    return Memory("Trip Planner", home=home)


def audit_apps(home):
    return {tuple(r) for r in db.connect(home / "openmemore.db").execute("SELECT DISTINCT client, app FROM audit_log")}


# --- direct calls -----------------------------------------------------------------------------


def test_save_search_profile_update_forget(memory, home):
    memory_id = memory.save("Prefers aisle seats.", kind="preference")
    assert memory.save("prefers aisle seats") is None  # the same statement: not stored twice
    [hit] = memory.search("aisle seats", chats=False)
    assert (hit.id, hit.type, hit.kind, hit.text, hit.source) == (memory_id, "memory", "preference",
                                                                  "Prefers aisle seats.", "trip-planner")
    assert hit.wrapped.startswith("<stored_context ") and "Prefers aisle seats." in hit.wrapped
    assert "Prefers aisle seats." in memory.profile().preferences[0].content
    new_id = memory.update(memory_id, "Prefers window seats.")
    assert [h.text for h in memory.search("seats", chats=False)] == ["Prefers window seats."]
    assert memory.forget(new_id) == 2
    assert audit_apps(home) == {("trip-planner", "openmemore library")}


def test_secrets_and_missing_ids_raise(memory):
    with pytest.raises(OpenMemoreError, match="not saved"):
        memory.save("My token is ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789", kind="fact")
    with pytest.raises(OpenMemoreError, match="not found"):
        memory.forget(999)


def test_record_and_recent(memory):
    first = memory.record("Which airline for Lisbon?", "TAP flies direct.")
    second = memory.record("And the hotel?", "Near Alfama.")
    assert first == second  # the same conversation key: one conversation
    [conversation] = memory.recent().conversations
    assert conversation.message_count == 4 and conversation.source == "trip-planner"


def test_context_is_a_prompt_ready_block(memory):
    assert memory.context("anything at all") == ""
    memory.save("Prefers aisle seats.", kind="preference")
    block = memory.context("aisle seats")
    assert block.startswith(NOTICE) and "<stored_context" in block and "Prefers aisle seats." in block


def test_scopes_apply_by_client_name(home):
    initialize(home)
    path = home / "config.toml"
    cfg = config.load_config(path)
    cfg.clients["work-agent"] = config.ClientOverride(read_scopes=("work",), write_scope="work")
    config.save_config(path, cfg)
    personal = Memory("home-agent", home=home)
    work = Memory("Work Agent", home=home)
    secret_id = personal.save("Personal thing.", kind="fact")
    work.save("Uses Go at work.", kind="fact")
    assert [h.text for h in work.search("thing work Go", chats=False)] == ["Uses Go at work."]
    with pytest.raises(OpenMemoreError, match="not found"):
        work.forget(secret_id)


# --- decorators -----------------------------------------------------------------------------


def test_remember_records_each_call_into_one_conversation(memory):
    @memory.remember(question="request")
    def plan(budget: int, request: str) -> dict:
        return {"plan": f"Trip for {request}", "budget": budget}

    plan(900, "Lisbon in May")
    plan(900, request="Porto in June")
    [conversation] = memory.recent(limit=5).conversations
    contents = [m.content for m in conversation.messages]
    assert "Lisbon in May" in contents[0] and '"plan": "Trip for Lisbon in May"' in contents[1]
    assert conversation.message_count == 4


def test_remember_defaults_to_the_first_string_argument_and_separates_functions(memory):
    @memory.remember
    def one(n: int, text: str) -> str:
        return text.upper()

    @memory.remember
    def two(text: str) -> str:
        return text[::-1]

    one(1, "hello")
    two("abc")
    assert sorted(c.title for c in memory.recent().conversations) == ["abc", "hello"]


def test_a_conversation_block_groups_calls_across_functions(memory):
    @memory.remember
    def ask(text: str) -> str:
        return "ok"

    @memory.remember
    def follow_up(text: str) -> str:
        return "sure"

    with memory.conversation("trip-42"):
        ask("Book Lisbon.")
        follow_up("Add a hotel.")
    [conversation] = memory.recent().conversations
    assert conversation.message_count == 4


def test_recall_passes_related_memories_in(memory):
    memory.save("Prefers aisle seats.", kind="preference")
    seen = {}

    @memory.recall(query="request")
    def plan(request: str, context: str = "") -> str:
        seen["context"] = context
        return "ok"

    plan("Which seat should I book?")
    assert "Prefers aisle seats." in seen["context"]


def test_recall_checks_the_function_when_decorating(memory):
    with pytest.raises(TypeError, match="needs an argument 'context'"):
        @memory.recall
        def no_context(request: str) -> str:
            return request

    with pytest.raises(TypeError, match="has no argument 'question'"):
        @memory.recall(query="question")
        def wrong_name(request: str, context: str = "") -> str:
            return request


def test_async_functions(memory):
    memory.save("Prefers aisle seats.", kind="preference")

    @memory.recall(query="request")
    @memory.remember(question="request")
    async def plan(request: str, context: str = "") -> str:
        await asyncio.sleep(0)
        return f"aisle: {'aisle' in context}"

    assert asyncio.run(plan("Which seat?")) == "aisle: True"
    [conversation] = memory.recent().conversations
    assert [unwrap(m.content) for m in conversation.messages] == ["Which seat?", "aisle: True"]  # wrapped, as over MCP


def test_the_store_failing_never_breaks_the_function(memory, monkeypatch, caplog):
    def broken(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(Memory, "record", broken)
    monkeypatch.setattr(Memory, "context", broken)

    @memory.recall
    @memory.remember
    def plan(request: str, context: str = "") -> str:
        return f"context={context!r}"

    with caplog.at_level(logging.WARNING, logger="openmemore"):
        assert plan("Lisbon") == "context=''"
    assert "couldn't record" in caplog.text and "couldn't recall" in caplog.text


def test_exceptions_from_the_function_pass_through_and_aren_t_recorded(memory):
    @memory.remember
    def fails(text: str) -> str:
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        fails("hello")
    assert memory.recent().conversations == []


# --- with the user's assistants --------------------------------------------------------------


@pytest.mark.anyio
async def test_assistants_and_programs_share_one_memory(home):
    initialize(home)
    memory = Memory("trip-planner", home=home)

    @memory.recall(query="request")
    @memory.remember(question="request")
    def plan(request: str, context: str = "") -> str:
        return "Aisle seat booked." if "aisle" in context.lower() else "Any seat booked."

    async with connect(home, "claude-ai") as claude:
        await claude.call_tool("save_memories", {"items": [{"kind": "preference", "content": "Prefers aisle seats."}]})
    assert plan("Book my flight seat to Lisbon.") == "Aisle seat booked."  # what an assistant saved
    async with connect(home, "cursor-vscode") as cursor:
        found = (await cursor.call_tool("search_context", {"query": "flight seat Lisbon"})).structured_content
    assert any(h["source"] == "trip-planner" and "Aisle seat booked." in h["content"] for h in found["results"])


def test_importing_openmemore_stays_light():
    """`import openmemore` must not load the store or the library: the server's start-up
    budget (SPEC.md Section 4.3) doesn't pay for them."""
    out = subprocess.run([sys.executable, "-c", "import sys, openmemore; print(sorted(m for m in sys.modules "
                          "if m.startswith('openmemore.')))"], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"
