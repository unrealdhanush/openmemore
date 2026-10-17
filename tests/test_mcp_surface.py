"""The tools as a model sees them: over MCP, in-process and across real stdio processes."""

import json
import sys

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from openmcp import db
from openmcp.cli import initialize
from openmcp.models import MAX_SAVE_ITEMS
from openmcp.safety.wrap import NOTICE
from openmcp.server import IMPORT_MY_MEMORY, build_server


@pytest.fixture
def server(home):
    initialize(home)
    return lambda client="claude-desktop": build_server(client, home)


def stdio(home, client):
    return Client(
        StdioServerParameters(
            command=sys.executable,
            args=["-m", "openmcp.cli", "serve", "--client", client],
            env={"OPENMCP_HOME": str(home)},
        ),
        read_timeout_seconds=30,
    )


@pytest.mark.anyio
async def test_m1_done_when_another_assistant_answers_about_the_previous_question(home):
    """SPEC M1 'done when', with the models' side scripted: Claude Desktop answers a question
    and records the exchange; a different assistant (its own server process, as in real use)
    finds it both without keywords ('what did I ask Claude?') and by topic."""
    async with stdio(home, "claude-desktop") as claude:
        recorded = await claude.call_tool("record_exchange", {
            "question": "How should I rate-limit my Flask API?",
            "answer_summary": "Recommended Flask-Limiter backed by Redis, 100 requests/minute per API key, "
                              "returning 429 with Retry-After. Next step: add it to the /search endpoint.",
        })
        assert not recorded.is_error

    async with stdio(home, "gemini-cli") as gemini:
        recent = await gemini.call_tool("recent_conversations", {"source": "claude"})
        [conv] = recent.structured_content["conversations"]
        assert conv["source"] == "claude-desktop"
        assert "rate-limit my Flask API" in conv["messages"][0]["content"]
        assert "Flask-Limiter" in conv["messages"][1]["content"]

        found = await gemini.call_tool("search_context", {"query": "flask rate limiting"})
        hits = found.structured_content["results"]
        assert hits and hits[0]["type"] == "chat" and hits[0]["conversation_id"] == conv["id"]


@pytest.mark.anyio
async def test_default_scopes_keep_personal_chats_from_a_work_only_client(home):
    """With the default config, Cursor reads only 'work', so it can't see what Claude
    Desktop recorded in 'personal'. The cross-assistant test needs a client that can."""
    async with stdio(home, "claude-desktop") as claude:
        await claude.call_tool("record_exchange", {"question": "Flask rate limits?", "answer_summary": "Use Flask-Limiter."})
    async with stdio(home, "cursor") as cursor:
        recent = await cursor.call_tool("recent_conversations", {})
        assert recent.structured_content["conversations"] == []


@pytest.mark.anyio
async def test_preferences_survive_into_a_new_session(home):
    """Save in one session; a fresh server process (a new chat) retrieves it from a natural
    question and from get_profile."""
    async with stdio(home, "claude-desktop") as first:
        saved = await first.call_tool("save_memories", {"items": [
            {"kind": "preference", "content": "Prefers Python for scripting."},
            {"kind": "preference", "content": "Prefers terse answers."},
        ]})
        assert len(saved.structured_content["saved"]) == 2

    async with stdio(home, "claude-desktop") as second:
        found = await second.call_tool("search_context", {"query": "what do you know about my preferences?"})
        text = found.content[0].text
        assert "Prefers Python for scripting." in text and "Prefers terse answers." in text
        assert text.lstrip("{\n ").startswith('"notice"')

        profile = await second.call_tool("get_profile", {})
        contents = [p["content"] for p in profile.structured_content["preferences"]]
        assert any("Prefers Python" in c for c in contents)


@pytest.mark.anyio
async def test_another_client_sees_memories_in_shared_scopes_only(server):
    async with Client(server("claude-desktop")) as desktop:
        await desktop.call_tool("save_memories", {"items": [{"kind": "fact", "content": "Has a cat named Miso."}]})
    async with Client(server("cursor")) as cursor:  # reads only 'work'
        result = await cursor.call_tool("search_context", {"query": "cat"})
        assert result.structured_content["results"] == []
    async with Client(server("gemini-cli")) as gemini:  # unknown client: defaults read 'personal'
        result = await gemini.call_tool("search_context", {"query": "cat"})
        assert len(result.structured_content["results"]) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("args, expected", [
    ({"items": [{"kind": "skill", "content": "x"}]}, "items.0.kind"),
    ({"items": [{"kind": "fact", "content": ""}]}, "items.0.content"),
    ({"items": [{"kind": "fact", "content": "x", "tags": ["a"]}]}, "items.0.tags"),
    ({"items": []}, "items"),
    ({"items": [{"kind": "fact", "content": f"m{i}"} for i in range(MAX_SAVE_ITEMS + 1)]}, "items"),
])
async def test_invalid_arguments_come_back_as_clear_errors(server, args, expected):
    async with Client(server()) as client:
        result = await client.call_tool("save_memories", args)
    assert result.is_error
    assert expected in result.content[0].text


@pytest.mark.anyio
async def test_search_limit_is_capped(server):
    async with Client(server()) as client:
        result = await client.call_tool("search_context", {"query": "x", "limit": 26})
    assert result.is_error and "limit" in result.content[0].text


@pytest.mark.anyio
async def test_anticipated_failures_reach_the_model(server):
    async with Client(server()) as client:
        result = await client.call_tool("forget", {"id": 42})
    assert result.is_error and "memory 42 not found" in result.content[0].text


@pytest.mark.anyio
async def test_every_call_is_audited_with_a_short_summary(server, home):
    long_query = "python " * 50
    async with Client(server()) as client:
        await client.call_tool("save_memories", {"items": [{"kind": "fact", "content": "Uses python."}]})
        await client.call_tool("search_context", {"query": long_query})
        await client.call_tool("get_profile", {})
        await client.call_tool("forget", {"id": 999})
        await client.read_resource("openmcp://profile")

    conn = db.connect(home / "openmcp.db")
    rows = {r["tool"]: r for r in conn.execute("SELECT * FROM audit_log")}
    assert set(rows) == {"save_memories", "search_context", "get_profile", "forget", "openmcp://profile"}
    assert rows["save_memories"]["result_count"] == 1
    assert len(rows["search_context"]["summary"]) < 80
    assert rows["forget"]["summary"] == "error: ToolError"
    assert all(r["client"] == "claude-desktop" for r in rows.values())


@pytest.mark.anyio
async def test_import_my_memory_prompt(server):
    async with Client(server()) as client:
        prompts = await client.list_prompts()
        assert [p.name for p in prompts.prompts] == ["import_my_memory"]
        result = await client.get_prompt("import_my_memory")
    assert result.messages[0].content.text == IMPORT_MY_MEMORY


@pytest.mark.anyio
async def test_profile_resource_matches_get_profile(server):
    async with Client(server()) as client:
        await client.call_tool("save_memories", {"items": [{"kind": "preference", "content": "Prefers dark mode."}]})
        resource = await client.read_resource("openmcp://profile")
        tool = await client.call_tool("get_profile", {})
    payload = json.loads(resource.contents[0].text)
    assert payload["notice"] == NOTICE
    assert payload == tool.structured_content


@pytest.mark.anyio
async def test_tool_descriptions_tell_models_when_to_act(server):
    async with Client(server()) as client:
        tools = {t.name: t.description for t in (await client.list_tools()).tools}
    assert set(tools) == {
        "ping", "save_memories", "search_context", "get_profile", "update_memory", "forget",
        "record_exchange", "recent_conversations",
    }
    assert "Never save secrets" in tools["save_memories"]
    assert "start of a conversation" in tools["get_profile"]
    assert "explicitly asks" in tools["forget"]
    assert "after each substantive question" in tools["record_exchange"]
    assert "what did I ask Claude earlier?" in tools["recent_conversations"]


@pytest.mark.anyio
async def test_server_instructions_ask_models_to_record_and_recall(server):
    async with Client(server()) as client:
        instructions = client.instructions
    assert "record_exchange" in instructions and "recent_conversations" in instructions
