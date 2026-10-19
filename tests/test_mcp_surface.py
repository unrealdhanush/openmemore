"""The tools as a model sees them: over MCP, in-process and across real stdio processes."""

import json
import shutil
import sys
from pathlib import Path

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types import Implementation

from openmcp import config, db
from openmcp.capture import vscode_copilot
from openmcp.cli import initialize
from conftest import connect, example_clients
from openmcp.models import MAX_SAVE_ITEMS
from openmcp.safety.wrap import NOTICE
from openmcp.server import IMPORT_MY_MEMORY

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def server(home):
    initialize(home)
    example_clients(home)
    return lambda app="claude-desktop": connect(home, app)


def stdio(home, app, user_home=None):
    """`openmcp serve` in its own process, as MCP clients start it; `app` is the
    application name the client reports."""
    env = {"OPENMCP_HOME": str(home)}
    if user_home is not None:
        env["HOME"] = env["USERPROFILE"] = str(user_home)  # Path.home(): HOME on POSIX, USERPROFILE on Windows
    return Client(
        StdioServerParameters(command=sys.executable, args=["-m", "openmcp.cli", "serve"], env=env),
        client_info=Implementation(name=app, version="test"), read_timeout_seconds=30,
    )


def write_copilot_chat(user_home):
    """A Copilot Chat session as VS Code saves it (synthetic fixture)."""
    sessions = vscode_copilot.user_dir(user_home) / "workspaceStorage" / "abc123" / "chatSessions"
    sessions.mkdir(parents=True)
    shutil.copy(FIXTURES / "vscode_copilot" / "chat_session.json", sessions / "vs-0001.json")


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
async def test_m1_5_done_when_a_transcript_is_found_without_record_exchange(home, tmp_path):
    """SPEC M1.5 'done when': a Copilot chat saved by VS Code (nobody calls record_exchange)
    is answered about from Claude Desktop, in its own server process, by name and by topic."""
    user_home = tmp_path / "user"
    initialize(home)
    config.enable_capture(home / "config.toml", "vscode-copilot")

    async with stdio(home, "claude-desktop", user_home) as claude:
        assert (await claude.call_tool("recent_conversations", {})).structured_content["conversations"] == []
        write_copilot_chat(user_home)  # the user chats in VS Code while Claude Desktop is open

        recent = await claude.call_tool("recent_conversations", {"source": "copilot"})
        [conv] = recent.structured_content["conversations"]
        assert conv["source"] == "vscode-copilot"
        assert "Retry-After header" in conv["messages"][0]["content"]

        found = await claude.call_tool("search_context", {"query": "parse Retry-After"})
        assert found.structured_content["results"][0]["conversation_id"] == conv["id"]

    rows = db.connect(home / "openmcp.db").execute(
        "SELECT summary FROM audit_log WHERE tool = 'recent_conversations' ORDER BY id"
    ).fetchall()
    assert "captured 5" in rows[-1][0]


@pytest.mark.anyio
async def test_captured_clients_exchanges_arent_stored_twice(server, home):
    """Every client gets record_exchange (tools are set before a client says who it is);
    for an app whose transcripts are captured, calls are skipped."""
    config.enable_capture(home / "config.toml", "claude-code")
    async with server("claude-code") as client:
        assert "record_exchange" in {t.name for t in (await client.list_tools()).tools}
        skipped = (await client.call_tool("record_exchange", {"question": "q", "answer_summary": "a"})).structured_content
    assert skipped["conversation_id"] is None and "captured" in skipped["note"]
    async with server("claude-desktop") as client:
        stored = (await client.call_tool("record_exchange", {"question": "q", "answer_summary": "a"})).structured_content
    assert stored["conversation_id"] is not None


@pytest.mark.anyio
async def test_record_exchange_accepts_code_over_mcp(server):
    async with server() as client:
        result = await client.call_tool("record_exchange", {
            "question": "Write the retry helper",
            "answer_summary": "Wrote retry().",
            "code": [{"path": "retry.py", "language": "python", "content": "def retry(): ..."}],
        })
        bad = await client.call_tool("record_exchange", {
            "question": "q", "answer_summary": "a", "code": [{"content": "x", "language": "py\n```"}],
        })
        cid = result.structured_content["conversation_id"]
        full = await client.call_tool("recent_conversations", {"conversation_id": cid})
    assert "```python\ndef retry(): ...\n```" in full.structured_content["conversations"][0]["messages"][1]["content"]
    assert bad.is_error and "code.0.language" in bad.content[0].text


@pytest.mark.anyio
async def test_a_work_only_client_cant_see_personal_chats(home):
    """A client configured to read only 'work' can't see what another recorded in
    'personal'. The cross-assistant test needs a client that can."""
    initialize(home)
    example_clients(home)
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
    async with server("claude-desktop") as desktop:
        await desktop.call_tool("save_memories", {"items": [{"kind": "fact", "content": "Has a cat named Miso."}]})
    async with server("cursor") as cursor:  # reads only 'work'
        result = await cursor.call_tool("search_context", {"query": "cat"})
        assert result.structured_content["results"] == []
    async with server("gemini-cli") as gemini:  # unknown client: defaults read 'personal'
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
    async with server() as client:
        result = await client.call_tool("save_memories", args)
    assert result.is_error
    assert expected in result.content[0].text


@pytest.mark.anyio
async def test_search_limit_is_capped(server):
    async with server() as client:
        result = await client.call_tool("search_context", {"query": "x", "limit": 26})
    assert result.is_error and "limit" in result.content[0].text


@pytest.mark.anyio
async def test_anticipated_failures_reach_the_model(server):
    async with server() as client:
        result = await client.call_tool("forget", {"id": 42})
    assert result.is_error and "memory 42 not found" in result.content[0].text


@pytest.mark.anyio
async def test_every_call_is_audited_with_a_short_summary(server, home):
    long_query = "python " * 50
    async with server() as client:
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
    async with server() as client:
        prompts = await client.list_prompts()
        assert [p.name for p in prompts.prompts] == ["import_my_memory"]
        result = await client.get_prompt("import_my_memory")
    assert result.messages[0].content.text == IMPORT_MY_MEMORY


@pytest.mark.anyio
async def test_profile_resource_matches_get_profile(server):
    async with server() as client:
        await client.call_tool("save_memories", {"items": [{"kind": "preference", "content": "Prefers dark mode."}]})
        resource = await client.read_resource("openmcp://profile")
        tool = await client.call_tool("get_profile", {})
    payload = json.loads(resource.contents[0].text)
    assert payload["notice"] == NOTICE
    assert payload == tool.structured_content


@pytest.mark.anyio
async def test_tool_descriptions_tell_models_when_to_act(server):
    async with server() as client:
        tools = {t.name: t.description for t in (await client.list_tools()).tools}
    assert set(tools) == {
        "ping", "save_memories", "search_context", "get_profile", "update_memory", "forget",
        "record_exchange", "recent_conversations",
    }
    assert "Never save secrets" in tools["save_memories"]
    assert "start of a conversation" in tools["get_profile"]
    assert "explicitly asks" in tools["forget"]
    assert "after every answer" in tools["record_exchange"]
    assert "later one is current" in tools["recent_conversations"]
    assert "what did I ask Claude earlier?" in tools["recent_conversations"]


@pytest.mark.anyio
async def test_server_instructions_ask_models_to_record_and_recall(server):
    async with server() as client:
        instructions = client.instructions
    assert "record_exchange" in instructions and "recent_conversations" in instructions


@pytest.mark.anyio
async def test_ignore_scopes_shares_everything_with_every_client(server, home):
    """With ignore_scopes = true, a client configured to read only 'work' sees and can
    change 'personal' items; new items still get its write scope as their label."""
    async with server("claude-desktop") as desktop:  # writes 'personal'
        saved = (await desktop.call_tool("save_memories", {"items": [{"kind": "fact", "content": "Has a cat named Miso."}]})).structured_content
        await desktop.call_tool("record_exchange", {"question": "Cat food brands?", "answer_summary": "Try a few."})
    [memory_id] = saved["saved"]

    async with server("cursor") as cursor:  # reads only 'work'
        assert (await cursor.call_tool("search_context", {"query": "cat"})).structured_content["results"] == []

    path = home / "config.toml"
    cfg = config.load_config(path)
    cfg.ignore_scopes = True
    config.save_config(path, cfg)

    async with server("cursor") as cursor:
        ping = (await cursor.call_tool("ping", {})).structured_content
        hits = (await cursor.call_tool("search_context", {"query": "cat"})).structured_content["results"]
        updated = await cursor.call_tool("update_memory", {"id": memory_id, "content": "Has two cats, Miso and Tofu."})
        new = (await cursor.call_tool("save_memories", {"items": [{"kind": "fact", "content": "Uses Go at work."}]})).structured_content
    assert (ping["read_scopes"], ping["write_scope"]) == (None, "work")
    assert {h["type"] for h in hits} == {"memory", "chat"}
    assert not updated.is_error
    rows = dict(db.connect(home / "openmcp.db").execute("SELECT content, scope FROM memories WHERE superseded_by IS NULL"))
    assert rows == {"Has two cats, Miso and Tofu.": "personal", "Uses Go at work.": "work"}  # labels kept
