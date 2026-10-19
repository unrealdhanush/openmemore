"""Recall exchanges ("what did I ask Copilot?") are marked and ranked below the
conversations they recap. Reproduces the 2026-10-18 manual test, where recaps of a
Copilot chat about cat names outranked the chat itself."""

import json

import pytest
from mcp import Client

from conftest import FakeEmbedder, caller
from openmcp import db, embed
from openmcp.capture import capture, claude_code, vscode_copilot
from openmcp.cli import initialize
from openmcp.config import CaptureConfig, Config
from openmcp.server import RECALL_WINDOW_S, RecallTracker, build_server
from openmcp.tools.conversations import record_exchange
from openmcp.tools.search import search_context
from test_capture import add_claude_session, add_vscode_session, bump_mtime

DESKTOP = caller("claude-desktop", read=["personal"], write="personal")
COPILOT = caller("vscode-copilot", read=["personal"], write="personal")

ORIGINAL = ("Suggest three names for an orange, very lazy cat",
            "Here are three names for an orange, very lazy cat: 1. Marmalade 2. Pumpkin 3. Dozer")
RECAPS = [
    ("what was the pet thing I asked Copilot about?",
     "Found it in the VS Code Copilot chat from Oct 17: you asked for names for an orange, very lazy cat; "
     "it suggested Marmalade, Pumpkin, and Dozer."),
    ("What did I just ask Copilot?", "Looked up recent Copilot conversations: the latest was about cat names."),
]


@pytest.fixture
def user_home(tmp_path):
    return tmp_path / "user"


def store_case(conn):
    """The original Copilot chat, then Claude Desktop's recaps of it (marked recall)."""
    original = record_exchange(conn, COPILOT, *ORIGINAL).conversation_id
    for question, answer in RECAPS:
        record_exchange(conn, DESKTOP, question, answer, recall=True)
    return original


# --- spotting recalls from the order of calls ----------------------------------------


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_tracker():
    clock = Clock()
    t = RecallTracker(clock)
    assert t.next_exchange_is_recall(has_code=False) is False

    t.returned_context()
    assert t.next_exchange_is_recall(has_code=False) is True
    assert t.next_exchange_is_recall(has_code=False) is False  # one read marks one exchange

    t.returned_context()
    assert t.next_exchange_is_recall(has_code=True) is False  # the lookup led to real work

    t.returned_context()
    clock.now += RECALL_WINDOW_S + 1
    assert t.next_exchange_is_recall(has_code=False) is False


@pytest.mark.anyio
async def test_an_exchange_right_after_a_lookup_is_marked_recall(home):
    initialize(home)
    conn = db.connect(home / "openmcp.db")
    record_exchange(conn, COPILOT, *ORIGINAL)

    async with Client(build_server("claude-desktop", home)) as client:
        await client.call_tool("get_profile", {})
        first = await client.call_tool("record_exchange", {"question": "Plan my week", "answer_summary": "Drafted."})
        await client.call_tool("recent_conversations", {"source": "copilot"})
        recap = await client.call_tool("record_exchange", {
            "question": RECAPS[0][0], "answer_summary": RECAPS[0][1],
        })
        await client.call_tool("search_context", {"query": "zzzz nothing matches"})
        after_empty = await client.call_tool("record_exchange", {"question": "q", "answer_summary": "a"})

    def recall_flags(result):
        ids = result.structured_content["message_ids"]
        return {r[0] for r in conn.execute(f"SELECT recall FROM messages WHERE id IN ({ids[0]}, {ids[1]})")}

    assert recall_flags(first) == {0}  # get_profile doesn't count
    assert recall_flags(recap) == {1}
    assert recall_flags(after_empty) == {0}  # the lookup found nothing to recap
    summaries = [r[0] for r in conn.execute("SELECT summary FROM audit_log WHERE tool = 'record_exchange' ORDER BY id")]
    assert [s.endswith(", recall") for s in summaries] == [False, True, False]


# --- transcripts ---------------------------------------------------------------------


def test_transcript_turns_that_called_recall_tools_are_marked(tmp_path):
    claude, _ = claude_code.parse(add_claude_session(tmp_path))
    assert [m.recall for m in claude.messages] == [False, False, True, True]  # turn 2 called search_context

    copilot, _ = vscode_copilot.parse(add_vscode_session(tmp_path))
    assert [m.recall for m in copilot.messages] == [False, False, True, True, False]  # request 2 ran recent_conversations


def test_a_turn_that_calls_a_recall_tool_later_is_updated(conn, user_home):
    path = add_claude_session(user_home)
    first_turn = path.read_text().splitlines(keepends=True)[:7]
    path.write_text("".join(first_turn))
    cfg = Config(capture=CaptureConfig(sources=("claude-code",)))
    capture(conn, cfg, user_home=user_home, days=36500)
    assert {r[0] for r in conn.execute("SELECT recall FROM messages")} == {0}

    with path.open("a") as f:
        f.write(json.dumps({"type": "assistant", "uuid": "a-x", "sessionId": "sess-0001", "message": {
            "role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "mcp__openmcp__recent_conversations", "input": {}}],
        }}) + "\n")
    bump_mtime(path)
    capture(conn, cfg, user_home=user_home, days=36500)
    assert {r[0] for r in conn.execute("SELECT recall FROM messages")} == {1}


# --- ranking ---------------------------------------------------------------------------


def test_recaps_rank_below_the_conversation_they_recap(conn, monkeypatch):
    """Pinned similarities from the real case: the recap matches the query far better."""
    query = "pet question asked Copilot"
    fake = FakeEmbedder(vectors={
        query: [1, 0],
        ORIGINAL[0]: [0.58, 0.8146],  # cosine 0.58
        ORIGINAL[1]: [0.58, 0.8146],
        RECAPS[0][0]: [0.9, 0.4359],  # 0.9
        RECAPS[0][1]: [0.72, 0.694],
        RECAPS[1][0]: [0.81, 0.5864],
        RECAPS[1][1]: [0.7, 0.7141],
    })
    monkeypatch.setattr(embed, "get_embedder", lambda allow_download=False: fake)
    original = store_case(conn)

    hits = search_context(conn, DESKTOP, query).results
    assert hits[0].conversation_id == original and not hits[0].recall
    assert all(h.recall for h in hits if h.conversation_id != original)


def test_without_better_matches_recaps_are_still_returned(conn):
    record_exchange(conn, DESKTOP, *RECAPS[0], recall=True)
    hits = search_context(conn, DESKTOP, "Marmalade").results
    assert hits and all(h.recall for h in hits)


@pytest.mark.real_model
def test_m2_regression_real_model_puts_the_original_first(conn, real_embedder):
    """The exact query Claude Desktop sent in the manual test."""
    original = store_case(conn)
    hits = search_context(conn, DESKTOP, "pet question asked Copilot").results
    assert hits[0].conversation_id == original


@pytest.mark.real_model
def test_searching_by_topic_finds_the_original_even_unmarked(conn, real_embedder):
    """Fix 1 (search for the topic): works even where recaps weren't marked."""
    original = record_exchange(conn, COPILOT, *ORIGINAL).conversation_id
    for question, answer in RECAPS:
        record_exchange(conn, DESKTOP, question, answer)  # not marked
    for query in ("cat names", "pet names"):
        assert search_context(conn, DESKTOP, query).results[0].conversation_id == original, query
