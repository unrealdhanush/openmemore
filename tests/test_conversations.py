from datetime import UTC, datetime, timedelta

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from conftest import caller
from openmcp import repo
from openmcp.models import RECENT_MESSAGES_PER_CONVERSATION, CodeBlock
from openmcp.tools.conversations import _add_message, chunk_text, recent_conversations, record_exchange
from openmcp.tools.search import search_context

DESKTOP = caller("claude-desktop", read=["personal", "work"], write="personal")
GEMINI = caller("gemini-cli", read=["personal"], write="personal")
CURSOR = caller("cursor", read=["work"], write="work")


def texts(view):
    return [m.content.split("\n")[1] for m in view.messages]


# --- record_exchange -----------------------------------------------------------------


def test_first_exchange_starts_a_conversation_owned_by_the_client(conn):
    result = record_exchange(conn, DESKTOP, "How do I rate-limit a Flask API?", "Suggested Flask-Limiter with Redis.")

    conv = repo.get_conversation(conn, result.conversation_id)
    assert (conv.provider, conv.scope, conv.title) == ("claude-desktop", "personal", "How do I rate-limit a Flask API?")
    assert conv.metadata == {"capture": "record_exchange"}
    messages = repo.conversation_messages(conn, conv.id)
    assert [(m.role, m.content) for m in messages] == [
        ("user", "How do I rate-limit a Flask API?"),
        ("assistant", "Suggested Flask-Limiter with Redis."),
    ]
    assert result.message_ids == [m.id for m in messages] and result.redacted == []


def test_later_exchanges_append_to_the_same_conversation(conn):
    first = record_exchange(conn, DESKTOP, "q1", "a1", title="Rate limiting")
    second = record_exchange(conn, DESKTOP, "q2", "a2", conversation_id=first.conversation_id, title="ignored")

    assert second.conversation_id == first.conversation_id
    assert [m.content for m in repo.conversation_messages(conn, first.conversation_id)] == ["q1", "a1", "q2", "a2"]
    assert repo.get_conversation(conn, first.conversation_id).title == "Rate limiting"


def test_same_message_at_the_same_instant_is_stored_once(conn):
    cid = record_exchange(conn, DESKTOP, "q", "a").conversation_id
    when = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)

    first = _add_message(conn, cid, "user", "retried", when)
    again = _add_message(conn, cid, "user", "retried", when)
    later = _add_message(conn, cid, "user", "retried", when + timedelta(seconds=1))

    assert first == again != later
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE text = 'retried'").fetchone()[0] == 2


def test_cannot_append_to_another_clients_conversation(conn):
    theirs = record_exchange(conn, DESKTOP, "q", "a").conversation_id
    with pytest.raises(ToolError, match="omit conversation_id"):
        record_exchange(conn, GEMINI, "q2", "a2", conversation_id=theirs)
    with pytest.raises(ToolError, match="not found"):
        record_exchange(conn, GEMINI, "q2", "a2", conversation_id=999)
    assert len(repo.conversation_messages(conn, theirs)) == 2


def test_secrets_are_redacted_not_rejected(conn):
    result = record_exchange(
        conn, DESKTOP,
        "Why does ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789 get a 401?",
        "The token in your question has expired; rotate it.",
    )
    assert result.redacted == ["contains a GitHub token"]
    stored = [m.content for m in repo.conversation_messages(conn, result.conversation_id)]
    assert stored[0] == "Why does [REDACTED] get a 401?"
    assert "ghp_" not in repo.get_conversation(conn, result.conversation_id).title


def test_long_messages_are_chunked_for_search(conn):
    answer = "\n\n".join(f"Paragraph {i} about caching strategy. " * 10 for i in range(8))
    result = record_exchange(conn, DESKTOP, "Caching?", answer)
    chunks = conn.execute("SELECT text FROM chunks WHERE message_id = ?", (result.message_ids[1],)).fetchall()
    assert len(chunks) > 1 and all(len(c[0]) <= 1000 for c in chunks)


@pytest.mark.parametrize("text", [
    "short",
    "word " * 600,
    "x" * 2500,  # no boundaries at all
    "para one.\n\n" + "y" * 900 + "\n\npara three",
])
def test_chunk_text_keeps_all_content_within_size(text):
    chunks = chunk_text(text, size=1000)
    assert all(0 < len(c) <= 1000 for c in chunks)
    assert "".join(chunks).replace(" ", "").replace("\n", "") == text.replace(" ", "").replace("\n", "")


# --- recent_conversations ------------------------------------------------------------


def test_recent_lists_newest_first_with_latest_messages(conn):
    old = record_exchange(conn, DESKTOP, "older question", "older answer").conversation_id
    new = record_exchange(conn, GEMINI, "newer question", "newer answer").conversation_id

    result = recent_conversations(conn, GEMINI)

    assert [c.id for c in result.conversations] == [new, old]
    view = result.conversations[1]
    assert (view.source, view.message_count) == ("claude-desktop", 2)
    assert texts(view) == ["older question", "older answer"]
    assert view.messages[0].content.startswith('<stored_context source="claude-desktop"')


def test_appending_moves_a_conversation_to_the_top(conn):
    a = record_exchange(conn, DESKTOP, "a", "a").conversation_id
    b = record_exchange(conn, DESKTOP, "b", "b").conversation_id
    conn.execute("UPDATE conversations SET updated_at = '2026-01-01T00:00:00.000Z' WHERE id = ?", (a,))
    conn.execute("UPDATE conversations SET updated_at = '2026-01-02T00:00:00.000Z' WHERE id = ?", (b,))
    assert [c.id for c in recent_conversations(conn, DESKTOP).conversations] == [b, a]
    record_exchange(conn, DESKTOP, "a2", "a2", conversation_id=a)
    assert [c.id for c in recent_conversations(conn, DESKTOP).conversations] == [a, b]


def test_source_filter_is_a_partial_case_insensitive_match(conn):
    record_exchange(conn, DESKTOP, "from claude", "x")
    record_exchange(conn, GEMINI, "from gemini", "x")
    sources = [c.source for c in recent_conversations(conn, GEMINI, source="Claude").conversations]
    assert sources == ["claude-desktop"]


def test_listing_shows_only_the_latest_turns_but_a_single_conversation_shows_more(conn):
    cid = record_exchange(conn, DESKTOP, "q0", "a0").conversation_id
    for i in range(1, 6):
        record_exchange(conn, DESKTOP, f"q{i}", f"a{i}", conversation_id=cid)

    listed = recent_conversations(conn, DESKTOP).conversations[0]
    assert listed.message_count == 12
    assert len(listed.messages) == RECENT_MESSAGES_PER_CONVERSATION
    assert texts(listed)[-1] == "a5"

    single = recent_conversations(conn, DESKTOP, conversation_id=cid).conversations[0]
    assert texts(single)[0] == "q0" and len(single.messages) == 12


def test_recent_respects_read_scopes(conn):
    cid = record_exchange(conn, DESKTOP, "personal chat", "x").conversation_id
    assert recent_conversations(conn, CURSOR).conversations == []
    with pytest.raises(ToolError, match="not found"):
        recent_conversations(conn, CURSOR, conversation_id=cid)


# --- search over captured chats -------------------------------------------------------


def test_search_finds_chats_and_points_back_to_the_conversation(conn):
    cid = record_exchange(conn, DESKTOP, "How do I rate-limit a Flask API?", "Use Flask-Limiter.").conversation_id
    hits = search_context(conn, GEMINI, "flask rate limiting").results
    assert hits and all(h.type == "chat" for h in hits)
    assert {h.conversation_id for h in hits} == {cid}
    assert hits[0].source == "claude-desktop" and hits[0].title.startswith("How do I rate-limit")


def test_memories_and_chats_are_interleaved_by_rank(conn):
    from conftest import insert
    mem = insert(conn, "Prefers Flask over Django.", kind="preference")
    record_exchange(conn, DESKTOP, "Flask question", "Flask answer")
    hits = search_context(conn, DESKTOP, "flask").results
    assert hits[0].type == "memory" and hits[0].id == mem
    assert hits[1].type == "chat"
    assert hits[0].score == hits[1].score  # both rank 1 in their own list


def test_chat_results_can_be_excluded(conn):
    record_exchange(conn, DESKTOP, "Flask question", "Flask answer")
    assert search_context(conn, DESKTOP, "flask", include_chats=False).results == []
    assert search_context(conn, DESKTOP, "flask", kinds=["fact"]).results == []
    assert search_context(conn, DESKTOP, "flask", project="X").results == []


def test_chat_search_respects_read_scopes(conn):
    record_exchange(conn, DESKTOP, "Flask question", "Flask answer")
    assert search_context(conn, CURSOR, "flask").results == []


def test_deleting_a_conversation_removes_its_messages_chunks_and_index_entries(conn):
    cid = record_exchange(conn, DESKTOP, "Flask question", "Flask answer").conversation_id
    assert repo.delete_conversation(conn, cid)
    for table in ("messages", "chunks"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'flask'").fetchone()[0] == 0


# --- code blocks and ordering --------------------------------------------------------


def test_code_is_stored_with_the_answer_and_read_back_in_full(conn):
    code = "def parse_retry_after(value):\n" + "    pass\n" * 400  # well past the listing cap
    result = record_exchange(
        conn, DESKTOP, "Respect Retry-After?", "Added parse_retry_after().",
        code=[CodeBlock(path="strava_client.py", language="python", content=code)],
    )
    answer = repo.conversation_messages(conn, result.conversation_id)[1].content
    assert answer.startswith("Added parse_retry_after().\n\n`strava_client.py`:\n```python\ndef parse_retry_after")
    assert answer.endswith("    pass\n```")

    listed = recent_conversations(conn, GEMINI).conversations[0].messages[1].content
    full = recent_conversations(conn, GEMINI, conversation_id=result.conversation_id).conversations[0].messages[1].content
    assert "[truncated]" in listed
    assert "[truncated]" not in full and code.rstrip() in full


def test_code_containing_fences_gets_a_longer_fence(conn):
    result = record_exchange(conn, DESKTOP, "q", "a", code=[CodeBlock(content="```\nnested\n```")])
    answer = repo.conversation_messages(conn, result.conversation_id)[1].content
    assert answer == "a\n\n````\n```\nnested\n```\n````"


def test_secrets_in_code_and_titles_are_redacted(conn):
    result = record_exchange(
        conn, DESKTOP, "q", "a", title="Fix ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789",
        code=[CodeBlock(content='TOKEN = "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"')],
    )
    stored = repo.conversation_messages(conn, result.conversation_id)[1].content
    assert "ghp_" not in stored and "ghp_" not in repo.get_conversation(conn, result.conversation_id).title
    assert result.redacted == ["contains a GitHub token"]


def test_later_turns_of_a_conversation_rank_first(conn):
    cid = record_exchange(conn, DESKTOP, "Retries for the poller?", "Use 3-4 attempts with backoff.").conversation_id
    record_exchange(conn, DESKTOP, "Decided: backoff capped at 5 attempts.", "Noted: 5 attempts.", conversation_id=cid)

    hits = search_context(conn, GEMINI, "attempts backoff").results
    times = [h.created_at for h in hits]
    assert times == sorted(times, reverse=True)
    assert "5 attempts" in hits[0].content


def test_chat_results_carry_the_ordering_note(conn):
    from conftest import insert
    insert(conn, "Prefers Flask.", kind="preference")
    assert search_context(conn, DESKTOP, "flask").note is None
    record_exchange(conn, DESKTOP, "Flask question", "Flask answer")
    assert "later one is current" in search_context(conn, DESKTOP, "flask").note
    assert "later one is current" in recent_conversations(conn, DESKTOP).note
