"""SPEC.md M3: importing ChatGPT and Claude data exports (synthetic fixtures only)."""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import FakeEmbedder, caller, plain
from export_factory import chatgpt_conversation, chatgpt_export, write_zip
from openmcp import db, repo
from openmcp.cli import app, initialize
from openmcp.ingest import chatgpt, claude
from openmcp.ingest.detect import ExportError, load
from openmcp.ingest.jobs import BATCH, import_export
from openmcp.tools.conversations import recent_conversations
from openmcp.tools.search import search_context

FIXTURES = Path(__file__).parent / "fixtures"
READER = caller("gemini-cli", read=["personal"], write="personal")


def raw(provider):
    return json.loads((FIXTURES / f"{provider}_export" / "conversations.json").read_text())


def texts(parsed):
    return [(m.role, m.content) for m in parsed.messages]


# --- ChatGPT -------------------------------------------------------------------------


def test_chatgpt_keeps_the_branch_the_user_ended_on_and_only_visible_text():
    parsed = chatgpt.conversation(raw("chatgpt")[0])
    assert texts(parsed) == [
        ("user", "How do I rate-limit a Flask API?"),
        ("assistant", "Use Flask-Limiter backed by Redis."),  # not the regenerated answer, tool call, or thoughts
        ("user", "What does this diagram show?"),  # the image part is dropped
        ("assistant", "It shows a token bucket refilling at 100 requests per minute."),
    ]
    assert (parsed.provider, parsed.external_id, parsed.title) == ("chatgpt", "chatgpt-flask", "Flask rate limiting")
    assert parsed.metadata == {"capture": "import", "model": "gpt-4o"}
    assert parsed.created_at == datetime.fromtimestamp(1_790_000_000, tz=UTC)
    assert [m.external_id for m in parsed.messages] == ["u1", "a2", "u2", "a3"]


def test_chatgpt_without_current_node_uses_the_latest_leaf_and_titles_from_the_question():
    parsed = chatgpt.conversation(raw("chatgpt")[1])
    assert texts(parsed)[0] == ("user", "Name three orange cat breeds.")
    assert parsed.title == "Name three orange cat breeds."


def test_chatgpt_conversations_with_nothing_to_keep_are_skipped():
    assert chatgpt.conversation(raw("chatgpt")[2]) is None
    assert chatgpt.conversation({"no": "id"}) is None
    assert chatgpt.conversation("not a conversation") is None


def test_chatgpt_tolerates_odd_shapes():
    conv = chatgpt_conversation(1)
    conv["mapping"]["c1-user-0"]["message"]["author"] = "user"  # a string, not an object
    conv["mapping"]["c1-assistant-0"]["parent"] = "c1-assistant-0"  # a cycle
    assert chatgpt.conversation(conv) is not None


# --- Claude --------------------------------------------------------------------------


def test_claude_keeps_text_blocks_and_maps_senders():
    parsed = claude.conversation(raw("claude")[0])
    assert texts(parsed) == [
        ("user", "Plan three days in Lisbon in March."),
        ("assistant", "Day 1: Alfama and the castle.\n\nDay 2: Belém. Day 3: Sintra."),  # tool blocks skipped
    ]
    assert (parsed.provider, parsed.external_id, parsed.title) == ("claude", "claude-trip", "Lisbon trip")


def test_claude_plain_text_messages_and_untitled_conversations():
    parsed = claude.conversation(raw("claude")[1])
    assert texts(parsed)[1] == ("assistant", "Clint Yeastwood.")
    assert parsed.title == "What's a good name for a sourdough starter?"
    assert claude.conversation(raw("claude")[2]) is None


# --- detection -------------------------------------------------------------------------


def test_exports_are_recognized_from_their_contents(tmp_path):
    zipped = write_zip(tmp_path / "export.zip", raw("chatgpt"), folder="some-export-2026/")
    assert load(zipped).provider == "chatgpt"
    assert load(FIXTURES / "claude_export" / "conversations.json").provider == "claude"
    assert load(FIXTURES / "claude_export").provider == "claude"  # an unpacked folder


@pytest.mark.parametrize("contents, message", [
    ({"conversations.json": "[{\"something\": 1}]"}, "doesn't look like"),
    ({"conversations.json": "{not json"}, "isn't valid JSON"),
    ({"conversations.json": "{\"a\": 1}"}, "should hold a list"),
    ({"other.json": "[]"}, "no conversations.json"),
])
def test_unreadable_exports_say_why(tmp_path, contents, message):
    import zipfile
    path = tmp_path / "x.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, data in contents.items():
            z.writestr(name, data)
    with pytest.raises(ExportError, match=message):
        load(path)


# --- importing -----------------------------------------------------------------------


def test_imported_conversations_are_searchable_and_listable(conn, tmp_path):
    report = import_export(conn, write_zip(tmp_path / "e.zip", raw("chatgpt")), scope="personal")

    assert (report.provider, report.total, report.conversations, report.skipped) == ("chatgpt", 4, 3, 1)
    listed = recent_conversations(conn, READER, source="chatgpt").conversations
    assert {c.title for c in listed} == {"Flask rate limiting", "Name three orange cat breeds.", "Deploy key"}
    assert search_context(conn, READER, "token bucket").results[0].source == "chatgpt"


def test_secrets_in_exports_are_redacted(conn, tmp_path):
    import_export(conn, write_zip(tmp_path / "e.zip", raw("chatgpt")), scope="personal")
    stored = " ".join(r[0] for r in conn.execute("SELECT content FROM messages"))
    assert "ghp_" not in stored and "[REDACTED]" in stored


def test_importing_the_same_file_again_does_nothing(conn, tmp_path):
    path = write_zip(tmp_path / "e.zip", raw("claude"))
    import_export(conn, path, scope="personal")
    again = import_export(conn, path, scope="personal")
    assert again.already_done and conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 4


def test_a_newer_export_adds_only_whats_new(conn, tmp_path):
    import_export(conn, write_zip(tmp_path / "old.zip", chatgpt_export(3)), scope="personal")
    newer = chatgpt_export(4)
    newer[0] = chatgpt_conversation(0, turns=3)  # the oldest conversation continued
    report = import_export(conn, write_zip(tmp_path / "new.zip", newer), scope="personal")
    assert (report.conversations, report.messages) == (2, 6)  # conversation 3 is new (4), conversation 0 grew (2)
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3 * 4 + 6


def test_conversations_the_user_deleted_stay_deleted(conn, tmp_path):
    import_export(conn, write_zip(tmp_path / "old.zip", chatgpt_export(2)), scope="personal")
    [conv] = [c for c in repo.list_conversations(conn, scopes=None) if c.external_id == "conv-0"]
    repo.delete_conversation(conn, conv.id)
    report = import_export(conn, write_zip(tmp_path / "new.zip", chatgpt_export(3)), scope="personal")
    assert report.skipped == 1
    assert "conv-0" not in {c.external_id for c in repo.list_conversations(conn, scopes=None)}


def test_imports_go_to_the_given_scope(conn, tmp_path):
    import_export(conn, write_zip(tmp_path / "e.zip", raw("claude")), scope="work")
    assert recent_conversations(conn, READER).conversations == []  # READER can't read 'work'
    assert recent_conversations(conn, caller(read=["work"], write="work")).conversations


def test_batches_are_embedded_as_they_land(conn, tmp_path):
    import_export(conn, write_zip(tmp_path / "e.zip", raw("claude")), scope="personal", embedder=FakeEmbedder())
    assert repo.count_pending_embeddings(conn) == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 4


def test_a_malformed_conversation_is_reported_not_fatal(conn, tmp_path, monkeypatch):
    original = chatgpt.conversation

    def flaky(raw_conv):
        if raw_conv.get("id") == "conv-1":
            raise TypeError("unexpected shape")
        return original(raw_conv)

    monkeypatch.setattr(chatgpt, "conversation", flaky)
    report = import_export(conn, write_zip(tmp_path / "e.zip", chatgpt_export(3)), scope="personal")
    assert report.conversations == 2 and len(report.errors) == 1 and "TypeError" in report.errors[0]


# --- SPEC M3 done-when -----------------------------------------------------------------


@pytest.fixture
def big_export(tmp_path):
    return write_zip(tmp_path / "chatgpt-export.zip", chatgpt_export(500))


def test_m3_done_when_the_newest_50_are_searchable_before_the_job_finishes(tmp_path, big_export):
    """Checked from a second connection (as a server would) after the first batch."""
    path = tmp_path / "store.db"
    importer, server = db.open_db(path), db.connect(path)
    seen = []

    def after_batch(done, total):
        if not seen:
            newest = search_context(server, READER, "topic499").results
            oldest = search_context(server, READER, "topic0").results
            seen.append((done, total, bool(newest), bool(oldest)))

    import_export(importer, big_export, scope="personal", progress=after_batch)
    assert seen == [(BATCH, 500, True, False)]
    assert search_context(server, READER, "topic0").results  # all there once it's done


def test_m3_done_when_killing_and_rerunning_resumes_without_duplicates(conn, big_export):
    def die_after_three_batches(done, total):
        if done == 3 * BATCH:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        import_export(conn, big_export, scope="personal", progress=die_after_three_batches)
    [job] = repo.list_jobs(conn)
    assert (job.status, job.processed, job.error) == ("failed", 150, "interrupted")
    assert conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 150

    report = import_export(conn, big_export, scope="personal")
    assert report.resumed_from == 150
    [job] = repo.list_jobs(conn)
    assert (job.status, job.processed, job.total) == ("done", 500, 500)
    assert conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 500
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 500 * 4  # 2 turns, no duplicates
    assert conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT conversation_id, external_id FROM messages)").fetchone()[0] == 2000


# --- CLI -------------------------------------------------------------------------------


runner = CliRunner()


def cli(*args):
    result = runner.invoke(app, list(args), env={"COLUMNS": "200"})
    return result.exit_code, " ".join(plain(result.output).split())


def test_import_and_status_commands(home, tmp_path):
    initialize(home)
    path = write_zip(tmp_path / "chatgpt-export.zip", raw("chatgpt"))
    code, out = cli("import", str(path))
    assert code == 0, out
    assert "Imported the ChatGPT export into scope 'personal': 3 conversations with new messages" in out
    assert "Skipped 1" in out

    code, out = cli("import", str(path))
    assert code == 0 and "Already imported" in out

    code, out = cli("status")
    assert "chatgpt-export.zip" in out and "done" in out and "4/4" in out
    assert "Conversations: 3 (3 from chatgpt)" in out


def test_import_command_refuses_what_it_cant_read(home, tmp_path):
    initialize(home)
    bad = tmp_path / "notes.json"
    bad.write_text("[1, 2, 3]")
    code, out = cli("import", str(bad))
    assert code == 2 and "doesn't look like a ChatGPT or Claude export" in out
    assert cli("import", str(tmp_path / "missing.zip"))[0] == 2


def test_status_before_any_import(home):
    initialize(home)
    code, out = cli("status")
    assert code == 0 and "No imports yet" in out


def test_everything_is_stored_before_embedding_starts_and_embedding_goes_newest_first(conn, big_export):
    """Interrupting the embedding phase leaves the whole export stored (and keyword
    searchable), with the newest conversations embedded first; `openmcp embed` finishes."""
    def stop_after_first_group(done, total):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        import_export(conn, big_export, scope="personal", embedder=FakeEmbedder(), embed_progress=stop_after_first_group)
    [job] = repo.list_jobs(conn)
    assert job.status == "done" and conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 500
    embedded_titles = {r[0] for r in conn.execute(
        "SELECT DISTINCT conv.title FROM chunks c JOIN messages m ON m.id = c.message_id "
        "JOIN conversations conv ON conv.id = m.conversation_id WHERE c.embedded = 1"
    )}
    assert embedded_titles == {f"Conversation {i}" for i in range(450, 500)}  # the newest 50

    from openmcp import embed
    assert embed.embed_pending(conn, FakeEmbedder()).remaining == 0
