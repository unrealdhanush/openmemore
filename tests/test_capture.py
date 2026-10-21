"""Transcript capture, against synthetic transcripts modeled on real ones."""

import json
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from conftest import caller
from openmemore import repo
from openmemore.capture import capture, claude_code, vscode_copilot
from openmemore.config import CaptureConfig, Config
from openmemore.tools.conversations import recent_conversations
from openmemore.tools.search import search_context

FIXTURES = Path(__file__).parent / "fixtures"
CFG = Config(capture=CaptureConfig(sources=("claude-code", "vscode-copilot")))
GEMINI = caller("gemini-cli", read=["personal"], write="personal")


@pytest.fixture
def user_home(tmp_path, monkeypatch) -> Path:
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    return tmp_path / "user"


def add_claude_session(user_home: Path, name: str = "session.jsonl") -> Path:
    path = user_home / ".claude" / "projects" / "-home-test-poller" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES / "claude_code" / "session.jsonl", path)
    return path


def add_vscode_session(user_home: Path) -> Path:
    storage = vscode_copilot.user_dir(user_home) / "workspaceStorage" / "abc123"
    (storage / "chatSessions").mkdir(parents=True)
    shutil.copy(FIXTURES / "vscode_copilot" / "workspace.json", storage / "workspace.json")
    path = storage / "chatSessions" / "vs-0001.json"
    shutil.copy(FIXTURES / "vscode_copilot" / "chat_session.json", path)
    return path


def append_lines(path: Path, *lines: dict) -> None:
    with path.open("a") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    bump_mtime(path)


def bump_mtime(path: Path) -> None:
    """Make the change visible to the mtime check even on coarse-grained filesystems."""
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


def contents(conn, conversation_id):
    return [(m.role, m.content) for m in repo.conversation_messages(conn, conversation_id)]


# --- Claude Code parser --------------------------------------------------------------


def test_claude_code_keeps_prompts_and_answer_text_only(tmp_path):
    parsed, resume = claude_code.parse(add_claude_session(tmp_path))

    assert [(m.role, m.content) for m in parsed.messages] == [
        ("user", "How should the poller retry 503s?"),
        ("assistant", "Use exponential backoff with full jitter.\n\nI added a retry helper to client.py."),
        ("user", "Cap it at 5 attempts. The token ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789 still fails."),
        ("assistant", "Capped at 5 attempts; that token needs rotating."),
    ]
    assert (parsed.provider, parsed.external_id, parsed.title) == ("claude-code", "sess-0001", "Poller retry strategy")
    assert parsed.metadata == {"capture": "transcript", "cwd": "/home/test/poller", "git_branch": "main",
                               "folder": "/home/test/poller"}
    assert parsed.messages[1].created_at == datetime(2026, 10, 1, 10, 0, 3, tzinfo=UTC)


def test_claude_code_resumes_at_the_last_prompt(tmp_path):
    path = add_claude_session(tmp_path)
    _, resume = claude_code.parse(path)
    line = path.read_bytes()[resume:].split(b"\n", 1)[0]
    assert json.loads(line)["uuid"] == "u-006"

    tail, again = claude_code.parse(path, resume)
    assert [m.content for m in tail.messages][0].startswith("Cap it at 5 attempts")
    assert tail.title is None and tail.created_at is None  # not in the re-read part: keep what's stored
    assert again == resume


def test_claude_code_leaves_a_half_written_line_for_later(tmp_path):
    path = add_claude_session(tmp_path)
    with path.open("a") as f:
        f.write('{"type": "user", "uuid": "u-9", "message": {"content": "half')
    parsed, _ = claude_code.parse(path)
    assert "half" not in " ".join(m.content for m in parsed.messages)


# --- VS Code parser ------------------------------------------------------------------


def test_vscode_session_parses_prompts_and_markdown(tmp_path):
    parsed, _ = vscode_copilot.parse(add_vscode_session(tmp_path))

    assert [(m.role, m.content) for m in parsed.messages] == [
        ("user", "How do I parse the Retry-After header?"),
        ("assistant", "Parse it as delta-seconds or an HTTP-date, as `client.py` already does for 429s."),
        ("user", "What did I ask Claude earlier?"),
        ("assistant", "You asked Claude how to structure retries."),
        ("user", "Never mind"),
    ]
    assert [m.external_id for m in parsed.messages[:2]] == ["request_1", "response_1"]
    assert (parsed.external_id, parsed.title) == ("vs-0001", "How do I parse the Retry-After header?")
    assert parsed.metadata == {"capture": "transcript", "workspace": "file:///home/test/poller",
                               "folder": str(Path("/home/test/poller")) if os.name == "nt" else "/home/test/poller"}
    assert parsed.created_at == datetime.fromtimestamp(1790000000, tz=UTC)


# --- capture into the store ----------------------------------------------------------


def test_capture_stores_conversations_another_assistant_can_read(conn, user_home):
    add_claude_session(user_home)
    add_vscode_session(user_home)

    report = capture(conn, CFG, user_home=user_home, days=36500)

    assert (report.files_read, report.conversations, report.messages, report.errors) == (2, 2, 9, [])
    found = recent_conversations(conn, GEMINI, source="claude-code").conversations
    assert [c.title for c in found] == ["Poller retry strategy"]
    hits = search_context(conn, GEMINI, "Retry-After header").results
    assert hits and hits[0].source == "vscode-copilot"


def test_capture_redacts_secrets(conn, user_home):
    add_claude_session(user_home)
    capture(conn, CFG, user_home=user_home, days=36500)
    [conv] = repo.list_conversations(conn, scopes=None)
    text = " ".join(content for _, content in contents(conn, conv.id))
    assert "ghp_" not in text and "[REDACTED]" in text


def test_unchanged_files_are_skipped(conn, user_home):
    add_claude_session(user_home)
    capture(conn, CFG, user_home=user_home, days=36500)
    again = capture(conn, CFG, user_home=user_home, days=36500)
    assert (again.files_read, again.messages) == (0, 0)


def test_a_growing_turn_is_updated_in_place_and_new_turns_are_added(conn, user_home):
    path = add_claude_session(user_home)
    capture(conn, CFG, user_home=user_home, days=36500)
    [conv] = repo.list_conversations(conn, scopes=None)
    before = repo.conversation_messages(conn, conv.id)

    append_lines(
        path,
        {"type": "assistant", "uuid": "a-007", "sessionId": "sess-0001", "timestamp": "2026-10-01T10:02:20.000Z",
         "message": {"role": "assistant", "content": [{"type": "text", "text": "Also log each retry."}]}},
        {"type": "user", "uuid": "u-008", "sessionId": "sess-0001", "timestamp": "2026-10-01T10:03:00.000Z",
         "message": {"role": "user", "content": "Ship it."}},
    )
    report = capture(conn, CFG, user_home=user_home, days=36500)

    after = repo.conversation_messages(conn, conv.id)
    assert report.messages == 2  # the last answer grew, one prompt was added
    assert [m.id for m in after[:4]] == [m.id for m in before]
    assert after[3].content.endswith("needs rotating.\n\nAlso log each retry.")
    assert after[4].content == "Ship it." and len(after) == 5
    updated_chunks = {r[0] for r in conn.execute("SELECT id FROM chunks WHERE message_id = ?", (after[3].id,))}
    [hit] = [h for h in search_context(conn, GEMINI, "log each retry").results if h.id in updated_chunks]
    assert hit.recall  # this turn called search_context (see the fixture)
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE message_id = ?", (after[3].id,)).fetchone()[0] == 1


def test_rewritten_vscode_session_adds_new_requests(conn, user_home):
    path = add_vscode_session(user_home)
    capture(conn, CFG, user_home=user_home, days=36500)

    data = json.loads(path.read_text())
    data["requests"].append({
        "requestId": "request_4", "responseId": "response_4", "timestamp": 1790000180000,
        "message": {"text": "Add a circuit breaker"}, "response": [{"value": "Added one."}],
    })
    path.write_text(json.dumps(data))
    bump_mtime(path)
    capture(conn, CFG, user_home=user_home, days=36500)

    [conv] = repo.list_conversations(conn, scopes=None)
    assert contents(conn, conv.id)[-2:] == [("user", "Add a circuit breaker"), ("assistant", "Added one.")]
    assert len(contents(conn, conv.id)) == 7


def test_a_deleted_conversation_is_not_captured_again(conn, user_home):
    path = add_claude_session(user_home)
    capture(conn, CFG, user_home=user_home, days=36500)
    [conv] = repo.list_conversations(conn, scopes=None)
    repo.delete_conversation(conn, conv.id)

    append_lines(path, {"type": "user", "uuid": "u-9", "sessionId": "sess-0001", "message": {"content": "More."}})
    capture(conn, CFG, user_home=user_home, days=36500)
    assert repo.list_conversations(conn, scopes=None) == []


def test_only_enabled_sources_and_recent_files_are_read(conn, user_home):
    add_claude_session(user_home)
    old = add_vscode_session(user_home)
    os.utime(old, (time.time() - 90 * 86400,) * 2)

    only_claude = Config(capture=CaptureConfig(sources=("claude-code",)))
    assert capture(conn, only_claude, user_home=user_home).files_read == 1
    assert capture(conn, CFG, user_home=user_home, days=30).files_read == 0
    assert capture(conn, Config(), user_home=user_home).files_read == 0


def test_capture_uses_the_sources_write_scope(conn, user_home):
    add_claude_session(user_home)
    cfg = Config.model_validate({
        "clients": {"claude-code": {"write_scope": "work"}},
        "capture": {"sources": ["claude-code"]},
    })
    capture(conn, cfg, user_home=user_home, days=36500)
    [conv] = repo.list_conversations(conn, scopes=None)
    assert conv.scope == "work"
    assert recent_conversations(conn, GEMINI).conversations == []  # GEMINI reads personal only


def test_time_budget_defers_files_to_the_next_run(conn, user_home):
    add_claude_session(user_home)
    add_vscode_session(user_home)
    report = capture(conn, CFG, user_home=user_home, days=36500, budget_ms=-1)
    assert (report.files_read, report.deferred) == (0, 2)
    assert capture(conn, CFG, user_home=user_home, days=36500).files_read == 2


def test_unreadable_transcripts_are_reported_not_raised(conn, user_home):
    path = add_vscode_session(user_home)
    path.write_text("{ not json")
    report = capture(conn, CFG, user_home=user_home, days=36500)
    assert len(report.errors) == 1 and "vs-0001.json" in report.errors[0]
    # Not retried until the file changes again.
    assert capture(conn, CFG, user_home=user_home, days=36500).errors == []


def test_a_file_that_shrank_is_read_from_the_top(conn, user_home):
    path = add_claude_session(user_home)
    capture(conn, CFG, user_home=user_home, days=36500)
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[:2]))  # just the first prompt
    bump_mtime(path)
    assert capture(conn, CFG, user_home=user_home, days=36500).errors == []
    [conv] = repo.list_conversations(conn, scopes=None)
    assert len(contents(conn, conv.id)) == 4  # nothing lost, nothing duplicated


def test_vscode_tool_messages_can_be_plain_strings(tmp_path):
    """File-editing tools store invocationMessage as a string, not {"value": ...}: a
    Copilot chat that wrote code crashed capture in the 2026-10-18 manual test."""
    path = add_vscode_session(tmp_path)
    data = json.loads(path.read_text())
    data["requests"].append({
        "requestId": "request_4", "responseId": "response_4", "timestamp": 1790000180000,
        "message": {"text": "Write a function that waters plants on a schedule"},
        "response": [
            {"kind": "toolInvocationSerialized", "invocationMessage": "Creating water.py", "toolCallId": "t9"},
            {"kind": "textEditGroup", "uri": {"path": "/home/test/water.py"}, "edits": []},
            {"value": "```python\ndef water_plants(schedule): ...\n```"},
        ],
        "result": {"metadata": {"toolCallRounds": [{"toolCalls": [{"name": "create_file", "arguments": "{}", "id": "c9"}]}]}},
    })
    path.write_text(json.dumps(data))
    parsed, _ = vscode_copilot.parse(path)
    assert parsed.messages[-1].content == "```python\ndef water_plants(schedule): ...\n```"
    assert not parsed.messages[-1].recall


def test_one_broken_transcript_doesnt_stop_the_others(conn, user_home, monkeypatch):
    from openmemore import capture as capture_module
    add_claude_session(user_home)
    add_vscode_session(user_home)
    original = capture_module.SOURCES["vscode-copilot"]

    def broken(path, offset):
        raise AttributeError("'str' object has no attribute 'get'")

    monkeypatch.setitem(capture_module.SOURCES, "vscode-copilot", capture_module.Source(original.files, broken, original.app))
    report = capture(conn, CFG, user_home=user_home, days=36500)
    assert report.conversations == 1  # Claude Code's session still stored
    assert len(report.errors) == 1 and "AttributeError" in report.errors[0]


# --- capturing by folder ([capture] include_folders / exclude_folders) ----------------------


def session_in(user_home: Path, folder: str, name: str, prompt: str) -> Path:
    """A one-prompt Claude Code session started in `folder`."""
    path = user_home / ".claude" / "projects" / name / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "user", "sessionId": name, "uuid": f"{name}-1", "cwd": folder,
                                "timestamp": "2026-10-19T10:00:00Z",
                                "message": {"role": "user", "content": prompt}}) + "\n")
    return path


def folders(include=(), exclude=()):
    return Config(capture=CaptureConfig(sources=("claude-code", "vscode-copilot"),
                                        include_folders=include, exclude_folders=exclude))


def titles(conn):
    return sorted(c.title for c in repo.list_conversations(conn, scopes=None))


def test_excluded_folders_and_their_subfolders_aren_t_captured(conn, user_home):
    session_in(user_home, "/work/client", "a", "Client secret plans.")
    session_in(user_home, "/work/client/api", "b", "Client API work.")
    session_in(user_home, "/work/client-tools", "c", "My own tools.")  # a sibling, not inside
    report = capture(conn, folders(exclude=("/work/client/",)), user_home=user_home, days=36500)
    assert titles(conn) == ["My own tools."] and report.filtered == 2


def test_include_folders_limit_capture_and_exclusions_win(conn, user_home):
    session_in(user_home, "/home/me/projects/garden", "a", "Garden app.")
    session_in(user_home, "/home/me/projects/garden/private", "b", "Private notes.")
    session_in(user_home, "/home/me/elsewhere", "c", "Elsewhere.")
    add_vscode_session(user_home)  # /home/test/poller
    cfg = folders(include=("/home/me/projects",), exclude=("/home/me/projects/garden/private",))
    capture(conn, cfg, user_home=user_home, days=36500)
    assert titles(conn) == ["Garden app."]


def test_without_a_folder_a_chat_is_captured_unless_folders_are_included(conn, user_home):
    path = session_in(user_home, "/x", "a", "Where am I?")
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    del lines[0]["cwd"]
    path.write_text(json.dumps(lines[0]) + "\n")
    capture(conn, folders(include=("/home",)), user_home=user_home, days=36500)
    assert titles(conn) == []
    capture(conn, folders(), user_home=user_home, days=36500)
    assert titles(conn) == ["Where am I?"]


def test_a_skipped_transcript_isn_t_read_again_until_its_folder_is_captured(conn, user_home, monkeypatch):
    from openmemore.capture import SOURCES, Source

    path = session_in(user_home, "/work/client", "a", "First.")
    excluded = folders(exclude=("/work/client",))
    capture(conn, excluded, user_home=user_home, days=36500)
    reads = []
    real = SOURCES["claude-code"]
    monkeypatch.setitem(SOURCES, "claude-code",
                        Source(real.files, lambda p, o: reads.append(o) or real.parse(p, o), real.app))
    append_lines(path, {"type": "user", "sessionId": "a", "uuid": "a-2", "cwd": "/work/client",
                        "timestamp": "2026-10-19T10:05:00Z", "message": {"role": "user", "content": "Second."}})
    report = capture(conn, excluded, user_home=user_home, days=36500)
    assert reads == [] and report.filtered == 1  # known to be excluded: not read
    capture(conn, folders(), user_home=user_home, days=36500)  # the folder is captured again
    assert reads == [0]  # read from the top, so nothing is missed
    [c] = repo.list_conversations(conn, scopes=None)
    assert [content for _, content in contents(conn, c.id)] == ["First.", "Second."]


def test_excluding_a_folder_later_stops_capturing_it_and_keeps_what_was_stored(conn, user_home):
    path = session_in(user_home, "/work/client", "a", "First.")
    capture(conn, folders(), user_home=user_home, days=36500)
    append_lines(path, {"type": "user", "sessionId": "a", "uuid": "a-2", "cwd": "/work/client",
                        "timestamp": "2026-10-19T10:05:00Z", "message": {"role": "user", "content": "Second."}})
    capture(conn, folders(exclude=("/work/client",)), user_home=user_home, days=36500)
    [c] = repo.list_conversations(conn, scopes=None)
    assert [content for _, content in contents(conn, c.id)] == ["First."]


def test_folders_take_home_shortcuts_and_vscode_workspaces(conn, user_home, monkeypatch):
    monkeypatch.setenv("HOME", "/home/test")
    monkeypatch.setenv("USERPROFILE", "/home/test")
    add_vscode_session(user_home)  # workspace file:///home/test/poller
    capture(conn, folders(exclude=("~/poller",)), user_home=user_home, days=36500)
    assert titles(conn) == []


def test_setup_mentions_folders_quietly_when_it_turns_on_capture(tmp_path):
    from openmemore.setup import Context, plan

    ctx = Context(base=tmp_path, user_home=tmp_path, platform="darwin", command=("openmemore",))
    notes = [s.note for s in plan("vscode-copilot", ctx) if s.note]
    assert notes == ["Only some projects? List folders under include_folders or exclude_folders in [capture]."]
    assert not any(s.note for s in plan("cursor", ctx))
