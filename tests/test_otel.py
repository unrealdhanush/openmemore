"""SPEC.md M6: conversations captured from Claude Code's OpenTelemetry log events, with
the same ids as its transcript reader, keeping only ids and text. Synthetic events,
shaped like the ones Claude Code 2.1 sent in the Section 14 experiment."""

import json
from contextlib import closing

import pytest

from openmcp import config, db, repo
from openmcp.capture import capture, claude_code
from openmcp.capture.otel import events, ingest
from openmcp.cli import initialize

SESSION = "0b9f7a52-3f0e-4c38-9f0a-1d2e3f4a5b6c"
EMAIL = "someone@example.com"


def attr(key, value):
    kind = "boolValue" if isinstance(value, bool) else "intValue" if isinstance(value, int) else "stringValue"
    return {"key": key, "value": {kind: str(value) if kind == "intValue" else value}}


def record(name, seq, at, **attributes):
    """One log record as Claude Code exports it, with identity attributes that must never be stored."""
    attributes = {
        "event.name": name, "event.sequence": seq, "event.timestamp": at, "session.id": SESSION,
        "user.email": EMAIL, "user.account_uuid": "acct-1", "organization.id": "org-1", "terminal.type": "vscode",
        **attributes,
    }
    return {"timeUnixNano": "1", "body": {"stringValue": f"claude_code.{name}"},
            "attributes": [attr(k, v) for k, v in attributes.items()]}


def otlp(*records):
    return {"resourceLogs": [{
        "resource": {"attributes": [attr("service.name", "claude-code")]},
        "scopeLogs": [{"scope": {"name": "com.anthropic.claude_code.events"}, "logRecords": list(records)}],
    }]}


def prompt(text, seq=1, pid="p1", uuid="u1", at="2026-10-19T10:00:00.000Z"):
    return record("user_prompt", seq, at, **{"prompt.id": pid, "message.uuid": uuid, "prompt": text,
                                             "prompt_length": len(text)})


def response(text, seq=2, pid="p1", at="2026-10-19T10:00:05.000Z"):
    return record("assistant_response", seq, at, **{"prompt.id": pid, "response": text, "message.uuid": f"r{seq}",
                                                    "model": "some-model"})


def tool(name, seq=3, pid="p1"):
    return record("tool_result", seq, "2026-10-19T10:00:03.000Z", **{"prompt.id": pid, "tool_name": name,
                                                                     "success": True})


@pytest.fixture
def cfg(home):
    initialize(home)
    return config.load_config(home / "config.toml")


def stored(conn):
    return [
        (c.provider, c.external_id, c.title, [(m.external_id, m.role, m.content, m.recall)
                                               for m in repo.conversation_messages(conn, c.id)])
        for c in repo.list_conversations(conn, scopes=None)
    ]


def test_a_turn_becomes_a_conversation(conn, cfg):
    report = ingest(conn, cfg, otlp(prompt("How do I parse TOML?"), response("Use tomllib.")))
    assert report.messages == 2 and report.skipped == []
    assert stored(conn) == [("claude-code", SESSION, "How do I parse TOML?", [
        ("u1", "user", "How do I parse TOML?", False), ("u1:answer", "assistant", "Use tomllib.", False),
    ])]
    [c] = repo.list_conversations(conn, scopes=None)
    assert (c.scope, c.metadata) == ("personal", {"capture": "otel"})


def test_only_ids_and_text_are_kept(conn, cfg):
    ingest(conn, cfg, otlp(prompt("Hi there, help me."), response("Sure.")))
    dump = "\n".join(str(tuple(row)) for row in conn.iterdump())
    assert EMAIL not in dump and "acct-1" not in dump and "org-1" not in dump and "vscode" not in dump


def test_an_answer_grows_as_its_parts_arrive(conn, cfg):
    ingest(conn, cfg, otlp(prompt("Plan a garden.")))
    ingest(conn, cfg, otlp(response("First, test the soil.", seq=2)))
    ingest(conn, cfg, otlp(response("Then pick plants.", seq=4), response("First, test the soil.", seq=2)))  # a resend
    [(_, _, _, messages)] = stored(conn)
    assert messages[1][2] == "First, test the soil.\n\nThen pick plants."


def test_otel_and_the_transcript_reader_build_the_same_conversation(conn, cfg, isolated_user):
    """Both sources can run: they write the same messages, not two copies."""
    project = isolated_user / ".claude" / "projects" / "-work-garden"
    project.mkdir(parents=True)
    lines = [
        {"type": "user", "sessionId": SESSION, "uuid": "u1", "timestamp": "2026-10-19T10:00:00.000Z",
         "message": {"role": "user", "content": "Plan a garden."}},
        {"type": "assistant", "sessionId": SESSION, "uuid": "a1", "timestamp": "2026-10-19T10:00:04.000Z",
         "message": {"role": "assistant", "content": [{"type": "text", "text": "First, test the soil."}]}},
        {"type": "assistant", "sessionId": SESSION, "uuid": "a2", "timestamp": "2026-10-19T10:00:06.000Z",
         "message": {"role": "assistant", "content": [{"type": "text", "text": "Then pick plants."}]}},
    ]
    (project / f"{SESSION}.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    ingest(conn, cfg, otlp(prompt("Plan a garden."), response("First, test the soil.", seq=2),
                           response("Then pick plants.", seq=4)))
    cfg.capture.sources = ("claude-code",)
    report = capture(conn, cfg, user_home=isolated_user)
    assert report.messages == 0  # nothing new: the transcript says what the events said
    [(_, _, _, messages)] = stored(conn)
    assert [m[0] for m in messages] == ["u1", "u1:answer"]


def test_a_turn_that_used_a_recall_tool_is_a_recap(conn, cfg):
    ingest(conn, cfg, otlp(prompt("What did I ask Copilot?"), tool("mcp__openmcp__recent_conversations"),
                           response("You asked about lists.")))
    ingest(conn, cfg, otlp(prompt("What did I ask Copilot?")))  # a resend keeps the mark
    [(_, _, _, messages)] = stored(conn)
    assert [m[3] for m in messages] == [True, True]


def test_other_tools_don_t_make_a_recap(conn, cfg):
    ingest(conn, cfg, otlp(prompt("Fix the bug."), tool("Edit"), response("Fixed.")))
    [(_, _, _, messages)] = stored(conn)
    assert [m[3] for m in messages] == [False, False]


def test_what_can_t_be_used_is_reported(conn, cfg):
    report = ingest(conn, cfg, otlp(prompt("<REDACTED>"), response("An answer to an unknown prompt.", pid="p9")))
    assert stored(conn) == []
    assert any("OTEL_LOG_USER_PROMPTS" in r for r in report.skipped)
    assert any("wasn't received" in r for r in report.skipped)


def test_slash_commands_aren_t_conversation(conn, cfg):
    ingest(conn, cfg, otlp(prompt("<command-name>/clear</command-name>"), response("(cleared)")))
    assert stored(conn) == []


def test_a_deleted_conversation_stays_deleted(conn, cfg):
    ingest(conn, cfg, otlp(prompt("Plan a garden."), response("Test the soil.")))
    [c] = repo.list_conversations(conn, scopes=None)
    with db.transaction(conn):
        repo.delete_conversation(conn, c.id)
    assert conn.execute("SELECT COUNT(*) FROM otel_prompts").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM otel_responses").fetchone()[0] == 0
    ingest(conn, cfg, otlp(prompt("And the fence?", seq=5, pid="p2", uuid="u2"), response("Cedar.", seq=6, pid="p2")))
    assert stored(conn) == []


def test_the_scope_is_claude_codes(home):
    initialize(home)
    path = home / "config.toml"
    cfg = config.load_config(path)
    cfg.clients["claude-code"] = config.ClientOverride(read_scopes=("work",), write_scope="work")
    with closing(db.open_db(home / "openmcp.db")) as conn:
        ingest(conn, cfg, otlp(prompt("Deploy steps?"), response("Run make deploy.")))
        assert [c.scope for c in repo.list_conversations(conn, scopes=None)] == ["work"]


def test_malformed_payloads_are_ignored():
    for payload in (None, [], {"resourceLogs": "x"}, {"resourceLogs": [{"scopeLogs": [{"logRecords": [{}]}]}]}):
        assert list(events(payload)) == []
    [event] = events(otlp(prompt("hi")))
    assert (event.name, event.sequence) == ("user_prompt", 1)
