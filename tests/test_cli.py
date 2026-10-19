import json
import os
import shutil
import subprocess
import sys
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import caller, insert, plain
from openmcp import db
from openmcp.audit import percentile, stats
from openmcp.cli import app, initialize
from openmcp.tools.conversations import record_exchange

runner = CliRunner()


@pytest.fixture
def store(home):
    initialize(home)
    with closing(db.connect(home / "openmcp.db")) as conn:
        a = insert(conn, "Prefers Python.", kind="preference", scope="personal")
        b = insert(conn, "Work uses Go.", kind="fact", scope="work", project="Svc")
        c = insert(conn, "Work uses Go 1.23.", kind="fact", scope="work", project="Svc")
        conn.execute("UPDATE memories SET superseded_by = ? WHERE id = ?", (c, b))
        yield conn, (a, b, c)


@dataclass
class Run:
    exit_code: int
    output: str


def run(*args, input=None) -> Run:
    result = runner.invoke(app, list(args), input=input, env={"COLUMNS": "200"})
    return Run(result.exit_code, plain(result.output))


def test_list_shows_active_memories_from_every_scope(store):
    out = run("memories", "list").output
    assert "Prefers Python." in out and "Work uses Go 1.23." in out
    assert "Work uses Go." not in out


def test_list_filters_and_all(store):
    assert "Prefers Python." not in run("memories", "list", "--scope", "work").output
    assert "Prefers Python." not in run("memories", "list", "--kind", "fact").output
    assert "replaced by" in run("memories", "list", "--all").output


def test_list_rejects_bad_kind(store):
    result = run("memories", "list", "--kind", "skill")
    assert result.exit_code == 2 and "--kind" in result.output


def test_show_includes_history(store):
    _, b, c = store[1]
    out = run("memories", "show", str(b)).output
    assert f"replaced by {c}" in out and "History" in out


def test_show_missing(store):
    assert run("memories", "show", "999").exit_code == 1


def test_rm_asks_and_can_be_declined(store):
    conn, (a, _, _) = store
    result = run("memories", "rm", str(a), input="n\n")
    assert result.exit_code == 1
    assert conn.execute("SELECT COUNT(*) FROM memories WHERE id = ?", (a,)).fetchone()[0] == 1


def test_rm_deletes_whole_chain_and_is_audited(store):
    conn, (_, b, c) = store
    result = run("memories", "rm", str(c), input="y\n")
    assert result.exit_code == 0 and "Deleted 2" in result.output
    assert conn.execute("SELECT COUNT(*) FROM memories WHERE id IN (?, ?)", (b, c)).fetchone()[0] == 0
    row = conn.execute("SELECT client, tool, result_count FROM audit_log WHERE tool = 'memories rm'").fetchone()
    assert tuple(row) == ("cli", "memories rm", 2)


def test_audit_lists_and_filters(store):
    run("memories", "list")
    run("memories", "show", "1")
    out = run("audit").output
    assert "memories list" in out and "memories show" in out
    assert "memories list" not in run("audit", "--tool", "memories show").output


def test_audit_stats(store):
    for _ in range(3):
        run("memories", "list")
    out = run("audit", "--stats").output
    assert "p95 ms" in out and "memories list" in out


def test_percentiles():
    values = [float(v) for v in range(1, 101)]
    assert (percentile(values, 50), percentile(values, 95)) == (50.0, 95.0)
    assert percentile([7.0], 95) == 7.0
    [s] = stats({"t": [1.0, 2.0, 3.0]})
    assert (s.calls, s.p50_ms, s.max_ms) == (3, 2.0, 3.0)


# --- conversations -------------------------------------------------------------------


@pytest.fixture
def chats(home):
    initialize(home)
    with closing(db.connect(home / "openmcp.db")) as conn:
        a = record_exchange(conn, caller("claude-desktop"), "How do I rate-limit Flask?", "Flask-Limiter.")
        b = record_exchange(conn, caller("gemini-cli"), "Explain WAL mode", "Readers don't block writers.")
        yield conn, a.conversation_id, b.conversation_id


def test_conversations_list_and_filter(chats):
    out = run("conversations", "list").output
    assert "How do I rate-limit Flask?" in out and "Explain WAL mode" in out
    assert "Explain WAL mode" not in run("conversations", "list", "--source", "claude").output


def test_conversations_show(chats):
    _, a, _ = chats
    out = run("conversations", "show", str(a)).output
    assert "How do I rate-limit Flask?" in out and "Flask-Limiter." in out and "assistant" in out
    assert run("conversations", "show", "999").exit_code == 1


def test_conversations_rm(chats):
    conn, a, b = chats
    result = run("conversations", "rm", str(a), "--yes")
    assert result.exit_code == 0 and "2 messages" in result.output
    assert [r[0] for r in conn.execute("SELECT id FROM conversations")] == [b]
    assert run("conversations", "rm", str(b), input="n\n").exit_code == 1


# --- setup and capture ---------------------------------------------------------------


def desktop_config_path(user_home):
    from openmcp.setup import Context, claude_desktop_config
    return claude_desktop_config(Context(base=user_home, user_home=user_home, platform=sys.platform, command=("x",)))


def test_setup_dry_run_changes_nothing(home):
    user_home = Path(os.environ["HOME"])
    result = run("setup", "claude-desktop", "--dry-run")
    assert result.exit_code == 0 and "dry run: not changed" in result.output
    assert "Settings → Profile" in result.output
    assert not desktop_config_path(user_home).exists()


def test_setup_writes_the_client_config(home):
    user_home = Path(os.environ["HOME"])
    result = run("setup", "claude-desktop")
    assert result.exit_code == 0, result.output
    entry = json.loads(desktop_config_path(user_home).read_text())["mcpServers"]["openmcp"]
    assert entry["args"][-3:] == ["serve", "--client", "claude-desktop"]
    assert entry["env"] == {"OPENMCP_HOME": str(home)}  # tests use a non-default data directory


def test_setup_turns_on_capture_and_catches_up(home, monkeypatch):
    monkeypatch.setenv("PATH", "")  # never run the real claude CLI from a test
    user_home = Path(os.environ["HOME"])
    project = user_home / ".claude" / "projects" / "-home-test-poller"
    project.mkdir(parents=True)
    shutil.copy(Path(__file__).parent / "fixtures" / "claude_code" / "session.jsonl", project / "s.jsonl")

    result = run("setup", "claude-code")
    assert result.exit_code == 1  # no claude CLI on the test PATH: that step is printed instead
    assert "claude mcp add" in result.output
    assert "4 messages added or updated in 1 conversations" in result.output
    assert "Poller retry strategy" in run("conversations", "list").output


def test_capture_command(home):
    assert "Transcript capture is off" in run("capture").output
    result = run("capture", "--source", "slack")
    assert result.exit_code == 2 and "--source" in result.output
    result = run("capture", "--source", "vscode-copilot", "--days", "7")
    assert result.exit_code == 0 and "read 0 transcript files" in result.output


# --- embed ---------------------------------------------------------------------------


def test_embed_command(store, fake_embedder):
    conn, _ = store  # three memories inserted without vectors
    result = run("embed")
    assert result.exit_code == 0, result.output
    assert "embedded 2 memories" in result.output  # the superseded version isn't embedded
    assert conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0] == 2


def test_embed_without_a_model_says_so(store):
    result = run("embed")
    assert result.exit_code == 1
    assert "Semantic search unavailable" in result.output and "Keyword search still works" in result.output


def test_capture_still_succeeds_without_a_model(home):
    result = run("capture", "--source", "vscode-copilot")
    assert result.exit_code == 0 and "Semantic search unavailable" in result.output


# --- setup from a development checkout -----------------------------------------------


@pytest.fixture
def checkout(monkeypatch, tmp_path):
    from openmcp import setup as client_setup
    monkeypatch.setattr(client_setup, "development_checkout", lambda: tmp_path / "repo")
    return client_setup


def test_setup_from_a_checkout_hands_off_to_the_installed_copy(home, checkout, monkeypatch):
    calls = []
    monkeypatch.setattr(checkout, "installed_command", lambda root: ("/opt/tools/openmcp",))
    monkeypatch.setattr("openmcp.cli.subprocess.run", lambda args: calls.append(args) or subprocess.CompletedProcess(args, 0))
    result = run("setup", "vscode-copilot", "--dry-run")
    assert result.exit_code == 0 and "installed openmcp at /opt/tools/openmcp" in " ".join(result.output.split())
    assert calls == [["/opt/tools/openmcp", "setup", "vscode-copilot", "--dry-run"]]
    assert not (Path(os.environ["HOME"]) / "Library").exists()  # nothing configured from the checkout


def test_setup_from_a_checkout_without_an_installed_copy_stops(home, checkout, monkeypatch):
    monkeypatch.setattr(checkout, "installed_command", lambda root: None)
    result = run("setup", "claude-desktop")
    assert result.exit_code == 2 and "no installed openmcp was found" in " ".join(result.output.split())


def test_setup_dev_registers_the_checkout(home, checkout, monkeypatch):
    monkeypatch.setattr(checkout, "installed_command", lambda root: pytest.fail("shouldn't look"))
    result = run("setup", "claude-desktop", "--dev")
    assert result.exit_code == 0, result.output
    assert json.loads(desktop_config_path(Path(os.environ["HOME"])).read_text())["mcpServers"]["openmcp"]
