"""SPEC.md M6: `openmcp daemon`. The receiver only takes its own token, on its own host,
as JSON; the capture loop stores changed transcripts without a client reading; the
service definitions run the daemon at login on each OS."""

import json
import os
import plistlib
import socket
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

import pytest
from starlette.testclient import TestClient
from typer.testing import CliRunner

from conftest import plain
from openmcp import config, daemon, db, repo, service
from openmcp.cli import app, initialize
from test_otel import otlp, prompt, response

PORT = 47999


@pytest.fixture
def receiver(home):
    initialize(home)
    token = daemon.ensure_token(home)
    client = TestClient(daemon.build_app(home, token, PORT), base_url=f"http://127.0.0.1:{PORT}")
    return client, {daemon.TOKEN_HEADER: token}


def test_the_token_is_made_once_and_kept(home):
    initialize(home)
    token = daemon.ensure_token(home)
    assert len(token) >= 24 and daemon.ensure_token(home) == token
    assert config.load_config(home / "config.toml").daemon.token == token


def test_the_receiver_stores_conversations(receiver, home):
    client, auth = receiver
    r = client.post("/v1/logs", json=otlp(prompt("How do I parse TOML?"), response("Use tomllib.")), headers=auth)
    assert (r.status_code, r.json()) == (200, {"partialSuccess": {}})
    with closing(db.connect(home / "openmcp.db")) as conn:
        [c] = repo.list_conversations(conn, scopes=None)
        assert [m.content for m in repo.conversation_messages(conn, c.id)] == ["How do I parse TOML?", "Use tomllib."]
        assert conn.execute("SELECT client, tool FROM audit_log").fetchall()[-1][:] == ("daemon", "otel")
    assert client.get("/health").json()["ok"] is True


@pytest.mark.parametrize("change, status", [
    ({"headers": {"host": "evil.example"}}, 403),
    ({"headers": {daemon.TOKEN_HEADER: "wrong"}}, 401),
    ({"headers": {daemon.TOKEN_HEADER: None}}, 401),
    ({"headers": {"content-type": "text/plain"}}, 415),  # what a web page can send without a preflight
    ({"headers": {"content-type": "application/x-protobuf"}}, 415),
])
def test_the_receiver_refuses(receiver, home, change, status):
    client, auth = receiver
    headers = {**auth, "content-type": "application/json", **change["headers"]}
    headers = {k: v for k, v in headers.items() if v is not None}
    body = json.dumps(otlp(prompt("Injected?"), response("Yes.")))
    assert client.post("/v1/logs", content=body, headers=headers).status_code == status
    with closing(db.connect(home / "openmcp.db")) as conn:
        assert repo.list_conversations(conn, scopes=None) == []


def write_transcript(user_home: Path, session: str, text: str) -> None:
    project = user_home / ".claude" / "projects" / "-work"
    project.mkdir(parents=True, exist_ok=True)
    line = {"type": "user", "sessionId": session, "uuid": f"{session}-u", "timestamp": "2026-10-19T10:00:00Z",
            "message": {"role": "user", "content": text}}
    (project / f"{session}.jsonl").write_text(json.dumps(line) + "\n")


def test_a_tick_captures_changed_transcripts_without_a_client(home, isolated_user):
    initialize(home)
    write_transcript(isolated_user, "s1", "Plan a garden.")
    assert daemon.tick(home).messages == 0  # capture is off until a source is enabled
    path = home / "config.toml"
    cfg = config.load_config(path)
    cfg.capture.sources = ("claude-code",)
    config.save_config(path, cfg)  # picked up without a restart
    assert daemon.tick(home).messages == 1
    assert daemon.tick(home).messages == 0  # unchanged since
    with closing(db.connect(home / "openmcp.db")) as conn:
        assert [r[0] for r in conn.execute("SELECT tool FROM audit_log WHERE client = 'daemon'")] == ["capture"]


def test_a_tick_embeds_what_is_pending(home, fake_embedder):
    initialize(home)
    from conftest import insert

    with closing(db.connect(home / "openmcp.db")) as conn:
        insert(conn, "Prefers tea.")
    assert daemon.tick(home).embedded == 1


def test_status_without_a_daemon(home):
    initialize(home)
    assert daemon.healthy(PORT) is None and daemon.read_state(home) is None


# --- service definitions ---------------------------------------------------------------

COMMAND = ("/opt/tools/openmcp/bin/openmcp",)


def test_macos_launchd_agent(tmp_path):
    svc = service.describe("darwin", COMMAND, tmp_path / "data", tmp_path / "home", uid=501)
    assert svc.file == tmp_path / "home" / "Library" / "LaunchAgents" / "dev.openmcp.daemon.plist"
    plist = plistlib.loads(svc.content.encode())
    assert plist["ProgramArguments"] == [*COMMAND, "daemon", "run", "--home", str(tmp_path / "data"),
                                         "--log-file", str(tmp_path / "data" / "logs" / "daemon.log")]
    assert plist["RunAtLoad"] and plist["KeepAlive"]
    assert svc.install[-1] == ["launchctl", "bootstrap", "gui/501", str(svc.file)]


def test_linux_systemd_unit(tmp_path):
    svc = service.describe("linux", ("/home/me/.local/bin/openmcp",), Path("/home/me/my data"), tmp_path)
    assert svc.file == tmp_path / ".config" / "systemd" / "user" / "openmcp.service"
    assert "ExecStart=/home/me/.local/bin/openmcp daemon run --home '/home/me/my data'" in svc.content
    assert "WantedBy=default.target" in svc.content
    assert svc.install[-1] == ["systemctl", "--user", "enable", "--now", "openmcp.service"]
    assert any("enable-linger" in n for n in svc.notes)


def test_windows_logon_task_uses_pythonw_when_there_is_one(tmp_path):
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    (scripts / "pythonw.exe").write_text("")
    svc = service.describe("win32", (str(scripts / "openmcp.exe"),), tmp_path / "data", tmp_path)
    create = svc.install[0]
    task = create[create.index("/TR") + 1]
    assert "pythonw.exe" in task and "-m openmcp daemon run" in task and "/SC" in create and "ONLOGON" in create
    without = service.describe("win32", (str(tmp_path / "openmcp.exe"),), tmp_path / "data", tmp_path)
    assert "openmcp.exe daemon run" in without.install[0][without.install[0].index("/TR") + 1]


def test_unknown_platform():
    with pytest.raises(service.ServiceError):
        service.describe("plan9", COMMAND, Path("/d"), Path("/h"))


class FakeRun:
    def __init__(self, fail=()):
        self.calls, self.fail = [], set(fail)

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        code = 1 if cmd[1] in self.fail else 0
        return subprocess.CompletedProcess(cmd, code, stdout="", stderr="no such service" if code else "")


def test_installing_and_removing(tmp_path):
    svc = service.describe("darwin", COMMAND, tmp_path / "data", tmp_path / "home", uid=501)
    run = FakeRun(fail={"bootout"})  # no earlier copy to remove: fine
    service.install(svc, run)
    assert svc.file.exists() and [c[1] for c in run.calls] == ["bootout", "bootstrap"]
    assert service.installed(svc)
    assert service.uninstall(svc, FakeRun()) and not svc.file.exists()

    failing = FakeRun(fail={"bootstrap"})
    with pytest.raises(service.ServiceError, match="no such service"):
        service.install(svc, failing)


# --- CLI ---------------------------------------------------------------------------------


def cli(*args):
    result = CliRunner().invoke(app, list(args), env={"COLUMNS": "200"})
    return result.exit_code, plain(result.output)


def test_telemetry_settings_for_claude_code(home):
    code, out = cli("daemon", "telemetry")
    env = json.loads(out[out.index("{"):])["env"]
    token = config.load_config(home / "config.toml").daemon.token
    assert code == 0 and env["OTEL_EXPORTER_OTLP_LOGS_HEADERS"] == f"x-openmcp-token={token}"
    assert env["OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"] == "http://127.0.0.1:47318/v1/logs"
    assert env["OTEL_LOG_USER_PROMPTS"] == "1" and env["OTEL_EXPORTER_OTLP_LOGS_PROTOCOL"] == "http/json"
    code, out = cli("daemon", "telemetry", "cursor")
    assert code == 2 and "doesn't export" in out


def test_install_dry_run_changes_nothing(home, isolated_user):
    code, out = cli("daemon", "install", "--dry-run")
    assert code == 0 and "dry run" in out and "--home" in out and "--log-file" in out
    assert not any(isolated_user.rglob("*.plist")) and not any(isolated_user.rglob("openmcp.service"))


# --- the real process --------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_the_daemon_process_receives_and_captures(home, isolated_user):
    """`python -m openmcp daemon run`, as the services start it: it answers health checks,
    stores what an exporter sends, captures transcripts on its own, and says where it is."""
    initialize(home)
    port = free_port()
    path = home / "config.toml"
    cfg = config.load_config(path)
    cfg.daemon.port, cfg.daemon.interval = port, 0.5
    cfg.capture.sources = ("claude-code",)
    config.save_config(path, cfg)
    token = daemon.ensure_token(home)
    log = home / "logs" / "daemon.log"
    env = {**os.environ, "HOME": str(isolated_user), "USERPROFILE": str(isolated_user)}
    proc = subprocess.Popen([sys.executable, "-m", "openmcp", "daemon", "run", "--home", str(home),
                             "--log-file", str(log)], env=env)
    try:
        assert daemon.wait_until_healthy(port, 30), log.read_text() if log.exists() else "no log"
        assert daemon.read_state(home)["pid"] == proc.pid

        import urllib.request

        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/logs", data=json.dumps(otlp(prompt("Hi from OTel."))).encode(),
            headers={"content-type": "application/json", daemon.TOKEN_HEADER: token}, method="POST")
        with urllib.request.urlopen(request, timeout=10) as r:
            assert r.status == 200
        write_transcript(isolated_user, "s2", "Hi from a transcript.")
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            with closing(db.connect(home / "openmcp.db")) as conn:
                titles = {c.title for c in repo.list_conversations(conn, scopes=None)}
            if titles == {"Hi from OTel.", "Hi from a transcript."}:
                break
            time.sleep(0.2)
        assert titles == {"Hi from OTel.", "Hi from a transcript."}
        assert "listening on" in log.read_text()
    finally:
        proc.terminate()
        proc.wait(timeout=15)
