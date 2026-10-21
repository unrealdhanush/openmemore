"""SPEC.md M6: `openmemore daemon`. The receiver only takes its own token, on its own host,
as JSON; the capture loop stores changed transcripts without a client reading; the
service definitions run the daemon at login on each OS."""

import json
import os
import plistlib
import shlex
import signal
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
from openmemore import config, daemon, db, repo, service
from openmemore.cli import app, initialize
from test_otel import call, logs_json, text, traces_json, traces_pb

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
    r = client.post("/v1/traces", json=traces_json(call([text("user", "How do I parse TOML?")], "Use tomllib.")),
                    headers=auth)
    assert (r.status_code, r.json()) == (200, {"partialSuccess": {}})
    with closing(db.connect(home / "openmemore.db")) as conn:
        [c] = repo.list_conversations(conn, scopes=None)
        assert [m.content for m in repo.conversation_messages(conn, c.id)] == ["How do I parse TOML?", "Use tomllib."]
        assert conn.execute("SELECT client, tool FROM audit_log").fetchall()[-1][:] == ("daemon", "otel")
    assert client.get("/health").json()["ok"] is True


def test_the_receiver_takes_protobuf_gzip_and_logs(receiver, home):
    import gzip

    client, auth = receiver
    body = traces_pb(call([text("user", "First?")], "One."))
    r = client.post("/v1/traces", content=body, headers={**auth, "content-type": "application/x-protobuf"})
    assert (r.status_code, r.content, r.headers["content-type"]) == (200, b"", "application/x-protobuf")
    zipped = gzip.compress(json.dumps(logs_json(call([text("user", "Second?")], "Two."))).encode())
    r = client.post("/v1/logs", content=zipped,
                    headers={**auth, "content-type": "application/json", "content-encoding": "gzip"})
    assert r.status_code == 200
    with closing(db.connect(home / "openmemore.db")) as conn:
        assert sorted(c.title for c in repo.list_conversations(conn, scopes=None)) == ["First?", "Second?"]


def test_metrics_are_accepted_and_dropped(receiver, home):
    client, auth = receiver
    assert client.post("/v1/metrics", json={"resourceMetrics": []}, headers=auth).status_code == 200


@pytest.mark.parametrize("change, status", [
    ({"headers": {"host": "evil.example"}}, 403),
    ({"headers": {daemon.TOKEN_HEADER: "wrong"}}, 401),
    ({"headers": {daemon.TOKEN_HEADER: None}}, 401),
    ({"headers": {"content-type": "text/plain"}}, 415),  # what a web page can send without a preflight
    ({"headers": {"content-type": "application/x-protobuf"}}, 400),  # JSON sent as protobuf: not OTLP
])
def test_the_receiver_refuses(receiver, home, change, status):
    client, auth = receiver
    headers = {**auth, "content-type": "application/json", **change["headers"]}
    headers = {k: v for k, v in headers.items() if v is not None}
    body = json.dumps(traces_json(call([text("user", "Injected?")], "Yes.")))
    assert client.post("/v1/traces", content=body, headers=headers).status_code == status
    with closing(db.connect(home / "openmemore.db")) as conn:
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
    with closing(db.connect(home / "openmemore.db")) as conn:
        assert [r[0] for r in conn.execute("SELECT tool FROM audit_log WHERE client = 'daemon'")] == ["capture"]


def test_a_tick_embeds_what_is_pending(home, fake_embedder):
    initialize(home)
    from conftest import insert

    with closing(db.connect(home / "openmemore.db")) as conn:
        insert(conn, "Prefers tea.")
    assert daemon.tick(home).embedded == 1


def test_status_without_a_daemon(home):
    initialize(home)
    assert daemon.healthy(PORT) is None and daemon.read_state(home) is None


# --- service definitions ---------------------------------------------------------------

COMMAND = ("/opt/tools/openmemore/bin/openmemore",)


def test_macos_launchd_agent(tmp_path):
    svc = service.describe("darwin", COMMAND, tmp_path / "data", tmp_path / "home", uid=501)
    assert svc.file == tmp_path / "home" / "Library" / "LaunchAgents" / "dev.openmemore.daemon.plist"
    plist = plistlib.loads(svc.content.encode())
    assert plist["ProgramArguments"] == [*COMMAND, "daemon", "run", "--home", str(tmp_path / "data"),
                                         "--log-file", str(tmp_path / "data" / "logs" / "daemon.log")]
    assert plist["RunAtLoad"] and plist["KeepAlive"]
    assert svc.install[-1] == ["launchctl", "bootstrap", "gui/501", str(svc.file)]


def test_linux_systemd_unit(tmp_path):
    data = Path("/home/me/my data")  # built on this OS: backslashes when the tests run on Windows
    svc = service.describe("linux", ("/home/me/.local/bin/openmemore",), data, tmp_path)
    assert svc.file == tmp_path / ".config" / "systemd" / "user" / "openmemore.service"
    assert "ExecStart=" + shlex.join(["/home/me/.local/bin/openmemore", "daemon", "run", "--home", str(data)]) in svc.content
    assert "WantedBy=default.target" in svc.content
    assert svc.install[-1] == ["systemctl", "--user", "enable", "--now", "openmemore.service"]
    assert any("enable-linger" in n for n in svc.notes)


def test_windows_logon_task_uses_pythonw_when_there_is_one(tmp_path):
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    (scripts / "pythonw.exe").write_text("")
    svc = service.describe("win32", (str(scripts / "openmemore.exe"),), tmp_path / "data", tmp_path)
    create = svc.install[0]
    task = create[create.index("/TR") + 1]
    assert "pythonw.exe" in task and "-m openmemore daemon run" in task and "/SC" in create and "ONLOGON" in create
    without = service.describe("win32", (str(tmp_path / "openmemore.exe"),), tmp_path / "data", tmp_path)
    assert "openmemore.exe daemon run" in without.install[0][without.install[0].index("/TR") + 1]


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


def test_telemetry_prints_the_standard_settings(home):
    code, out = cli("daemon", "telemetry")
    token = config.load_config(home / "config.toml").daemon.token
    assert code == 0
    assert "OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:47318" in out
    assert f"OTEL_EXPORTER_OTLP_HEADERS=x-openmemore-token={token}" in out
    assert "gen_ai.input.messages" in out and "claude" not in out.lower()


def test_install_dry_run_changes_nothing(home, isolated_user):
    code, out = cli("daemon", "install", "--dry-run")
    assert code == 0 and "dry run" in out and "--home" in out and "--log-file" in out
    assert not any(isolated_user.rglob("*.plist")) and not any(isolated_user.rglob("openmemore.service"))


# --- the real process --------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_the_daemon_process_receives_and_captures(home, isolated_user):
    """`python -m openmemore daemon run`, as the services start it: it answers health checks,
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
    proc = subprocess.Popen([sys.executable, "-m", "openmemore", "daemon", "run", "--home", str(home),
                             "--log-file", str(log)], env=env)
    pid = None
    try:
        answer = daemon.wait_until_healthy(port, 30)
        assert answer, log.read_text() if log.exists() else "no log"
        # Not proc.pid: on Windows the venv's python.exe is a launcher that starts the real one.
        pid = answer["pid"]
        assert daemon.read_state(home)["pid"] == pid

        import urllib.request

        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/traces",
            data=json.dumps(traces_json(call([text("user", "Hi from OTel.")], "Hello."))).encode(),
            headers={"content-type": "application/json", daemon.TOKEN_HEADER: token}, method="POST")
        with urllib.request.urlopen(request, timeout=10) as r:
            assert r.status == 200
        write_transcript(isolated_user, "s2", "Hi from a transcript.")
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            with closing(db.connect(home / "openmemore.db")) as conn:
                titles = {c.title for c in repo.list_conversations(conn, scopes=None)}
            if titles == {"Hi from OTel.", "Hi from a transcript."}:
                break
            time.sleep(0.2)
        assert titles == {"Hi from OTel.", "Hi from a transcript."}
        assert "listening on" in log.read_text()
    finally:
        proc.terminate()
        proc.wait(timeout=15)
        if pid is not None and pid != proc.pid:
            try:
                os.kill(pid, signal.SIGTERM)  # the launcher's child, which terminate() doesn't reach
            except OSError:
                pass
