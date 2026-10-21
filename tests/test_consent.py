"""OpenMemore asks before running in the background, in a native dialog when
there's a desktop session, else in the terminal; the default is no. The dialogs are
faked here: tests never show one."""

import subprocess

import pytest
from typer.testing import CliRunner

from conftest import plain
from openmemore import config, consent, db, service
from openmemore.cli import _offer_daemon, app, initialize

real_ask = consent.ask
"""The real function, captured at import, before conftest's autouse fixture replaces it."""


class Dialog:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.result = (returncode, stdout, stderr)
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        code, out, err = self.result
        return subprocess.CompletedProcess(args, code, stdout=out, stderr=err)


def asker(platform, dialog=None, env=None, tty=False, reply="", tools=("osascript", "powershell", "zenity")):
    return consent.Asker(
        platform=platform, env=env or {}, which=lambda name: f"/usr/bin/{name}" if name in tools else None,
        run=dialog or Dialog(), isatty=lambda: tty, prompt=lambda _: reply,
    )


def _real_ask(a):
    return real_ask(asker=a)


def test_macos_dialog():
    allow = Dialog(0, "button returned:Allow\n")
    assert _real_ask(asker("darwin", allow)) == "allowed"
    args, _ = allow.calls[0]
    assert args[0] == "osascript" and args[-2:] == [consent.MESSAGE, consent.TITLE]  # text as arguments, not script
    assert consent.MESSAGE not in args[2]
    assert _real_ask(asker("darwin", Dialog(1, stderr="execution error: User canceled. (-128)"))) == "denied"


def test_a_dialog_that_can_t_be_shown_falls_back_to_the_terminal():
    broken = Dialog(1, stderr="No user interaction allowed. (-1713)")
    assert _real_ask(asker("darwin", broken, tty=True, reply="y")) == "allowed"
    assert _real_ask(asker("darwin", broken, tty=True, reply="")) == "denied"  # the default is no
    assert _real_ask(asker("darwin", broken, tty=False)) == "unavailable"


def test_over_ssh_it_asks_in_the_terminal():
    dialog = Dialog(0, "button returned:Allow")
    assert _real_ask(asker("darwin", dialog, env={"SSH_CONNECTION": "1 2 3 4"}, tty=True, reply="no")) == "denied"
    assert dialog.calls == []


def test_windows_message_box():
    yes = Dialog(0, "Yes\r\n")
    assert _real_ask(asker("win32", yes)) == "allowed"
    args, kwargs = yes.calls[0]
    assert kwargs["env"]["OPENMEMORE_CONSENT_TEXT"] == consent.MESSAGE and consent.MESSAGE not in " ".join(args)
    assert _real_ask(asker("win32", Dialog(0, "No"))) == "denied"


def test_linux_needs_a_display():
    assert _real_ask(asker("linux", Dialog(0), env={"DISPLAY": ":0"})) == "allowed"
    assert _real_ask(asker("linux", Dialog(1), env={"WAYLAND_DISPLAY": "wayland-0"})) == "denied"
    no_display = Dialog(0)
    assert _real_ask(asker("linux", no_display, tty=True, reply="yes")) == "allowed" and no_display.calls == []
    kde = Dialog(0)
    _real_ask(asker("linux", kde, env={"DISPLAY": ":0"}, tools=("kdialog",)))
    assert kde.calls[0][0][0] == "kdialog"


# --- the install flow ----------------------------------------------------------------------


@pytest.fixture
def installs(monkeypatch):
    """Record service installs instead of running launchctl, systemctl, or schtasks."""
    from openmemore import daemon

    done = []
    monkeypatch.setattr(service, "install", lambda svc, run=None: done.append(svc))
    monkeypatch.setattr(daemon, "wait_until_healthy", lambda port, seconds=10.0: {"pid": 4242})
    return done


def cli(*args):
    result = CliRunner().invoke(app, list(args), env={"COLUMNS": "200"})
    return result.exit_code, " ".join(plain(result.output).split())


def audit_rows(home):
    return [tuple(r) for r in db.connect(home / "openmemore.db").execute(
        "SELECT tool, summary FROM audit_log WHERE tool = 'daemon install'")]


def test_nothing_is_installed_without_permission(home, installs):
    code, out = cli("daemon", "install")
    assert code == 1 and installs == [] and "Not running in the background" in out
    assert audit_rows(home) == []


def test_installing_once_allowed(home, installs, consent_answers):
    consent_answers.append("allowed")
    code, out = cli("daemon", "install")
    assert code == 0 and len(installs) == 1 and "Running in the background (pid 4242" in out
    assert audit_rows(home) == [("daemon install", "allowed when asked")]


def test_yes_is_permission_given_up_front(home, installs, consent_answers):
    consent_answers.append("denied")  # not asked at all
    code, _ = cli("daemon", "install", "--yes")
    assert code == 0 and len(installs) == 1 and consent_answers == ["denied"]
    assert audit_rows(home) == [("daemon install", "allowed with --yes")]


def test_when_there_is_no_way_to_ask(home, installs, consent_answers):
    consent_answers.append("unavailable")
    code, out = cli("daemon", "install")
    assert code == 1 and installs == [] and "openmemore daemon install --yes" in out


def test_dry_run_doesn_t_ask(home, installs, consent_answers):
    consent_answers.append("allowed")
    code, out = cli("daemon", "install", "--dry-run")
    assert code == 0 and installs == [] and consent_answers == ["allowed"] and "after asking you" in out


def free_port():
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_setup_offers_it_only_when_there_is_something_to_capture(home, installs, consent_answers):
    initialize(home)
    consent_answers.append("allowed")
    _offer_daemon()
    assert installs == [] and consent_answers == ["allowed"]  # no capture source: not offered
    path = home / "config.toml"
    cfg = config.load_config(path)
    cfg.capture.sources = ("vscode-copilot",)
    cfg.daemon.port = free_port()  # not the default: a real daemon may be running on this machine
    config.save_config(path, cfg)
    _offer_daemon()
    assert len(installs) == 1 and consent_answers == []
