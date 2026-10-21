"""Asking the user before OpenMemore runs in the background (SPEC.md M6).

No OS asks for a per-user background task the way it does for, say, the camera: macOS
shows a "Background Items Added" notice afterwards, Windows and Linux nothing. So
OpenMemore asks itself, before installing anything: in a native dialog when there's a
desktop session (osascript on macOS, a PowerShell message box on Windows, zenity or
kdialog on Linux), else in the terminal. The default is no. Scripts pass --yes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

TITLE = "OpenMemore would like to run in the background"
MESSAGE = (
    "It checks your AI assistants' saved chats every few seconds and receives Claude Code's conversations, "
    "so your other assistants can find them right away. It starts when you log in, listens on 127.0.0.1 "
    "only, and keeps everything on this computer.\n\n"
    "You can turn it off any time with: openmemore daemon uninstall"
)
ALLOW, DENY = "Allow", "Don't Allow"
DIALOG_TIMEOUT_S = 600

Answer = Literal["allowed", "denied", "unavailable"]


@dataclass
class Asker:
    """How to ask, injectable for tests."""

    platform: str = sys.platform
    env: Mapping[str, str] = field(default_factory=lambda: dict(os.environ))
    which: Callable[[str], str | None] = shutil.which
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run
    isatty: Callable[[], bool] = lambda: sys.stdin.isatty()
    prompt: Callable[[str], str] = input


def desktop_session(asker: Asker) -> bool:
    """Whether a dialog would reach the person at the keyboard: not over SSH, and on
    Linux, only with a display."""
    if asker.env.get("SSH_CONNECTION") or asker.env.get("SSH_TTY"):
        return False
    if asker.platform.startswith("linux"):
        return bool(asker.env.get("DISPLAY") or asker.env.get("WAYLAND_DISPLAY"))
    return asker.platform in ("darwin", "win32")


def _dialog_command(asker: Asker, title: str, message: str) -> tuple[list[str], dict[str, str] | None] | None:
    """The command that shows a yes/no dialog here, and any environment it needs."""
    if asker.platform == "darwin" and asker.which("osascript"):
        # The text goes in as arguments, so nothing in it is read as AppleScript.
        script = (
            "on run argv\n"
            f'display dialog (item 1 of argv) with title (item 2 of argv) buttons {{"{DENY}", "{ALLOW}"}} '
            f'default button "{DENY}" cancel button "{DENY}" with icon note\n'
            "end run"
        )
        return ["osascript", "-e", script, message, title], None
    if asker.platform == "win32" and (shell := asker.which("powershell") or asker.which("pwsh")):
        script = (
            "Add-Type -AssemblyName PresentationFramework; "
            "[System.Windows.MessageBox]::Show($env:OPENMEMORE_CONSENT_TEXT, $env:OPENMEMORE_CONSENT_TITLE, 'YesNo', "
            "'Question', 'No')"
        )
        env = {**asker.env, "OPENMEMORE_CONSENT_TEXT": message, "OPENMEMORE_CONSENT_TITLE": title}
        return [shell, "-NoProfile", "-NonInteractive", "-Command", script], env
    if asker.platform.startswith("linux"):
        if asker.which("zenity"):
            return ["zenity", "--question", "--title", title, "--text", message, "--ok-label", ALLOW,
                    "--cancel-label", DENY, "--no-wrap"], None
        if asker.which("kdialog"):
            return ["kdialog", "--title", title, "--yesno", message, "--yes-label", ALLOW, "--no-label", DENY], None
    return None


def _dialog(asker: Asker, title: str, message: str) -> Answer:
    command = _dialog_command(asker, title, message) if desktop_session(asker) else None
    if command is None:
        return "unavailable"
    args, env = command
    try:
        done = asker.run(args, capture_output=True, text=True, timeout=DIALOG_TIMEOUT_S, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    out = (done.stdout or "").strip()
    if asker.platform == "darwin":
        if done.returncode == 0:
            return "allowed" if out.endswith(ALLOW) else "denied"
        # -128 is the user pressing the cancel button; anything else means no dialog could be shown.
        return "denied" if "-128" in (done.stderr or "") else "unavailable"
    if asker.platform == "win32":
        return {"Yes": "allowed", "No": "denied"}.get(out, "unavailable")
    return {0: "allowed", 1: "denied"}.get(done.returncode, "unavailable")  # zenity, kdialog


def _terminal(asker: Asker, title: str, message: str) -> Answer:
    if not asker.isatty():
        return "unavailable"
    print(f"\n{title}.\n{message}")
    try:
        reply = asker.prompt("Allow? [y/N] ")
    except EOFError:
        return "denied"
    return "allowed" if reply.strip().lower() in ("y", "yes") else "denied"


def ask(title: str = TITLE, message: str = MESSAGE, asker: Asker | None = None) -> Answer:
    """Ask in a dialog if there's a desktop, else in the terminal. 'unavailable' means
    there was no way to ask (no desktop and no terminal): treat it as no."""
    asker = asker or Asker()
    answer = _dialog(asker, title, message)
    return answer if answer != "unavailable" else _terminal(asker, title, message)
