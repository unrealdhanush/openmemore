"""Installing `openmcp daemon` as a per-user service (SPEC.md M6), so it starts at login
and restarts if it stops: a launchd agent on macOS, a systemd user unit on Linux, a
logon task on Windows. Only OpenMCP's own service definition is written; nothing of
another application's is touched.
"""

from __future__ import annotations

import os
import plistlib
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

LABEL = "dev.openmcp.daemon"
"""launchd label."""
UNIT = "openmcp.service"
"""systemd user unit."""
TASK = "OpenMCP daemon"
"""Windows scheduled task."""
LOG_FILE = "logs/daemon.log"


class ServiceError(Exception):
    """Why the service couldn't be installed or removed; the message is for the user."""


@dataclass
class Service:
    """What installing (or removing) the service takes on this platform."""

    name: str
    file: Path | None
    """The service definition written on install (none on Windows: the task holds it)."""
    content: str | None
    install: list[list[str]]
    uninstall: list[list[str]]
    notes: list[str] = field(default_factory=list)


def daemon_args(command: tuple[str, ...], base: Path) -> list[str]:
    return [*command, "daemon", "run", "--home", str(base), "--log-file", str(base / LOG_FILE)]


def describe(platform: str, command: tuple[str, ...], base: Path, user_home: Path, uid: int | None = None) -> Service:
    """The service for `platform` ('darwin', 'linux', 'win32'), running `command`."""
    args = daemon_args(command, base)
    if platform == "darwin":
        plist = user_home / "Library" / "LaunchAgents" / f"{LABEL}.plist"
        domain = f"gui/{uid if uid is not None else os.getuid()}"
        content = plistlib.dumps({
            "Label": LABEL,
            "ProgramArguments": args,
            "RunAtLoad": True,
            "KeepAlive": True,
            "ProcessType": "Background",
            "ThrottleInterval": 10,
        }).decode()
        return Service(
            name="launchd agent", file=plist, content=content,
            # Remove an earlier copy first, so reinstalling picks up a new command.
            install=[["launchctl", "bootout", domain, str(plist)], ["launchctl", "bootstrap", domain, str(plist)]],
            uninstall=[["launchctl", "bootout", domain, str(plist)]],
        )
    if platform.startswith("linux"):
        unit = user_home / ".config" / "systemd" / "user" / UNIT
        content = (
            "[Unit]\nDescription=OpenMCP background capture\n\n"
            f"[Service]\nExecStart={shlex.join(args)}\nRestart=on-failure\nRestartSec=10\n\n"
            "[Install]\nWantedBy=default.target\n"
        )
        return Service(
            name="systemd user unit", file=unit, content=content,
            install=[["systemctl", "--user", "daemon-reload"], ["systemctl", "--user", "enable", "--now", UNIT]],
            uninstall=[["systemctl", "--user", "disable", "--now", UNIT]],
            notes=["It starts when you log in. To keep it running while you're logged out too, run: "
                   "loginctl enable-linger"],
        )
    if platform == "win32":
        # pythonw has no console window; it sits next to openmcp.exe in the tool's environment.
        exe = Path(command[0])
        pythonw = exe.with_name("pythonw.exe")
        runner = [str(pythonw), "-m", "openmcp"] if len(command) == 1 and pythonw.exists() else list(command)
        task = subprocess.list2cmdline(daemon_args(tuple(runner), base))
        return Service(
            name="scheduled task", file=None, content=None,
            install=[["schtasks", "/Create", "/TN", TASK, "/SC", "ONLOGON", "/RL", "LIMITED", "/TR", task, "/F"],
                     ["schtasks", "/Run", "/TN", TASK]],
            uninstall=[["schtasks", "/End", "/TN", TASK], ["schtasks", "/Delete", "/TN", TASK, "/F"]],
        )
    raise ServiceError(f"don't know how to install a service on {platform}; run `openmcp daemon run` yourself")


Runner = Callable[..., subprocess.CompletedProcess]


def _run_all(commands: list[list[str]], run: Runner, *, tolerate_first: bool) -> None:
    for i, cmd in enumerate(commands):
        done = run(cmd, capture_output=True, text=True)
        if done.returncode != 0 and not (tolerate_first and i == 0):
            raise ServiceError(f"`{' '.join(cmd)}` failed: {(done.stderr or done.stdout).strip()}")


def install(service: Service, run: Runner = subprocess.run) -> None:
    if service.file is not None:
        service.file.parent.mkdir(parents=True, exist_ok=True)
        service.file.write_text(service.content or "", encoding="utf-8")
    # launchd's first step removes an earlier copy, which fails harmlessly when there isn't one.
    _run_all(service.install, run, tolerate_first=service.name == "launchd agent")


def uninstall(service: Service, run: Runner = subprocess.run) -> bool:
    """Stop and remove the service. Returns whether there was one."""
    existed = service.file is not None and service.file.exists()
    for cmd in service.uninstall:
        done = run(cmd, capture_output=True, text=True)
        existed = existed or done.returncode == 0
    if service.file is not None and service.file.exists():
        service.file.unlink()
        if service.name == "systemd user unit":
            run(["systemctl", "--user", "daemon-reload"], capture_output=True, text=True)
    return existed


def installed(service: Service, run: Runner = subprocess.run) -> bool:
    if service.file is not None:
        return service.file.exists()
    done = run(["schtasks", "/Query", "/TN", TASK], capture_output=True, text=True)
    return done.returncode == 0
