"""`openmcp daemon` (SPEC.md M6): capture in the background, without waiting for a
client to read.

One process, normally run by the OS as a per-user service (openmcp.service):
- every few seconds it captures changed transcripts of the enabled sources and embeds
  what's new, so another assistant finds a conversation moments after it happens;
- it receives OpenTelemetry log exports (OTLP/HTTP with JSON) on 127.0.0.1, for clients
  that export their conversations (openmcp.capture.otel).

The receiver accepts only requests to its own Host, with the token from config.toml
in an x-openmcp-token header, and with Content-Type application/json: a web page can't
send that to another origin without a CORS preflight, which this server never grants,
so a site the user visits can't write into their conversations.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import sqlite3
import sys
import threading
import time
import urllib.request
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from openmcp import __version__, audit, config, db, embed, repo

CLIENT = "daemon"
"""Audit-log client name for what the daemon captures."""
TOKEN_HEADER = "x-openmcp-token"
STATE_FILE = "daemon.json"
MAX_BODY = 16 * 1024 * 1024
EMBED_BUDGET_MS = 2000
"""Embedding time per tick, so a large backlog doesn't hold up capture."""
HEALTH_TIMEOUT_S = 0.5


def ensure_token(base: Path) -> str:
    """The receiver's token, created (and saved to config.toml) on first use."""
    path = config.config_path(base)
    cfg = config.load_config(path)
    if cfg.daemon.token:
        return cfg.daemon.token
    cfg.daemon.token = secrets.token_urlsafe(24)
    config.save_config(path, cfg)
    return cfg.daemon.token


def endpoint(cfg: config.Config) -> str:
    return f"http://127.0.0.1:{cfg.daemon.port}"


def build_app(base: Path, token: str, port: int) -> Starlette:
    path = config.db_path(base)
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    def refuse(request: Request) -> Response | None:
        if request.headers.get("host") not in hosts:
            return PlainTextResponse("Unknown host.", status_code=403)
        return None

    def health(request: Request) -> Response:
        return refuse(request) or JSONResponse({"ok": True, "version": __version__, "pid": os.getpid()})

    async def logs(request: Request) -> Response:
        if refused := refuse(request):
            return refused
        if not hmac.compare_digest(request.headers.get(TOKEN_HEADER, ""), token):
            return PlainTextResponse(f"Missing or wrong {TOKEN_HEADER} header.", status_code=401)
        if not request.headers.get("content-type", "").startswith("application/json"):
            return PlainTextResponse(
                "Only OTLP/HTTP with JSON is accepted: set OTEL_EXPORTER_OTLP_PROTOCOL=http/json.", status_code=415
            )
        body = await request.body()
        if len(body) > MAX_BODY:
            return PlainTextResponse("Request too large.", status_code=413)
        try:
            payload = json.loads(body)
        except ValueError:
            return PlainTextResponse("Expected JSON.", status_code=400)
        from openmcp.capture import otel

        with closing(db.connect(path)) as conn:
            cfg = config.load_config(config.config_path(base))
            report = otel.ingest(conn, cfg, payload)
            if report.messages or report.skipped:
                _audit(conn, "otel", f"{report.messages} messages", report.messages, report.skipped)
        for reason in sorted(set(report.skipped)):
            print(f"openmcp daemon: skipped {reason}", file=sys.stderr)
        return JSONResponse({"partialSuccess": {}})

    return Starlette(routes=[Route("/health", health), Route("/v1/logs", logs, methods=["POST"])])


def _audit(conn: sqlite3.Connection, tool: str, summary: str, count: int, notes: list[str] = ()) -> None:
    with audit.audited(conn, CLIENT, tool) as call:
        call.summary = summary + (f"; skipped {len(notes)}" if notes else "")
        call.result_count = count


@dataclass
class TickReport:
    messages: int = 0
    embedded: int = 0
    errors: tuple[str, ...] = ()


def tick(base: Path) -> TickReport:
    """One round: capture changed transcripts of the enabled sources, then embed what's
    pending. Config is read each time, so `openmcp setup` takes effect without a restart."""
    cfg = config.load_config(config.config_path(base))
    report = TickReport()
    with closing(db.connect(config.db_path(base))) as conn:
        if cfg.capture.sources:
            from openmcp.capture import capture

            captured = capture(conn, cfg)
            report.messages, report.errors = captured.messages, tuple(captured.errors)
            if captured.messages:
                _audit(conn, "capture", f"{captured.messages} messages from {captured.conversations} conversations",
                       captured.messages)
        embedder = embed.get_embedder()
        if embedder is not None and repo.count_pending_embeddings(conn):
            done = embed.embed_pending(conn, embedder, budget_ms=EMBED_BUDGET_MS)
            report.embedded = done.memories + done.chunks
    return report


# --- running -------------------------------------------------------------------------


def state_path(base: Path) -> Path:
    return base / STATE_FILE


def read_state(base: Path) -> dict[str, Any] | None:
    try:
        return json.loads(state_path(base).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def healthy(port: int) -> dict[str, Any] | None:
    """The running daemon's /health answer, or None if nothing answers on `port`."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=HEALTH_TIMEOUT_S) as r:
            return json.loads(r.read())
    except (OSError, ValueError):
        return None


@contextmanager
def _state(base: Path, port: int) -> Iterator[None]:
    path = state_path(base)
    path.write_text(json.dumps({
        "pid": os.getpid(), "port": port, "version": __version__,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }), encoding="utf-8")
    try:
        yield
    finally:
        try:
            if read_state(base) and read_state(base).get("pid") == os.getpid():
                path.unlink()
        except OSError:
            pass


def run(base: Path, log: Any = None) -> None:
    """Run until stopped: the capture loop in a thread, the receiver in this one."""
    import uvicorn

    log = log or sys.stderr
    token = ensure_token(base)
    cfg = config.load_config(config.config_path(base))
    port = cfg.daemon.port
    if healthy(port):
        raise RuntimeError(f"an openmcp daemon is already running on port {port}")
    server = uvicorn.Server(uvicorn.Config(build_app(base, token, port), host="127.0.0.1", port=port,
                                           log_level="warning"))
    stop = threading.Event()

    def loop() -> None:
        while not stop.is_set():
            try:
                report = tick(base)
                for error in report.errors:
                    print(f"openmcp daemon: capture: {error}", file=log, flush=True)
            except Exception as e:  # noqa: BLE001 - one bad round mustn't stop the daemon
                print(f"openmcp daemon: {type(e).__name__}: {e}", file=log, flush=True)
            interval = config.load_config(config.config_path(base)).daemon.interval
            stop.wait(interval)

    print(f"openmcp daemon {__version__}: listening on {endpoint(cfg)}, data in {base}", file=log, flush=True)
    with _state(base, port):
        worker = threading.Thread(target=loop, name="openmcp-capture", daemon=True)
        worker.start()
        try:
            server.run()
        finally:
            stop.set()
            worker.join(timeout=5)
    if not server.started:
        raise RuntimeError(f"couldn't listen on 127.0.0.1:{port}; is it in use? Change [daemon] port in config.toml")


def wait_until_healthy(port: int, seconds: float = 10.0) -> dict[str, Any] | None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if answer := healthy(port):
            return answer
        time.sleep(0.2)
    return None
