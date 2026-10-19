"""`openmcp review` (SPEC.md M5): a local page to approve, edit, and reject proposed
memories, and to browse, edit, and delete memories.

It listens on 127.0.0.1 only. A random token, generated for each run and given in the
URL the CLI prints, is exchanged for an HttpOnly, SameSite=Strict cookie; every other
request needs that cookie. Requests naming another Host are refused (DNS rebinding),
and requests that change something must carry the X-OpenMCP header, which a page on
another origin can't send without a CORS preflight this server never grants. Stored
text reaches the page as JSON and is shown with textContent, never as HTML.
"""

from __future__ import annotations

import hmac
import secrets
import socket
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from starlette.routing import Route

from openmcp import audit, config, db, embed, proposals, repo
from openmcp.config import ClientScopes
from openmcp.models import KINDS, Label, Memory, MemoryText, Proposal
from openmcp.tools import Caller
from openmcp.tools import memories as memory_tools

CLIENT = "review"
"""Audit-log client name for what the user does on the review page."""
COOKIE = "openmcp_review"
CHANGE_HEADER = "x-openmcp"
MAX_MEMORIES = 2000


class MemoryEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: MemoryText
    subject: Label | None = None


class Resolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolution: Literal["keep_from", "keep_to", "both"]


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ids: list[int]
    decision: str


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def page() -> str:
    return resources.files("openmcp").joinpath("review.html").read_text(encoding="utf-8")


def _memory(m: Memory) -> dict[str, Any]:
    return m.model_dump(mode="json", exclude={"content_hash"})


class _Guard(BaseHTTPMiddleware):
    def __init__(self, app, token: str, hosts: set[str]) -> None:
        super().__init__(app)
        self.token = token
        self.hosts = hosts

    async def dispatch(self, request: Request, call_next):
        if request.headers.get("host") not in self.hosts:
            return PlainTextResponse("Unknown host.", status_code=403)
        given = request.query_params.get("token")
        if request.url.path == "/" and given is not None:
            if not hmac.compare_digest(given, self.token):
                return PlainTextResponse("This link has expired. Run `openmcp review` again.", status_code=403)
            response = RedirectResponse("/", status_code=303)  # keep the token out of the address bar
            response.set_cookie(COOKIE, self.token, httponly=True, samesite="strict", path="/")
            return response
        if not hmac.compare_digest(request.cookies.get(COOKIE, ""), self.token):
            return PlainTextResponse("Open the link `openmcp review` printed.", status_code=403)
        if request.method not in ("GET", "HEAD") and request.headers.get(CHANGE_HEADER) != "1":
            return PlainTextResponse("Missing X-OpenMCP header.", status_code=403)
        return await call_next(request)


def build_app(base: Path, token: str, port: int) -> Starlette:
    path = config.db_path(base)
    cfg = config.load_config(config.config_path(base))
    owner = Caller(client=CLIENT, scopes=ClientScopes(read_scopes=None, write_scope=cfg.defaults.write_scope))

    @contextmanager
    def connection() -> Iterator[sqlite3.Connection]:
        with closing(db.connect(path)) as conn:
            yield conn

    def error(message: str, status: int = 400) -> JSONResponse:
        return JSONResponse({"error": message}, status_code=status)

    async def body(request: Request, model: type[BaseModel]) -> BaseModel:
        try:
            return model.model_validate(await request.json())
        except ValueError as e:  # bad JSON or ValidationError
            raise _BadRequest(e.errors()[0]["msg"] if isinstance(e, ValidationError) else "expected JSON") from None

    def proposal_view(conn: sqlite3.Connection, p: Proposal) -> dict[str, Any]:
        view = p.model_dump(mode="json", exclude={"content_hash"})
        view["conversations"] = [
            {"id": c.id, "title": c.title, "source": c.provider}
            for cid in p.conversation_ids if (c := repo.get_conversation(conn, cid))
        ]
        view["merging"] = [_memory(m) for mid in p.memory_ids if (m := repo.get_memory(conn, mid))]
        similar = repo.get_memory(conn, p.similar_memory_id) if p.similar_memory_id else None
        view["similar"] = _memory(similar) if similar and similar.active else None
        replaces = repo.get_memory(conn, p.replaces_memory_id) if p.replaces_memory_id else None
        view["replaces"] = _memory(replaces) if replaces else None
        disputed = repo.get_memory(conn, p.contradicts_memory_id) if p.contradicts_memory_id else None
        view["contradicts"] = _memory(disputed) if disputed else None
        return view

    def index(request: Request) -> Response:
        return HTMLResponse(page(), headers={
            "Content-Security-Policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
            "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
            "Referrer-Policy": "no-referrer",
        })

    def state(request: Request) -> Response:
        with connection() as conn:
            counts = repo.store_counts(conn)
            labels = conn.execute(
                "SELECT DISTINCT scope FROM memories UNION SELECT DISTINCT scope FROM conversations ORDER BY 1"
            ).fetchall()
            projects = repo.project_labels(conn, None)
        with connection() as conn:
            conflicts = repo.count_open_contradictions(conn)
        return JSONResponse({
            "pending_proposals": counts["pending_proposals"],
            "conflicts": conflicts,
            "due_for_consolidation": counts["due_for_consolidation"],
            "memories": counts["memories"],
            "scopes": [r[0] for r in labels],
            "kinds": list(KINDS),
            "projects": projects,
            "semantic_search": embed.status(),
        })

    def list_proposals(request: Request) -> Response:
        status = request.query_params.get("status", "pending")
        if status not in ("pending", "approved", "rejected"):
            return error("status must be pending, approved, or rejected")
        with connection() as conn:
            found = repo.list_proposals(conn, status=status)
            if status != "pending":
                found = found[::-1][:200]  # decided ones: the latest first
            return JSONResponse([proposal_view(conn, p) for p in found])

    async def approve(request: Request) -> Response:
        edits = await body(request, proposals.Approval)
        proposal_id = int(request.path_params["id"])
        with connection() as conn, audit.audited(conn, CLIENT, "approve") as call:
            call.summary = f"proposal={proposal_id}"
            memory_id = proposals.approve(conn, proposal_id, edits)
            call.summary += f", memory={memory_id}"
            call.result_count = 1
        return JSONResponse({"memory_id": memory_id})

    def reject(request: Request) -> Response:
        proposal_id = int(request.path_params["id"])
        with connection() as conn, audit.audited(conn, CLIENT, "reject") as call:
            call.summary = f"proposal={proposal_id}"
            proposals.reject(conn, proposal_id)
            call.result_count = 1
        return JSONResponse({"ok": True})

    async def decide_all(request: Request) -> Response:
        d = await body(request, Decision)
        if d.decision not in ("approve", "reject"):
            return error("decision must be approve or reject")
        done, failed = 0, []
        with connection() as conn, audit.audited(conn, CLIENT, f"{d.decision} all") as call:
            for proposal_id in d.ids:
                try:
                    if d.decision == "approve":
                        proposals.approve(conn, proposal_id)
                    else:
                        proposals.reject(conn, proposal_id)
                    done += 1
                except proposals.ProposalError as e:
                    failed.append({"id": proposal_id, "error": str(e)})
            call.summary = f"{done} of {len(d.ids)} proposals"
            call.result_count = done
        return JSONResponse({"done": done, "failed": failed})

    def conflicts(request: Request) -> Response:
        from openmcp import relations

        with connection() as conn:
            items = []
            for link in relations.open_contradictions(conn):
                a, b = repo.get_memory(conn, link.from_memory), repo.get_memory(conn, link.to_memory)
                if a and b:
                    items.append({"id": link.id, "source": link.source, "created_at": repo.to_iso(link.created_at),
                                  "from": _memory(a), "to": _memory(b)})
        return JSONResponse(items)

    async def resolve_conflict(request: Request) -> Response:
        from datetime import UTC, datetime

        from openmcp import relations

        r = await body(request, Resolution)
        link_id = int(request.path_params["id"])
        with connection() as conn, audit.audited(conn, CLIENT, "resolve conflict") as call:
            call.summary = f"conflict={link_id}, {r.resolution}"
            try:
                with db.transaction(conn):
                    relations.resolve(conn, link_id, r.resolution, datetime.now(UTC), source=CLIENT)
            except relations.ResolveError as e:
                return error(str(e), 404 if "not found" in str(e) else 400)
            call.result_count = 1
        return JSONResponse({"ok": True})

    def duplicates(request: Request) -> Response:
        with connection() as conn, audit.audited(conn, CLIENT, "find duplicates") as call:
            proposed = proposals.find_duplicates(conn)
            call.result_count = proposed
        return JSONResponse({"proposed": proposed})

    def list_memories(request: Request) -> Response:
        q = request.query_params
        kinds = [k for k in q.getlist("kind") if k in KINDS]
        with connection() as conn:
            found = repo.list_memories(
                conn, scopes=q.getlist("scope") or None, kinds=kinds, project=q.get("project") or None,
                limit=MAX_MEMORIES,
            )
        if text := q.get("q", "").casefold().strip():
            found = [
                m for m in found
                if text in m.content.casefold() or text in (m.subject or "").casefold()
                or text in (m.project or "").casefold()
            ]
        return JSONResponse([_memory(m) for m in found])

    def history(request: Request) -> Response:
        with connection() as conn:
            chain = repo.memory_chain(conn, int(request.path_params["id"]))
        if not chain:
            return error("memory not found", 404)
        return JSONResponse([_memory(m) for m in chain])

    async def edit_memory(request: Request) -> Response:
        e = await body(request, MemoryEdit)
        memory_id = int(request.path_params["id"])
        with connection() as conn, audit.audited(conn, CLIENT, "update_memory") as call:
            call.summary = f"id={memory_id}"
            result = memory_tools.update_memory(conn, owner, memory_id, e.content, e.subject)
            call.result_count = 1
        return JSONResponse(result.model_dump())

    def delete_memory(request: Request) -> Response:
        memory_id = int(request.path_params["id"])
        with connection() as conn, audit.audited(conn, CLIENT, "forget") as call:
            call.summary = f"id={memory_id}"
            result = memory_tools.forget(conn, owner, memory_id)
            call.result_count = result.deleted
        return JSONResponse(result.model_dump())

    def conversation(request: Request) -> Response:
        conversation_id = int(request.path_params["id"])
        with connection() as conn:
            c = repo.get_conversation(conn, conversation_id)
            if c is None:
                return error("conversation not found", 404)
            messages = repo.conversation_messages(conn, c.id)
        return JSONResponse({
            "id": c.id, "title": c.title, "source": c.provider, "scope": c.scope,
            "messages": [m.model_dump(mode="json", include={"id", "role", "content", "created_at", "recall"})
                         for m in messages],
        })

    async def handle_errors(request: Request, exc: Exception) -> Response:
        if isinstance(exc, _BadRequest | proposals.ProposalError):
            return error(str(exc))
        if isinstance(exc, ToolError):
            return error(str(exc), 404 if "not found" in str(exc) else 400)
        raise exc

    routes = [
        Route("/", index),
        Route("/api/state", state),
        Route("/api/proposals", list_proposals),
        Route("/api/proposals/decide", decide_all, methods=["POST"]),
        Route("/api/proposals/{id:int}/approve", approve, methods=["POST"]),
        Route("/api/proposals/{id:int}/reject", reject, methods=["POST"]),
        Route("/api/duplicates", duplicates, methods=["POST"]),
        Route("/api/conflicts", conflicts),
        Route("/api/conflicts/{id:int}/resolve", resolve_conflict, methods=["POST"]),
        Route("/api/memories", list_memories),
        Route("/api/memories/{id:int}/history", history),
        Route("/api/memories/{id:int}", edit_memory, methods=["POST"]),
        Route("/api/memories/{id:int}", delete_memory, methods=["DELETE"]),
        Route("/api/conversations/{id:int}", conversation),
    ]
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    return Starlette(
        routes=routes,
        middleware=[Middleware(_Guard, token=token, hosts=hosts)],
        exception_handlers={_BadRequest: handle_errors, proposals.ProposalError: handle_errors, ToolError: handle_errors},
    )


class _BadRequest(Exception):
    pass


def new_token() -> str:
    return secrets.token_urlsafe(24)
