"""OpenMCP as a library (SPEC.md M7.5): the user's own agents and programs share memory
with their assistants, in-process, with no server.

    from openmcp import Memory

    memory = Memory("trip-planner")

    @memory.recall(query="request")
    @memory.remember(question="request")
    def plan(request: str, context: str = "") -> str: ...

It calls the same tool functions as the MCP server (openmcp.tools), as a client named
like any other, so scopes, the secret filter, wrapping, embeddings, and the audit log
all apply. Experimental until the project's final name (SPEC.md M9).
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import json
import logging
import sqlite3
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from mcp.server.mcpserver.exceptions import ToolError

from openmcp import audit, config, db
from openmcp.clients import client_name
from openmcp.models import Kind, MemoryInput, Profile, RecentConversations
from openmcp.safety.wrap import NOTICE, unwrap
from openmcp.tools import Caller

log = logging.getLogger("openmcp")
APP = "openmcp library"
"""What the audit log records as the application for library calls."""

F = TypeVar("F", bound=Callable[..., Any])


class OpenMCPError(Exception):
    """A library call couldn't be done (not found, not allowed in this client's scopes,
    a secret, ...); the message says why."""


@dataclass(frozen=True)
class Hit:
    """A search result."""

    type: str
    """"memory" or "chat"."""
    id: int
    text: str
    """The stored text, for program logic."""
    wrapped: str
    """The same text labelled as stored data (SPEC.md Section 8), for putting in a prompt."""
    source: str
    score: float
    kind: str | None = None
    subject: str | None = None
    project: str | None = None
    conversation_id: int | None = None
    title: str | None = None
    created_at: datetime | None = None


_conversation: contextvars.ContextVar[str | None] = contextvars.ContextVar("openmcp_conversation", default=None)


class Memory:
    """The user's shared memory, as the client `name` (scopes from [clients.<name>] in
    config.toml, like an app's MCP clientInfo name). `home` overrides the data directory."""

    def __init__(self, name: str, *, home: str | Path | None = None) -> None:
        self.client = client_name(name)
        self.home = Path(home).expanduser() if home is not None else None
        self._lock = threading.Lock()
        self._ready = False
        self._conversations: dict[str, int] = {}

    def __repr__(self) -> str:
        return f"Memory({self.client!r})"

    # --- plumbing -------------------------------------------------------------------

    def _base(self) -> Path:
        return self.home or config.home()

    def _caller(self) -> Caller:
        cfg = config.load_config(config.config_path(self._base()))
        return Caller(client=self.client, scopes=config.resolve_client(cfg, self.client))

    @contextmanager
    def _call(self, tool: str) -> Iterator[tuple[sqlite3.Connection, audit.AuditCall, Caller]]:
        """A short-lived connection per call (as the server does: safe across threads),
        audited, with tool errors as OpenMCPError."""
        base = self._base()
        with self._lock:
            if not self._ready:
                from openmcp.cli import initialize

                initialize(base)
                self._ready = True
        caller = self._caller()
        try:
            with closing(db.connect(config.db_path(base))) as conn, \
                    audit.audited(conn, caller.client, tool, APP) as call:
                yield conn, call, caller
        except ToolError as e:
            raise OpenMCPError(str(e)) from None

    # --- reading --------------------------------------------------------------------

    def search(self, query: str, *, limit: int = 10, kinds: Sequence[Kind] = (), project: str | None = None,
               chats: bool = True) -> list[Hit]:
        """Memories and conversations related to `query`, by meaning and by words."""
        from openmcp.tools.search import search_context

        with self._call("search_context") as (conn, call, caller):
            call.summary = f"q={query[:60]!r}"
            result = search_context(conn, caller, query, kinds, project, chats, limit)
            call.result_count = len(result.results)
        return [
            Hit(type=h.type, id=h.id, text=unwrap(h.content), wrapped=h.content, source=h.source, score=h.score,
                kind=h.kind, subject=h.subject, project=h.project, conversation_id=h.conversation_id, title=h.title,
                created_at=h.created_at)
            for h in result.results
        ]

    def profile(self, project: str | None = None) -> Profile:
        """Preferences, facts, active projects, and recent decisions (wrapped)."""
        from openmcp.tools.profile import get_profile

        with self._call("get_profile") as (conn, call, caller):
            result = get_profile(conn, caller, project)
            call.result_count = sum(len(g) for g in (result.preferences, result.facts, result.active_projects,
                                                     result.recent_decisions))
        return result

    def recent(self, limit: int = 5, source: str | None = None) -> RecentConversations:
        from openmcp.tools.conversations import recent_conversations

        with self._call("recent_conversations") as (conn, call, caller):
            result = recent_conversations(conn, caller, source, None, limit)
            call.result_count = len(result.conversations)
        return result

    def context(self, query: str, *, limit: int = 5, kinds: Sequence[Kind] = (), profile: bool = False) -> str:
        """A block to put in a prompt: the stored-data notice, then related items, each
        wrapped. Empty if nothing is related."""
        items = [h.wrapped for h in self.search(query, limit=limit, kinds=kinds)]
        if profile:
            p = self.profile()
            items = [i.content for g in (p.preferences, p.facts, p.active_projects, p.recent_decisions) for i in g] + items
        return "\n".join([NOTICE, *items]) if items else ""

    # --- writing --------------------------------------------------------------------

    def save(self, content: str, *, kind: Kind = "note", subject: str | None = None,
             project: str | None = None) -> int | None:
        """Save a memory in this client's write scope. Returns its id, or None if the
        same statement is already stored."""
        from openmcp.tools.memories import save_memories

        with self._call("save_memories") as (conn, call, caller):
            result = save_memories(conn, caller, [MemoryInput(kind=kind, content=content, subject=subject,
                                                              project=project)])
            call.summary, call.result_count = "1 items", len(result.saved)
        if result.rejected:
            raise OpenMCPError(f"not saved: {result.rejected[0].reason}")
        return result.saved[0] if result.saved else None

    def update(self, memory_id: int, content: str) -> int:
        """Correct a memory; returns the new version's id (the old one is kept as history)."""
        from openmcp.tools.memories import update_memory

        with self._call("update_memory") as (conn, call, caller):
            call.summary = f"id={memory_id}"
            return update_memory(conn, caller, memory_id, content).new_id

    def forget(self, memory_id: int) -> int:
        """Delete a memory and its history; returns how many versions were deleted."""
        from openmcp.tools.memories import forget

        with self._call("forget") as (conn, call, caller):
            call.summary = f"id={memory_id}"
            return forget(conn, caller, memory_id).deleted

    def record(self, question: str, answer: str, *, conversation: str | None = None,
               title: str | None = None) -> int | None:
        """Store an exchange (secrets redacted). Exchanges with the same `conversation`
        key, in this process, go into one conversation. Returns its id."""
        from openmcp.tools.conversations import record_exchange

        key = conversation or _conversation.get() or "default"
        with self._call("record_exchange") as (conn, call, caller):
            result = record_exchange(conn, caller, question, answer, self._conversations.get(key), title)
            call.summary = f"conversation={result.conversation_id}"
            call.result_count = len(result.message_ids)
        if result.conversation_id is not None:
            self._conversations[key] = result.conversation_id
        return result.conversation_id

    @contextmanager
    def conversation(self, key: str) -> Iterator[None]:
        """Group everything recorded inside (by any function) into one conversation."""
        token = _conversation.set(key)
        try:
            yield
        finally:
            _conversation.reset(token)

    # --- decorators -----------------------------------------------------------------

    def remember(self, func: F | None = None, *, question: str | None = None,
                 conversation: str | None = None) -> Any:
        """Store each call of the decorated function as an exchange: the question (the
        argument named `question`, else the first string argument, else all of them)
        and the result. A failure to store is logged, never raised."""
        def decorate(fn: F) -> F:
            signature = inspect.signature(fn)
            if question is not None and question not in signature.parameters:
                raise TypeError(f"{fn.__qualname__} has no argument {question!r}")
            group = conversation or f"{fn.__module__}.{fn.__qualname__}"

            def store(args: tuple, kwargs: dict, result: Any) -> None:
                try:
                    asked = _question(signature, args, kwargs, question)
                    self.record(asked, _text(result), conversation=_conversation.get() or group)
                except Exception:  # noqa: BLE001 - memory must never break the user's function
                    log.warning("openmcp: couldn't record a call of %s", fn.__qualname__, exc_info=True)

            if inspect.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    result = await fn(*args, **kwargs)
                    ctx = contextvars.copy_context()
                    await asyncio.to_thread(ctx.run, store, args, kwargs, result)
                    return result
                return async_wrapper  # type: ignore[return-value]

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                result = fn(*args, **kwargs)
                store(args, kwargs, result)
                return result
            return wrapper  # type: ignore[return-value]

        return decorate(func) if func is not None else decorate

    def recall(self, func: F | None = None, *, query: str | None = None, into: str = "context", limit: int = 5,
               kinds: Sequence[Kind] = (), profile: bool = False) -> Any:
        """Before each call, pass `memory.context(...)` for the argument named `query`
        (else the first string argument) into the argument named `into`. A failure to
        read is logged, and the function gets an empty context."""
        def decorate(fn: F) -> F:
            signature = inspect.signature(fn)
            accepts = into in signature.parameters or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
            if not accepts:
                raise TypeError(f"{fn.__qualname__} needs an argument {into!r} for the recalled context")
            if query is not None and query not in signature.parameters:
                raise TypeError(f"{fn.__qualname__} has no argument {query!r}")

            def fetch(args: tuple, kwargs: dict) -> str:
                try:
                    return self.context(_question(signature, args, kwargs, query), limit=limit, kinds=kinds,
                                        profile=profile)
                except Exception:  # noqa: BLE001 - memory must never break the user's function
                    log.warning("openmcp: couldn't recall context for %s", fn.__qualname__, exc_info=True)
                    return ""

            if inspect.iscoroutinefunction(fn):
                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    kwargs[into] = await asyncio.to_thread(fetch, args, kwargs)
                    return await fn(*args, **kwargs)
                return async_wrapper  # type: ignore[return-value]

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                kwargs[into] = fetch(args, kwargs)
                return fn(*args, **kwargs)
            return wrapper  # type: ignore[return-value]

        return decorate(func) if func is not None else decorate


def _question(signature: inspect.Signature, args: tuple, kwargs: dict, name: str | None) -> str:
    bound = signature.bind_partial(*args, **kwargs)
    values = dict(bound.arguments)
    if name is not None:
        return _text(values.get(name, ""))
    first = next((v for v in values.values() if isinstance(v, str) and v.strip()), None)
    if first is not None:
        return first
    return ", ".join(f"{k}={_text(v)}" for k, v in values.items() if k not in ("self", "cls"))


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
