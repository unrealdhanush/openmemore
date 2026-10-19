"""Capture through OpenTelemetry (SPEC.md M6): conversations from any app or library that
records them following OpenTelemetry's GenAI semantic conventions
(open-telemetry/semantic-conventions-genai, status Development), received by
`openmcp daemon` as OTLP logs or traces (openmcp.capture.otlp).

There's nothing app-specific here. A model call ("inference operation") is a span, or a
`gen_ai.client.inference.operation.details` event, carrying:
- `gen_ai.input.messages`: the history sent to the model, oldest first;
- `gen_ai.output.messages`: what the model returned, one message per choice;
- `gen_ai.conversation.id`, when the library has one.
Messages are `{role, parts}`, text being parts of type "text". The app is the standard
resource attribute `service.name`, which names the client (and so its scopes) the same
way an MCP clientInfo name does.

The earlier version of the conventions, which released instrumentation still emits
(opentelemetry-instrumentation-openai-v2 2.3), is read too: one log event per message
(`gen_ai.user.message`, `gen_ai.assistant.message`, ..., and `gen_ai.choice` for the
answer), content in the body, tied to their call by its span id.

Every call re-sends the history, so messages are identified by their role and text:
the same message arriving again changes nothing. Without a conversation id, a
conversation is identified by its opening message. A turn that called one of
OpenMCP's recall tools (a standard `tool_call` part) is a recap. Only the messages'
text is kept: other attributes, and system instructions, are never stored.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from openmcp import db, repo
from openmcp.capture.common import is_recall_tool
from openmcp.capture.otlp import Record
from openmcp.clients import client_name
from openmcp.config import Config, resolve_client
from openmcp.ingest import store_conversation
from openmcp.models import ParsedConversation, ParsedMessage

INPUT, OUTPUT, CONVERSATION = "gen_ai.input.messages", "gen_ai.output.messages", "gen_ai.conversation.id"
SERVICE = "service.name"
LEGACY_ROLES = {"gen_ai.system.message": "system", "gen_ai.user.message": "user",
                "gen_ai.assistant.message": "assistant", "gen_ai.tool.message": "tool"}
LEGACY_CHOICE = "gen_ai.choice"
METADATA = {"capture": "otel"}
ROLES = ("user", "assistant")


@dataclass
class OtelReport:
    operations: int = 0
    """Model calls that carried messages."""
    messages: int = 0
    """Messages added or changed."""
    skipped: list[str] = field(default_factory=list)


def _messages(value: Any) -> list[dict[str, Any]]:
    """A messages attribute: structured, or a JSON string (allowed on spans)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    return [m for m in value if isinstance(m, dict)] if isinstance(value, list) else []


def _parts(message: dict[str, Any]) -> list[dict[str, Any]]:
    parts = message.get("parts")
    return [p for p in parts if isinstance(p, dict)] if isinstance(parts, list) else []


def _text(message: dict[str, Any]) -> str:
    return "\n\n".join(
        p["content"].strip() for p in _parts(message)
        if p.get("type") == "text" and isinstance(p.get("content"), str) and p["content"].strip()
    )


def _calls_recall_tool(messages: Iterable[dict[str, Any]]) -> bool:
    return any(
        p.get("type") == "tool_call" and isinstance(p.get("name"), str) and is_recall_tool(p["name"])
        for m in messages for p in _parts(m)
    )


def _id(role: str, text: str) -> str:
    return f"{role}:{hashlib.sha256(text.encode()).hexdigest()[:24]}"


@dataclass(frozen=True)
class Operation:
    app: str | None
    conversation: str | None
    history: list[dict[str, Any]]
    output: list[dict[str, Any]]
    start: datetime | None
    end: datetime | None


def _app(record: Record) -> str | None:
    value = record.resource.get(SERVICE)
    return value if isinstance(value, str) else None


def _legacy_message(record: Record) -> tuple[str, dict[str, Any]] | None:
    """An event of the earlier conventions, as ("input" | "output", message in today's shape):
    gen_ai.{system,user,assistant,tool}.message with {content, tool_calls, id} in the body,
    and gen_ai.choice with {index, message: {role, content, tool_calls}}."""
    body = record.body if isinstance(record.body, dict) else {}
    if record.name == LEGACY_CHOICE:
        source, role = body.get("message") if isinstance(body.get("message"), dict) else {}, "assistant"
    elif record.name in LEGACY_ROLES:
        source, role = body, LEGACY_ROLES[record.name]
    else:
        return None
    role = source.get("role") if isinstance(source.get("role"), str) else role
    parts: list[dict[str, Any]] = []
    if isinstance(source.get("content"), str) and source["content"]:
        kind = "tool_call_response" if role == "tool" else "text"
        parts.append({"type": kind, "content": source["content"]} if kind == "text"
                     else {"type": kind, "id": source.get("id"), "response": source["content"]})
    for call in source.get("tool_calls") if isinstance(source.get("tool_calls"), list) else []:
        function = call.get("function") if isinstance(call, dict) else None
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            parts.append({"type": "tool_call", "id": call.get("id"), "name": function["name"]})
    return ("output" if record.name == LEGACY_CHOICE else "input"), {"role": role, "parts": parts}


def operations(records: Iterable[Record]) -> list[Operation]:
    """The model calls among these records that carry message content, oldest first: one
    per span or event with gen_ai.input/output.messages, and one per span's worth of
    earlier-convention message events (which a span id ties together)."""
    found: list[Operation] = []
    legacy: dict[tuple[str | None, str], list[tuple[str, dict[str, Any], Record]]] = {}
    for r in records:
        if INPUT in r.attributes or OUTPUT in r.attributes:
            conversation = r.attributes.get(CONVERSATION)
            found.append(Operation(
                app=_app(r), conversation=conversation if isinstance(conversation, str) else None,
                history=_messages(r.attributes.get(INPUT)), output=_messages(r.attributes.get(OUTPUT)),
                start=r.start, end=r.end,
            ))
        elif (message := _legacy_message(r)) is not None:
            # Events without a span id can't be told apart: each is a call on its own.
            legacy.setdefault((_app(r), r.span or f"event:{id(r)}"), []).append((*message, r))
    for (app, _), events in legacy.items():
        times = [r.start for _, _, r in events if r.start]
        found.append(Operation(
            app=app, conversation=None,
            history=[m for kind, m, _ in events if kind == "input"],
            output=[m for kind, m, _ in events if kind == "output"],
            start=min(times) if times else None, end=max(times) if times else None,
        ))
    return sorted(found, key=lambda o: (o.end or o.start or datetime.max.astimezone()))


def ingest(conn: sqlite3.Connection, cfg: Config, records: Iterable[Record]) -> OtelReport:
    report = OtelReport()
    for op in operations(records):
        report.operations += 1
        _store(conn, cfg, op, report)
    return report


def _turns(op: Operation) -> list[tuple[str, str, bool, bool]]:
    """(role, text, recall, latest) for each user and assistant message with text. A
    turn (a user message and what follows it) is a recap if any message in it, or this
    call's answer when it's the latest turn, calls one of OpenMCP's recall tools. That's
    worked out from the history alone, so every later call that re-sends it agrees."""
    owner: list[int | None] = []
    recall: dict[int, bool] = {}
    current: int | None = None
    for i, m in enumerate(op.history):
        if m.get("role") == "user":
            current, recall[i] = i, False
        elif current is not None and _calls_recall_tool([m]):
            recall[current] = True
        owner.append(current)
    if current is not None and _calls_recall_tool(op.output):
        recall[current] = True
    return [
        (m["role"], text, recall.get(owner[i], False) if owner[i] is not None else False, i == current)
        for i, m in enumerate(op.history)
        if m.get("role") in ROLES and (text := _text(m))
    ]


def _store(conn: sqlite3.Connection, cfg: Config, op: Operation, report: OtelReport) -> None:
    turns = _turns(op)
    opening = next((text for role, text, _, _ in turns if role == "user"), None)
    if opening is None:
        report.skipped.append("a model call without a user message")
        return
    latest_recall = next((r for role, _, r, latest in turns if latest), False)
    answer = next((t for m in op.output if (t := _text(m))), None)  # the first choice with text

    provider = client_name(op.app or "unknown_service")
    conversation = op.conversation or f"opening:{_id('user', f'{provider}|{opening}')}"
    messages = [
        ParsedMessage(external_id=_id(role, text), role=role, content=text,
                      created_at=op.start if latest else None, recall=recall)
        for role, text, recall, latest in turns
    ]
    if answer:
        messages.append(ParsedMessage(external_id=_id("assistant", answer), role="assistant", content=answer,
                                      created_at=op.end, recall=latest_recall))
    scope = resolve_client(cfg, provider).write_scope
    with db.transaction(conn):
        is_new = repo.find_conversation(conn, provider, conversation) is None
        _, changed = store_conversation(conn, ParsedConversation(
            provider=provider,
            external_id=conversation,
            title=opening if is_new else None,
            created_at=op.start,
            updated_at=op.end or op.start,
            metadata=METADATA,
            messages=messages,
        ), scope)
    report.messages += changed
