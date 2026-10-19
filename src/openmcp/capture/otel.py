"""Capture through OpenTelemetry (SPEC.md M6): conversations from clients that export
their text as OTLP log events, received by `openmcp daemon` (openmcp.daemon).

Claude Code 2.1 does, with CLAUDE_CODE_ENABLE_TELEMETRY=1 and OTEL_LOG_USER_PROMPTS=1
(Section 14). A turn is a `user_prompt` event (prompt text, `prompt.id`, `message.uuid`,
`session.id`), then one `assistant_response` event per model call (its text, the same
`prompt.id`), and `tool_result` events naming the tools it used. They're stored with the
ids the transcript reader uses (openmcp.capture.claude_code): the session id names the
conversation, the prompt's uuid its message, and '<uuid>:answer' the answer, whose parts
are joined the same way. Both sources can run at once and update the same messages.

Only the ids and the text are kept: the events' other attributes (user, email,
organization, terminal) are never stored. Events from other services, or of other
kinds, are ignored.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from openmcp import db, repo
from openmcp.capture import claude_code
from openmcp.capture.common import is_recall_tool
from openmcp.clients import client_name
from openmcp.config import Config, resolve_client
from openmcp.ingest import store_conversation
from openmcp.models import ParsedConversation, ParsedMessage

CLAUDE_CODE_APP = "claude-code"
"""The application these conversations belong to, as it reports itself over MCP."""
METADATA = {"capture": "otel"}
TOOL_EVENTS = ("tool_result", "tool_decision")
EVENT_PREFIX = "claude_code."


@dataclass(frozen=True)
class Event:
    name: str
    attributes: dict[str, Any]
    at: datetime | None
    sequence: int


@dataclass
class OtelReport:
    events: int = 0
    messages: int = 0
    """Messages added or changed."""
    skipped: list[str] = field(default_factory=list)
    """Why events couldn't be used, e.g. prompts sent without their text."""


def _value(value: Any) -> Any:
    if not isinstance(value, dict):
        return None
    for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if key in value:
            return value[key]
    return None


def _attributes(items: Any) -> dict[str, Any]:
    if not isinstance(items, list):
        return {}
    return {
        item["key"]: _value(item.get("value"))
        for item in items
        if isinstance(item, dict) and isinstance(item.get("key"), str)
    }


def _when(attributes: dict[str, Any], record: dict[str, Any]) -> datetime | None:
    stamp = attributes.get("event.timestamp")
    if isinstance(stamp, str):
        try:
            return datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError:
            pass
    nanos = record.get("timeUnixNano")
    try:
        return datetime.fromtimestamp(int(nanos) / 1e9, UTC) if nanos else None
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def events(payload: Any) -> Iterator[Event]:
    """The log events in an OTLP/JSON logs request (ExportLogsServiceRequest)."""
    if not isinstance(payload, dict):
        return
    for resource_logs in payload.get("resourceLogs") or []:
        if not isinstance(resource_logs, dict):
            continue
        for scope_logs in resource_logs.get("scopeLogs") or []:
            if not isinstance(scope_logs, dict):
                continue
            for record in scope_logs.get("logRecords") or []:
                if not isinstance(record, dict):
                    continue
                attributes = _attributes(record.get("attributes"))
                name = attributes.get("event.name") or _value(record.get("body"))
                if not isinstance(name, str):
                    continue
                yield Event(
                    name=name.removeprefix(EVENT_PREFIX),
                    attributes=attributes,
                    at=_when(attributes, record),
                    sequence=_int(attributes.get("event.sequence")),
                )


def _text(attributes: dict[str, Any], key: str) -> str | None:
    value = attributes.get(key)
    return value if isinstance(value, str) and value.strip() else None


def ingest(conn: sqlite3.Connection, cfg: Config, payload: Any) -> OtelReport:
    """Store the conversations in one OTLP logs request."""
    report = OtelReport()
    scope = resolve_client(cfg, client_name(CLAUDE_CODE_APP)).write_scope
    for event in sorted(events(payload), key=lambda e: (e.at or datetime.min.replace(tzinfo=UTC), e.sequence)):
        a = event.attributes
        if event.name == "user_prompt":
            report.events += 1
            _prompt(conn, event, scope, report)
        elif event.name == "assistant_response":
            report.events += 1
            _response(conn, event, scope, report)
        elif event.name in TOOL_EVENTS and isinstance(a.get("tool_name"), str) and is_recall_tool(a["tool_name"]):
            report.events += 1
            if isinstance(a.get("prompt.id"), str):
                with db.transaction(conn):
                    repo.mark_otel_recall(conn, a["prompt.id"])
    return report


def _prompt(conn: sqlite3.Connection, event: Event, scope: str, report: OtelReport) -> None:
    a = event.attributes
    session, prompt_id, uuid = a.get("session.id"), a.get("prompt.id"), a.get("message.uuid")
    if not all(isinstance(v, str) and v for v in (session, prompt_id, uuid)):
        report.skipped.append("a user_prompt event without session.id, prompt.id, or message.uuid")
        return
    raw = _text(a, "prompt")
    if raw is None or raw == "<REDACTED>":
        report.skipped.append("a prompt sent without its text: set OTEL_LOG_USER_PROMPTS=1 for Claude Code")
        return
    text = claude_code.clean_prompt(raw)
    if text is None:  # a slash command or other scaffolding: not part of the conversation
        return
    with db.transaction(conn):
        known = repo.get_otel_prompt(conn, prompt_id)  # a repeated event keeps the turn's recall mark
        is_new = repo.find_conversation(conn, claude_code.PROVIDER, session) is None
        conversation_id, changed = store_conversation(conn, ParsedConversation(
            provider=claude_code.PROVIDER,
            external_id=session,
            title=text if is_new else None,  # the transcript's own title, if it has one, wins later
            created_at=event.at,
            updated_at=event.at,
            metadata=METADATA,
            messages=[ParsedMessage(
                external_id=uuid, role="user", content=text, created_at=event.at,
                recall=bool(known and known["recall"]),
            )],
        ), scope)
        if conversation_id is not None:  # None: the user deleted this conversation
            repo.save_otel_prompt(conn, prompt_id, conversation_id, uuid, event.at or datetime.now(UTC))
            report.messages += changed


def _response(conn: sqlite3.Connection, event: Event, scope: str, report: OtelReport) -> None:
    a = event.attributes
    prompt_id, text = a.get("prompt.id"), _text(a, "response")
    if not isinstance(prompt_id, str) or text is None:
        return  # a model call that only used tools has no text
    with db.transaction(conn):
        prompt = repo.get_otel_prompt(conn, prompt_id)
        if prompt is None:
            report.skipped.append("a response to a prompt that wasn't received (the daemon may have been down)")
            return
        parts = repo.add_otel_response(conn, prompt_id, event.sequence, text.strip(), event.at)
        conversation = repo.get_conversation(conn, prompt["conversation_id"])
        first_at = parts[0][1]
        _, changed = store_conversation(conn, ParsedConversation(
            provider=claude_code.PROVIDER,
            external_id=conversation.external_id,
            updated_at=event.at,
            metadata=METADATA,
            messages=[ParsedMessage(
                external_id=f"{prompt['message_uuid']}:answer",
                role="assistant",
                content="\n\n".join(t for t, _ in parts),  # as the transcript reader joins an answer's parts
                created_at=datetime.fromisoformat(first_at) if first_at else event.at,
                recall=bool(prompt["recall"]),
            )],
        ), scope)
        report.messages += changed
