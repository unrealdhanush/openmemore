"""OTLP export requests (SPEC.md M6): logs and traces, as OTLP/HTTP sends them, in JSON
or protobuf, read into one shape.

Protobuf is read directly from the wire format, for just the messages OpenMemore uses
(opentelemetry-proto v1.11.1, `opentelemetry/proto/{common,resource,logs,trace}/v1`),
so the standard needs no extra package. Unknown fields are skipped, as protobuf allows.
"""

from __future__ import annotations

import base64
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

Signal = Literal["logs", "traces"]


@dataclass(frozen=True)
class Record:
    """A log record, a span, or a span's event: what OpenMemore reads from each."""

    name: str | None
    attributes: dict[str, Any]
    resource: dict[str, Any]
    start: datetime | None
    end: datetime | None
    body: Any = None
    """A log record's body (earlier GenAI conventions put message content there)."""
    span: str | None = None
    """The span a log record belongs to (trace id + span id, hex), linking events of one call."""


def _time(nanos: Any) -> datetime | None:
    try:
        value = int(nanos)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(value / 1e9, UTC) if value > 0 else None


# --- OTLP/JSON -------------------------------------------------------------------------


def _json_value(value: Any) -> Any:
    """An AnyValue in OTLP/JSON: {"stringValue": ...}, {"arrayValue": {"values": [...]}}, ..."""
    if not isinstance(value, dict):
        return None
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "intValue" in value:
        try:
            return int(value["intValue"])  # int64 travels as a string
        except (TypeError, ValueError):
            return None
    if "doubleValue" in value:
        return value["doubleValue"]
    if "arrayValue" in value:
        return [_json_value(v) for v in (value["arrayValue"] or {}).get("values") or []]
    if "kvlistValue" in value:
        return _json_attributes((value["kvlistValue"] or {}).get("values"))
    if "bytesValue" in value:
        try:
            return base64.b64decode(value["bytesValue"])
        except (TypeError, ValueError):
            return None
    return None


def _json_attributes(items: Any) -> dict[str, Any]:
    if not isinstance(items, list):
        return {}
    return {
        item["key"]: _json_value(item.get("value"))
        for item in items
        if isinstance(item, dict) and isinstance(item.get("key"), str)
    }


def _list(value: Any) -> list[dict[str, Any]]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def from_json(signal: Signal, payload: Any) -> Iterator[Record]:
    if not isinstance(payload, dict):
        return
    key, scopes_key, items_key = (
        ("resourceLogs", "scopeLogs", "logRecords") if signal == "logs" else ("resourceSpans", "scopeSpans", "spans")
    )
    for resource_block in _list(payload.get(key)):
        resource = _json_attributes((resource_block.get("resource") or {}).get("attributes"))
        for scope in _list(resource_block.get(scopes_key)):
            for item in _list(scope.get(items_key)):
                attributes = _json_attributes(item.get("attributes"))
                if signal == "logs":
                    at = _time(item.get("timeUnixNano")) or _time(item.get("observedTimeUnixNano"))
                    name = item.get("eventName") or attributes.get("event.name")
                    trace_id, span_id = item.get("traceId"), item.get("spanId")
                    span = f"{trace_id}:{span_id}" if isinstance(trace_id, str) and trace_id and span_id else None
                    yield Record(name if isinstance(name, str) else None, attributes, resource, at, at,
                                 _json_value(item.get("body")), span)
                    continue
                start, end = _time(item.get("startTimeUnixNano")), _time(item.get("endTimeUnixNano"))
                yield Record(item.get("name") if isinstance(item.get("name"), str) else None, attributes, resource,
                             start, end)
                for event in _list(item.get("events")):
                    at = _time(event.get("timeUnixNano"))
                    yield Record(event.get("name"), _json_attributes(event.get("attributes")), resource, at, at)


# --- OTLP/protobuf ---------------------------------------------------------------------


def _fields(buf: bytes) -> Iterator[tuple[int, int, Any]]:
    """(field number, wire type, value) for each field of a message. Varints come as
    ints, length-delimited fields and fixed-width ones as their raw bytes."""
    pos, end = 0, len(buf)

    def varint() -> int:
        nonlocal pos
        shift = result = 0
        while True:
            if pos >= end:
                raise ValueError("truncated varint")
            byte = buf[pos]
            pos += 1
            result |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return result
            shift += 7
            if shift > 63:
                raise ValueError("varint too long")

    while pos < end:
        key = varint()
        number, wire = key >> 3, key & 7
        if wire == 0:
            yield number, wire, varint()
        elif wire == 1:
            yield number, wire, buf[pos:pos + 8]
            pos += 8
        elif wire == 2:
            length = varint()
            if pos + length > end:
                raise ValueError("truncated field")
            yield number, wire, buf[pos:pos + length]
            pos += length
        elif wire == 5:
            yield number, wire, buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        if pos > end:
            raise ValueError("truncated field")


def _fixed64(raw: bytes) -> int:
    return struct.unpack("<Q", raw)[0] if len(raw) == 8 else 0


def _text(raw: Any) -> str:
    return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else ""


def _pb_value(buf: bytes) -> Any:
    """common.v1.AnyValue."""
    for number, wire, raw in _fields(buf):
        if number == 1 and wire == 2:
            return _text(raw)
        if number == 2 and wire == 0:
            return bool(raw)
        if number == 3 and wire == 0:
            return raw - (1 << 64) if raw >= 1 << 63 else raw  # int64, two's complement
        if number == 4 and wire == 1:
            return struct.unpack("<d", raw)[0]
        if number == 5 and wire == 2:  # ArrayValue
            return [_pb_value(v) for n, w, v in _fields(raw) if n == 1 and w == 2]
        if number == 6 and wire == 2:  # KeyValueList
            return _pb_attributes(v for n, w, v in _fields(raw) if n == 1 and w == 2)
        if number == 7 and wire == 2:
            return bytes(raw)
    return None


def _pb_attributes(key_values: Iterator[bytes]) -> dict[str, Any]:
    """common.v1.KeyValue messages."""
    out: dict[str, Any] = {}
    for kv in key_values:
        key, value = None, None
        for number, wire, raw in _fields(kv):
            if number == 1 and wire == 2:
                key = _text(raw)
            elif number == 2 and wire == 2:
                value = _pb_value(raw)
        if key is not None:
            out[key] = value
    return out


def _pb_resource(buf: bytes) -> dict[str, Any]:
    return _pb_attributes(raw for n, w, raw in _fields(buf) if n == 1 and w == 2)


def _pb_log(buf: bytes, resource: dict[str, Any]) -> Record:
    """logs.v1.LogRecord: time 1, observed time 11, body 5, attributes 6, trace_id 9,
    span_id 10, event_name 12."""
    times, attributes, name, body, ids = {}, [], None, None, {}
    for number, wire, raw in _fields(buf):
        if number in (1, 11) and wire == 1:
            times[number] = _fixed64(raw)
        elif number == 5 and wire == 2:
            body = _pb_value(raw)
        elif number == 6 and wire == 2:
            attributes.append(raw)
        elif number in (9, 10) and wire == 2:
            ids[number] = raw.hex()
        elif number == 12 and wire == 2:
            name = _text(raw)
    attrs = _pb_attributes(iter(attributes))
    at = _time(times.get(1)) or _time(times.get(11))
    name = name or attrs.get("event.name")
    span = f"{ids[9]}:{ids[10]}" if ids.get(9) and ids.get(10) else None
    return Record(name if isinstance(name, str) else None, attrs, resource, at, at, body, span)


def _pb_span(buf: bytes, resource: dict[str, Any]) -> Iterator[Record]:
    """trace.v1.Span: name 5, start 7, end 8, attributes 9, events 11 (time 1, name 2, attributes 3)."""
    name, start, end, attributes, events = None, None, None, [], []
    for number, wire, raw in _fields(buf):
        if number == 5 and wire == 2:
            name = _text(raw)
        elif number == 7 and wire == 1:
            start = _fixed64(raw)
        elif number == 8 and wire == 1:
            end = _fixed64(raw)
        elif number == 9 and wire == 2:
            attributes.append(raw)
        elif number == 11 and wire == 2:
            events.append(raw)
    yield Record(name, _pb_attributes(iter(attributes)), resource, _time(start), _time(end))
    for event in events:
        event_name, at, event_attributes = None, None, []
        for number, wire, raw in _fields(event):
            if number == 1 and wire == 1:
                at = _fixed64(raw)
            elif number == 2 and wire == 2:
                event_name = _text(raw)
            elif number == 3 and wire == 2:
                event_attributes.append(raw)
        yield Record(event_name, _pb_attributes(iter(event_attributes)), resource, _time(at), _time(at))


def from_protobuf(signal: Signal, body: bytes) -> list[Record]:
    """An ExportLogsServiceRequest or ExportTraceServiceRequest. Raises ValueError if it's
    not one."""
    records: list[Record] = []
    for number, wire, block in _fields(body):  # resource_logs / resource_spans
        if number != 1 or wire != 2:
            continue
        resource: dict[str, Any] = {}
        scopes = []
        for n, w, raw in _fields(block):
            if n == 1 and w == 2:
                resource = _pb_resource(raw)
            elif n == 2 and w == 2:
                scopes.append(raw)
        for scope in scopes:
            for n, w, raw in _fields(scope):  # scope_logs.log_records / scope_spans.spans
                if n != 2 or w != 2:
                    continue
                if signal == "logs":
                    records.append(_pb_log(raw, resource))
                else:
                    records.extend(_pb_span(raw, resource))
    return records
