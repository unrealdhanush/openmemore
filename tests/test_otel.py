"""SPEC.md M6: conversations captured from OpenTelemetry's GenAI semantic conventions, from
any app, as OTLP logs or traces, in JSON or protobuf. Protobuf payloads are encoded with
the official opentelemetry-proto package (a test-only dependency), so the reader is
checked against the reference encoding, not against itself."""

import json
from contextlib import closing
from datetime import UTC, datetime, timedelta

import pytest
from opentelemetry.proto.collector.logs.v1 import logs_service_pb2
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2
from opentelemetry.proto.common.v1 import common_pb2

from openmemore import config, db, repo
from openmemore.capture import otlp
from openmemore.capture.otel import ingest
from openmemore.cli import initialize

T0 = datetime(2026, 10, 19, 10, 0, tzinfo=UTC)
EVENT = "gen_ai.client.inference.operation.details"
IDENTITY = {"user.email": "someone@example.com", "enduser.id": "user-42", "host.name": "laptop-of-someone"}


def text(role, content):
    return {"role": role, "parts": [{"type": "text", "content": content}]}


def tool_call(name):
    return {"role": "assistant", "parts": [{"type": "tool_call", "id": "call_1", "name": name, "arguments": {}}]}


def tool_result(content):
    return {"role": "tool", "parts": [{"type": "tool_call_response", "id": "call_1", "response": content}]}


def call(history, answer=None, *, conversation=None, at=0, service="garden-agent", **extra):
    """One model call, as the conventions describe it."""
    attributes = {"gen_ai.operation.name": "chat", "gen_ai.provider.name": "openai",
                  "gen_ai.input.messages": history, **extra}
    if answer is not None:
        output = answer if isinstance(answer, list) else [{**text("assistant", answer), "finish_reason": "stop"}]
        attributes["gen_ai.output.messages"] = output
    if conversation:
        attributes["gen_ai.conversation.id"] = conversation
    start = T0 + timedelta(seconds=at)
    return {"service": service, "attributes": attributes, "start": start, "end": start + timedelta(seconds=2)}


def nanos(when):
    return str(int(when.timestamp() * 1e9))


# --- OTLP/JSON --------------------------------------------------------------------------


def any_json(value):
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, list):
        return {"arrayValue": {"values": [any_json(v) for v in value]}}
    if isinstance(value, dict):
        return {"kvlistValue": {"values": [{"key": k, "value": any_json(v)} for k, v in value.items()]}}
    raise TypeError(value)


def resource_json(service):
    return {"attributes": [{"key": k, "value": any_json(v)} for k, v in {"service.name": service, **IDENTITY}.items()]}


def traces_json(*calls, as_strings=False):
    """Spans; with as_strings, messages as JSON strings (allowed on spans)."""
    def attrs(c):
        return [{"key": k, "value": any_json(json.dumps(v) if as_strings and k.endswith("messages") else v)}
                for k, v in c["attributes"].items()]
    return {"resourceSpans": [{
        "resource": resource_json(c["service"]),
        "scopeSpans": [{"scope": {"name": "opentelemetry.instrumentation.openai_v2"}, "spans": [{
            "traceId": "5b8efff798038103d269b633813fc60c", "spanId": "eee19b7ec3c1b174", "kind": 3,
            "name": "chat some-model", "startTimeUnixNano": nanos(c["start"]), "endTimeUnixNano": nanos(c["end"]),
            "attributes": attrs(c),
        }]}],
    } for c in calls]}


def logs_json(*calls):
    return {"resourceLogs": [{
        "resource": resource_json(c["service"]),
        "scopeLogs": [{"scope": {"name": "opentelemetry.instrumentation.openai_v2"}, "logRecords": [{
            "timeUnixNano": nanos(c["end"]), "eventName": EVENT, "severityNumber": 5,
            "attributes": [{"key": k, "value": any_json(v)} for k, v in c["attributes"].items()],
        }]}],
    } for c in calls]}


# --- OTLP/protobuf, with the official classes ----------------------------------------------


def any_pb(value):
    v = common_pb2.AnyValue()
    if isinstance(value, bool):
        v.bool_value = value
    elif isinstance(value, int):
        v.int_value = value
    elif isinstance(value, float):
        v.double_value = value
    elif isinstance(value, bytes):
        v.bytes_value = value
    elif isinstance(value, str):
        v.string_value = value
    elif isinstance(value, list):
        v.array_value.values.extend(any_pb(x) for x in value)
    elif isinstance(value, dict):
        v.kvlist_value.values.extend(common_pb2.KeyValue(key=k, value=any_pb(x)) for k, x in value.items())
    return v


def kvs_pb(attributes):
    return [common_pb2.KeyValue(key=k, value=any_pb(v)) for k, v in attributes.items()]


def traces_pb(*calls):
    request = trace_service_pb2.ExportTraceServiceRequest()
    for c in calls:
        rs = request.resource_spans.add()
        rs.resource.attributes.extend(kvs_pb({"service.name": c["service"], **IDENTITY}))
        span = rs.scope_spans.add().spans.add()
        span.trace_id, span.span_id, span.name, span.kind = b"\x01" * 16, b"\x02" * 8, "chat some-model", 3
        span.start_time_unix_nano = int(nanos(c["start"]))
        span.end_time_unix_nano = int(nanos(c["end"]))
        span.attributes.extend(kvs_pb(c["attributes"]))
        span.status.code = 1  # fields OpenMemore doesn't read: skipped
        span.links.add(trace_id=b"\x03" * 16, span_id=b"\x04" * 8)
    return request.SerializeToString()


def logs_pb(*calls):
    request = logs_service_pb2.ExportLogsServiceRequest()
    for c in calls:
        rl = request.resource_logs.add()
        rl.resource.attributes.extend(kvs_pb({"service.name": c["service"], **IDENTITY}))
        record = rl.scope_logs.add().log_records.add()
        record.time_unix_nano = int(nanos(c["end"]))
        record.event_name, record.severity_number = EVENT, 5
        record.attributes.extend(kvs_pb(c["attributes"]))
    return request.SerializeToString()


# --- fixtures ------------------------------------------------------------------------------


@pytest.fixture
def cfg(home):
    initialize(home)
    return config.load_config(home / "config.toml")


def stored(conn):
    return [
        (c.provider, c.title, [(m.role, m.content, m.recall) for m in repo.conversation_messages(conn, c.id)])
        for c in sorted(repo.list_conversations(conn, scopes=None), key=lambda c: c.id)
    ]


def send(conn, cfg, payload, signal="traces"):
    records = otlp.from_protobuf(signal, payload) if isinstance(payload, bytes) else otlp.from_json(signal, payload)
    return ingest(conn, cfg, records)


HISTORY = [text("system", "You are a garden planner."), text("user", "Plan a vegetable garden.")]
TURN_2 = [*HISTORY, text("assistant", "Start with tomatoes."), text("user", "And for shade?")]
CHAT = [
    call(HISTORY, "Start with tomatoes."),
    call(TURN_2, "Lettuce and spinach.", at=10),
]
EXPECTED = [("garden-agent", "Plan a vegetable garden.", [
    ("user", "Plan a vegetable garden.", False), ("assistant", "Start with tomatoes.", False),
    ("user", "And for shade?", False), ("assistant", "Lettuce and spinach.", False),
])]


# --- the mapping ---------------------------------------------------------------------------


def test_a_conversation_builds_up_over_calls_without_repeats(conn, cfg):
    report = send(conn, cfg, traces_json(*CHAT))
    assert (report.operations, report.messages) == (2, 4)
    assert stored(conn) == EXPECTED
    [c] = repo.list_conversations(conn, scopes=None)
    assert c.metadata == {"capture": "otel"} and c.scope == "personal"
    assert send(conn, cfg, traces_json(*CHAT)).messages == 0  # a resend changes nothing


@pytest.mark.parametrize("encode, signal", [
    (lambda: traces_json(*CHAT), "traces"),
    (lambda: traces_json(*CHAT, as_strings=True), "traces"),
    (lambda: logs_json(*CHAT), "logs"),
    (lambda: traces_pb(*CHAT), "traces"),
    (lambda: logs_pb(*CHAT), "logs"),
], ids=["spans-json", "spans-json-strings", "events-json", "spans-protobuf", "events-protobuf"])
def test_every_standard_encoding_gives_the_same_conversation(conn, cfg, encode, signal):
    send(conn, cfg, encode(), signal)
    assert stored(conn) == EXPECTED


def test_spans_and_events_for_the_same_calls_don_t_duplicate(conn, cfg):
    send(conn, cfg, traces_json(*CHAT))
    assert send(conn, cfg, logs_json(*CHAT), "logs").messages == 0
    assert stored(conn) == EXPECTED


def test_only_conversation_text_is_kept(conn, cfg):
    send(conn, cfg, traces_json(call(
        [text("system", "SECRET SYSTEM PROMPT"), text("user", "What's the weather?"), tool_call("get_weather"),
         tool_result("rainy, 57F"), {"role": "assistant", "parts": [{"type": "reasoning", "content": "PRIVATE THOUGHTS"}]}],
        "It's rainy.", **{"gen_ai.system_instructions": [{"type": "text", "content": "INSTRUCTIONS"}]})))
    assert stored(conn) == [("garden-agent", "What's the weather?", [
        ("user", "What's the weather?", False), ("assistant", "It's rainy.", False)])]
    dump = "\n".join(conn.iterdump())
    for hidden in ("SECRET SYSTEM PROMPT", "rainy, 57F", "PRIVATE THOUGHTS", "INSTRUCTIONS", *IDENTITY.values()):
        assert hidden not in dump


def test_the_conversation_id_separates_chats(conn, cfg):
    same_opening = [text("user", "Hello!")]
    send(conn, cfg, traces_json(call(same_opening, "Hi.", conversation="conv_a"),
                                call(same_opening, "Hey.", conversation="conv_b")))
    assert sorted(c.external_id for c in repo.list_conversations(conn, scopes=None)) == ["conv_a", "conv_b"]


def test_without_an_id_the_opening_message_identifies_the_chat(conn, cfg):
    send(conn, cfg, traces_json(call([text("user", "Plan a garden.")], "Sure."),
                                call([text("user", "Fix my bike.")], "Okay.")))
    send(conn, cfg, traces_json(call([text("user", "Plan a garden."), text("assistant", "Sure."),
                                      text("user", "What about herbs?")], "Basil.", at=5)))
    assert [(t, len(m)) for _, t, m in stored(conn)] == [("Plan a garden.", 4), ("Fix my bike.", 2)]


def test_a_turn_that_used_a_recall_tool_is_a_recap_and_stays_one(conn, cfg):
    ask = [text("user", "What did I ask Copilot?")]
    send(conn, cfg, traces_json(call(ask, [tool_call("mcp_openmemore_recent_conversations")])))
    looked_up = [*ask, tool_call("mcp_openmemore_recent_conversations"), tool_result("lists vs tuples")]
    send(conn, cfg, traces_json(call(looked_up, "You asked about lists.", at=3)))
    next_turn = [*looked_up, text("assistant", "You asked about lists."), text("user", "Plan a garden.")]
    send(conn, cfg, traces_json(call(next_turn, "Start with tomatoes.", at=9)))
    [(_, _, messages)] = stored(conn)
    assert [(c, r) for _, c, r in messages] == [
        ("What did I ask Copilot?", True), ("You asked about lists.", True),
        ("Plan a garden.", False), ("Start with tomatoes.", False),
    ]


def test_other_tools_don_t_make_a_recap(conn, cfg):
    send(conn, cfg, traces_json(call([text("user", "Weather?"), tool_call("get_weather"), tool_result("sun")], "Sunny.")))
    [(_, _, messages)] = stored(conn)
    assert [r for _, _, r in messages] == [False, False]


def test_the_first_choice_with_text_is_the_answer(conn, cfg):
    send(conn, cfg, traces_json(call([text("user", "Name a fruit.")],
                                     [text("assistant", "Apple."), text("assistant", "Pear.")])))
    [(_, _, messages)] = stored(conn)
    assert messages[-1][1] == "Apple."


def test_the_app_is_its_service_name_and_gets_that_clients_scope(home):
    initialize(home)
    cfg = config.load_config(home / "config.toml")
    cfg.clients["work-bot"] = config.ClientOverride(read_scopes=("work",), write_scope="work")
    with closing(db.open_db(home / "openmemore.db")) as conn:
        send(conn, cfg, traces_json(call([text("user", "Deploy steps?")], "make deploy", service="Work Bot")))
        [c] = repo.list_conversations(conn, scopes=None)
        assert (c.provider, c.scope) == ("work-bot", "work")


def test_secrets_are_redacted_and_deleted_chats_stay_deleted(conn, cfg):
    send(conn, cfg, traces_json(call([text("user", "My key is ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789")], "Rotate it.")))
    [(_, _, messages)] = stored(conn)
    assert "ghp_" not in messages[0][1] and "[REDACTED]" in messages[0][1]
    [c] = repo.list_conversations(conn, scopes=None)
    with db.transaction(conn):
        repo.delete_conversation(conn, c.id)
    send(conn, cfg, traces_json(call([text("user", "My key is ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"),
                                      text("assistant", "Rotate it."), text("user", "Done.")], "Good.", at=4)))
    assert stored(conn) == []


def test_what_isn_t_a_conversation_is_ignored(conn, cfg):
    report = send(conn, cfg, traces_json(call([text("system", "Only a system prompt.")], "Ok."),
                                         {"service": "x", "attributes": {"http.method": "GET"}, "start": T0, "end": T0}))
    assert stored(conn) == [] and report.operations == 1 and report.skipped == ["a model call without a user message"]


# --- the protobuf reader -------------------------------------------------------------------


def test_protobuf_values_decode_like_the_reference():
    value = {"s": "text", "b": True, "i": -42, "big": 2**62, "d": 1.5, "raw": b"\x00\xff", "list": [1, "two", [3.0]],
             "nested": {"a": {"b": False}}}
    request = trace_service_pb2.ExportTraceServiceRequest()
    span = request.resource_spans.add().scope_spans.add().spans.add()
    span.attributes.extend(kvs_pb(value))
    [record] = otlp.from_protobuf("traces", request.SerializeToString())
    assert record.attributes == value


def test_span_events_are_read_too():
    request = trace_service_pb2.ExportTraceServiceRequest()
    span = request.resource_spans.add().scope_spans.add().spans.add()
    event = span.events.add(time_unix_nano=int(nanos(T0)), name=EVENT)
    event.attributes.extend(kvs_pb({"gen_ai.conversation.id": "conv_1"}))
    _, from_event = otlp.from_protobuf("traces", request.SerializeToString())
    assert (from_event.name, from_event.attributes, from_event.start) == (EVENT, {"gen_ai.conversation.id": "conv_1"}, T0)


@pytest.mark.parametrize("garbage", [b"\xff\xff\xff", b"\x0a\x10short", b"\x0f"])
def test_protobuf_that_isn_t_otlp_is_refused(garbage):
    with pytest.raises(ValueError):
        otlp.from_protobuf("traces", garbage)


def test_malformed_json_is_ignored():
    for payload in (None, [], {"resourceSpans": "x"}, {"resourceSpans": [{"scopeSpans": [{"spans": [{}]}]}]}):
        assert all(r.attributes == {} for r in otlp.from_json("traces", payload))


# --- the earlier conventions, which released instrumentation still emits --------------------


def legacy_events(history, answer, span="aa" * 8, trace="bb" * 16, at=0):
    """One call as opentelemetry-instrumentation-openai-v2 2.3 records it: an event per message."""
    def event(name, body, offset=0):
        return {"name": name, "body": body, "span": span, "trace": trace, "at": T0 + timedelta(seconds=at + offset)}
    events = [event(f"gen_ai.{m['role']}.message", {k: v for k, v in m.items() if k != "role"}) for m in history]
    events.append(event("gen_ai.choice", {"index": 0, "finish_reason": "stop",
                                          "message": {"role": "assistant", "content": answer}}, 1))
    return events


def legacy_json(*events, service="plant-app"):
    return {"resourceLogs": [{"resource": resource_json(service), "scopeLogs": [{"logRecords": [{
        "timeUnixNano": nanos(e["at"]), "eventName": e["name"], "traceId": e["trace"], "spanId": e["span"],
        "attributes": [{"key": "gen_ai.system", "value": any_json("openai")}], "body": any_json(e["body"]),
    } for e in events]}]}]}


def legacy_pb(*events, service="plant-app"):
    request = logs_service_pb2.ExportLogsServiceRequest()
    rl = request.resource_logs.add()
    rl.resource.attributes.extend(kvs_pb({"service.name": service}))
    scope = rl.scope_logs.add()
    for e in events:
        record = scope.log_records.add(time_unix_nano=int(nanos(e["at"])), event_name=e["name"],
                                       trace_id=bytes.fromhex(e["trace"]), span_id=bytes.fromhex(e["span"]))
        record.body.CopyFrom(any_pb(e["body"]))
        record.attributes.extend(kvs_pb({"gen_ai.system": "openai"}))
    return request.SerializeToString()


LEGACY_CHAT = [
    *legacy_events([{"role": "system", "content": "You help with houseplants."},
                    {"role": "user", "content": "Why are my fern's leaves brown?"}], "Too little humidity.", span="01" * 8),
    *legacy_events([{"role": "system", "content": "You help with houseplants."},
                    {"role": "user", "content": "Why are my fern's leaves brown?"},
                    {"role": "assistant", "content": "Too little humidity."},
                    {"role": "user", "content": "How often should I mist it?"}], "Daily.", span="02" * 8, at=5),
]
LEGACY_EXPECTED = [("plant-app", "Why are my fern's leaves brown?", [
    ("user", "Why are my fern's leaves brown?", False), ("assistant", "Too little humidity.", False),
    ("user", "How often should I mist it?", False), ("assistant", "Daily.", False),
])]


@pytest.mark.parametrize("encode", [legacy_json, legacy_pb], ids=["json", "protobuf"])
def test_earlier_convention_events_are_grouped_by_their_call(conn, cfg, encode):
    send(conn, cfg, encode(*LEGACY_CHAT), "logs")
    assert stored(conn) == LEGACY_EXPECTED
    assert "You help with houseplants." not in "\n".join(conn.iterdump())


def test_earlier_convention_tool_calls_mark_a_recap(conn, cfg):
    ask = [{"role": "user", "content": "What did I ask Copilot?"}]
    calls = [{"id": "c1", "type": "function", "function": {"name": "mcp_openmemore_search_context", "arguments": "{}"}}]
    events = [*legacy_events(ask, None, span="03" * 8)[:-1],
              {"name": "gen_ai.choice", "span": "03" * 8, "trace": "bb" * 16, "at": T0,
               "body": {"index": 0, "message": {"role": "assistant", "tool_calls": calls}}},
              *legacy_events([*ask, {"role": "assistant", "tool_calls": calls},
                              {"role": "tool", "id": "c1", "content": "lists"}], "Lists.", span="04" * 8, at=2)]
    send(conn, cfg, legacy_json(*events), "logs")
    [(_, _, messages)] = stored(conn)
    assert [(c, r) for _, c, r in messages] == [("What did I ask Copilot?", True), ("Lists.", True)]
