"""Automatic consolidation with a model OpenMemore calls itself, through the open
standard for LLM APIs (Open Responses, or Chat Completions). Off by default; a local
endpoint sends nothing off the machine; a remote one needs permission first."""

import json
import os
import socket
import stat
import threading
from contextlib import closing
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest
from typer.testing import CliRunner

from conftest import plain
from openmemore import config, consolidate, daemon, db, llm, repo
from openmemore.cli import app, initialize
from openmemore.ingest import store_conversation
from openmemore.models import ParsedConversation, ParsedMessage

LOCAL = "http://127.0.0.1:11434/v1"


def responses_body(text):
    return {"id": "resp_1", "status": "completed", "output": [
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": text, "annotations": []}]},
    ]}


class FakeModel:
    """An endpoint answering with canned memories, recording what it was asked."""

    def __init__(self, *answers, api="responses", status=200):
        self.answers, self.api, self.status, self.requests = list(answers), api, status, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.url.path, request.headers.get("authorization"), json.loads(request.content)))
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "nope", "type": "invalid_request"}})
        text = self.answers.pop(0) if self.answers else '{"memories": []}'
        if self.api == "responses":
            return httpx.Response(200, json=responses_body(text))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": text}}]})

    @property
    def transport(self):
        return httpx.MockTransport(self)


def memories(*items):
    return json.dumps({"memories": list(items)})


# --- the protocol clients -------------------------------------------------------------------


def test_open_responses_request_and_answer():
    fake = FakeModel("OK")
    model = llm.Model(endpoint=LOCAL + "/", model="some-model", api_key="sk-test")
    assert llm.complete(model, "Be brief.", "Are you there?", transport=fake.transport) == "OK"
    path, auth, body = fake.requests[0]
    assert path == "/v1/responses" and auth == "Bearer sk-test"
    assert (body["model"], body["instructions"], body["input"]) == ("some-model", "Be brief.", "Are you there?")


def test_chat_completions_request_and_answer():
    fake = FakeModel("OK", api="chat-completions")
    model = llm.Model(endpoint=LOCAL, model="m", api="chat-completions")
    assert llm.complete(model, "Be brief.", "Hi?", transport=fake.transport) == "OK"
    path, auth, body = fake.requests[0]
    assert path == "/v1/chat/completions" and auth is None
    assert body["messages"] == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi?"}]


@pytest.mark.parametrize("response, message", [
    (httpx.Response(404, json={"error": {"message": "no route"}}), 'set api = "chat-completions"'),
    (httpx.Response(401, json={"error": {"message": "bad key"}}), "answered 401: bad key"),
    (httpx.Response(200, json={"status": "failed", "error": {"message": "overloaded"}}), "overloaded"),
    (httpx.Response(302, headers={"location": "https://elsewhere.example/v1/responses"}), "redirected elsewhere"),
    (httpx.Response(200, text="<html>"), "didn't return JSON"),
])
def test_model_errors_say_what_went_wrong(response, message):
    model = llm.Model(endpoint=LOCAL, model="m")
    with pytest.raises(llm.ModelError, match=message.replace("(", r"\(")):
        llm.complete(model, "i", "p", transport=httpx.MockTransport(lambda r: response))


def test_what_counts_as_local():
    for endpoint in ("http://127.0.0.1:11434/v1", "http://localhost:1234/v1", "http://[::1]:8000/v1", "http://127.8.0.1/v1"):
        assert llm.is_local(endpoint), endpoint
    for endpoint in ("https://api.openai.com/v1", "http://192.168.1.5:11434/v1", "http://localhost.evil.example/v1"):
        assert not llm.is_local(endpoint), endpoint


# --- a run ----------------------------------------------------------------------------------


@pytest.fixture
def store(home):
    initialize(home)
    conn = db.connect(home / "openmemore.db")
    yield home, conn
    conn.close()


def chat(conn, external_id, *turns, scope="personal"):
    conversation_id, _ = store_conversation(conn, ParsedConversation(
        provider="vscode-copilot", external_id=external_id, title=f"chat {external_id}",
        messages=[ParsedMessage(external_id=f"{external_id}-{i}", role=r, content=t) for i, (r, t) in enumerate(turns)],
    ), scope)
    return conversation_id


def cfg_for(home):
    return config.load_config(home / "config.toml")


def test_a_run_turns_conversations_into_reviewable_proposals(store):
    home, conn = store
    work = chat(conn, "w", ("user", "We deploy on Fridays."), ("assistant", "Noted."), scope="work")
    model = llm.Model(endpoint=LOCAL, model="small-model")
    fake = FakeModel(memories({"kind": "fact", "content": "The team deploys on Fridays.", "reason": "stated"},
                              {"kind": "nonsense", "content": "dropped"}))
    report = consolidate.run(conn, cfg_for(home), model, transport=fake.transport)
    assert (report.conversations, report.proposals, report.dropped, report.errors) == (1, 1, 1, [])
    [p] = repo.list_proposals(conn)
    assert (p.content, p.scope, p.source_client, p.conversation_ids, p.status) == (
        "The team deploys on Fridays.", "work", "model:small-model", [work], "pending")
    assert repo.list_memories(conn, scopes=None) == []  # nothing is a memory until approved
    assert repo.count_due_conversations(conn, None) == 0
    prompt = fake.requests[0][2]["input"]
    assert "We deploy on Fridays." in prompt and "<stored_context" in prompt  # wrapped as data
    run = consolidate.last_run(conn)
    assert (run["trigger"], run["model"], run["host"], run["conversations"], run["proposals"], run["error"]) == (
        "manual", "small-model", "127.0.0.1", 1, 1, None)


def test_drift_becomes_an_update_of_the_memory_it_contradicts(store):
    from conftest import insert

    home, conn = store
    old = insert(conn, "Prefers tea in the morning.", kind="preference")
    chat(conn, "c", ("user", "I've switched to coffee in the mornings now."))
    fake = FakeModel(memories(
        {"kind": "preference", "content": "Prefers coffee in the morning.", "replaces": old},
        {"kind": "fact", "content": "Has a cat.", "replaces": 999},  # an id it wasn't shown: cleared
    ))
    consolidate.run(conn, cfg_for(home), llm.Model(endpoint=LOCAL, model="m"), transport=fake.transport)
    coffee, cat = repo.list_proposals(conn)
    assert (coffee.replaces_memory_id, cat.replaces_memory_id) == (old, None)
    assert f"id {old} [preference] Prefers tea in the morning." in fake.requests[0][2]["input"]


def test_a_model_error_stops_the_run_and_leaves_the_conversation_due(store):
    home, conn = store
    chat(conn, "a", ("user", "Plan a garden."))
    report = consolidate.run(conn, cfg_for(home), llm.Model(endpoint=LOCAL, model="m"),
                             transport=FakeModel(status=500).transport)
    assert report.conversations == 0 and "answered 500" in report.errors[0]
    assert repo.count_due_conversations(conn, None) == 1
    assert "answered 500" in consolidate.last_run(conn)["error"]


def test_runs_respect_the_limit_and_answers_wrapped_in_prose(store):
    home, conn = store
    for name in "abc":
        chat(conn, name, ("user", f"Note {name}."))
    fake = FakeModel(*["Sure! ```json\n" + memories({"kind": "note", "content": f"Note {n}."}) + "\n```" for n in "abc"])
    report = consolidate.run(conn, cfg_for(home), llm.Model(endpoint=LOCAL, model="m"), limit=2, transport=fake.transport)
    assert (report.conversations, report.proposals) == (2, 2)
    assert repo.count_due_conversations(conn, None) == 1


# --- off by default, permission, and the schedule ---------------------------------------------


def test_off_by_default(store):
    home, conn = store
    assert config.Config().consolidation.enabled is False and cfg_for(home).consolidation.enabled is False
    assert daemon.consolidation_due(home) is None


def set_up(home, endpoint=LOCAL, **changes):
    path = home / "config.toml"
    cfg = config.load_config(path)
    cfg.consolidation = config.ConsolidationConfig(enabled=True, endpoint=endpoint, model="m", **changes)
    config.save_config(path, cfg)
    return cfg


def test_a_remote_endpoint_needs_its_host_allowed(store):
    home, _ = store
    set_up(home, endpoint="https://api.example.com/v1")
    assert daemon.consolidation_due(home) is None
    set_up(home, endpoint="https://api.example.com/v1", allowed_host="api.example.com")
    assert daemon.consolidation_due(home) is not None
    set_up(home, endpoint="https://other.example.com/v1", allowed_host="api.example.com")
    assert daemon.consolidation_due(home) is None


def test_the_daily_schedule(store):
    home, conn = store
    cfg = set_up(home, schedule="03:00")
    today = datetime.now().astimezone().replace(hour=3, minute=0, second=0, microsecond=0)
    assert not consolidate.due(conn, cfg, today - timedelta(minutes=1))
    assert consolidate.due(conn, cfg, today + timedelta(hours=5))  # missed while asleep: runs when next checked
    consolidate.run(conn, cfg, llm.Model(endpoint=LOCAL, model="m"), trigger="schedule",
                    transport=FakeModel().transport)
    assert not consolidate.due(conn, cfg, datetime.now(UTC))
    assert consolidate.due(conn, cfg, today + timedelta(days=1, hours=1))


def test_the_key_is_kept_in_a_file_only_the_user_can_read(home, monkeypatch):
    initialize(home)
    monkeypatch.delenv(consolidate.KEY_ENV, raising=False)
    consolidate.save_key(home, "sk-secret\n")
    assert consolidate.read_key(home) == "sk-secret"
    if os.name != "nt":
        assert stat.S_IMODE(consolidate.key_path(home).stat().st_mode) == 0o600
    assert "sk-secret" not in (home / "config.toml").read_text()
    monkeypatch.setenv(consolidate.KEY_ENV, "sk-from-env")
    assert consolidate.read_key(home) == "sk-from-env"


# --- CLI --------------------------------------------------------------------------------------


def cli(*args, input=None):
    result = CliRunner().invoke(app, list(args), input=input, env={"COLUMNS": "200"})
    return result.exit_code, " ".join(plain(result.output).split())


@pytest.fixture
def answers(monkeypatch):
    """The model, as setup's test request and `consolidate run` see it."""
    fake = FakeModel()
    real = llm.complete
    monkeypatch.setattr(llm, "complete", lambda model, i, p, transport=None: real(model, i, p, transport=fake.transport))
    return fake


def test_setup_with_a_local_endpoint(home, answers):
    answers.answers.append("OK")
    code, out = cli("consolidate", "setup", "--endpoint", LOCAL, "--model", "small-model", "--schedule", "04:30")
    c = config.load_config(home / "config.toml").consolidation
    assert code == 0 and (c.enabled, c.model, c.schedule, c.allowed_host) == (True, "small-model", "04:30", None)
    assert "daily at 04:30 with small-model at 127.0.0.1" in out


def test_setup_with_a_remote_endpoint_asks_first(home, answers, consent_answers):
    code, out = cli("consolidate", "setup", "--endpoint", "https://api.example.com/v1", "--model", "big")
    assert code == 1 and "won't be sent to api.example.com" in out
    assert config.load_config(home / "config.toml").consolidation.enabled is False
    consent_answers.append("allowed")
    answers.answers.append("OK")
    code, _ = cli("consolidate", "setup", "--endpoint", "https://api.example.com/v1", "--model", "big",
                  "--key", input="sk-remote\n")
    c = config.load_config(home / "config.toml").consolidation
    assert code == 0 and (c.enabled, c.allowed_host) == (True, "api.example.com")
    assert consolidate.read_key(home) == "sk-remote" and answers.requests[-1][1] == "Bearer sk-remote"


def test_a_failing_test_request_leaves_it_off(home, answers):
    answers.status = 404
    code, out = cli("consolidate", "setup", "--endpoint", LOCAL, "--model", "m")
    assert code == 1 and "test request failed" in out and 'api = "chat-completions"' in out
    assert config.load_config(home / "config.toml").consolidation.enabled is False


def test_run_status_and_off(home, answers):
    code, out = cli("consolidate", "run")
    assert code == 2 and "Not set up yet" in out
    set_up(home)
    with closing(db.connect(home / "openmemore.db")) as conn:
        chat(conn, "a", ("user", "I prefer window seats."))
    answers.answers.append(memories({"kind": "preference", "content": "Prefers window seats."}))
    code, out = cli("consolidate", "run")
    assert code == 0 and "Consolidated 1 conversations into 1 proposals" in out and "openmemore review" in out
    code, out = cli("consolidate", "status")
    assert "last run" in out and "1 proposals" in out
    code, out = cli("consolidate", "off", "--forget")
    assert config.load_config(home / "config.toml").consolidation.enabled is False
    code, out = cli("status")
    assert "Automatic consolidation: off" in out


# --- done when: a local endpoint sends nothing off the machine ---------------------------------


class StandIn(BaseHTTPRequestHandler):
    """A local Open Responses server, as Ollama or LM Studio would serve one."""

    def do_POST(self):
        json.loads(self.rfile.read(int(self.headers["content-length"])))
        data = json.dumps(responses_body(memories({"kind": "preference", "content": "Prefers aisle seats."}))).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_a_local_endpoint_sends_nothing_off_the_machine(store, monkeypatch):
    home, conn = store
    server = HTTPServer(("127.0.0.1", 0), StandIn)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    peers = []
    real_connect = socket.socket.connect

    def connect(sock, address):
        peers.append(address[0])
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    for proxy in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(proxy, "http://203.0.113.9:3128")  # a proxy elsewhere must not be used
    try:
        chat(conn, "a", ("user", "Book me an aisle seat, as always."))
        model = llm.Model(endpoint=f"http://127.0.0.1:{server.server_port}/v1", model="local-model")
        report = consolidate.run(conn, cfg_for(home), model)
    finally:
        server.shutdown()
    assert (report.proposals, report.errors) == (1, [])
    assert peers and all(peer == "127.0.0.1" for peer in peers)
