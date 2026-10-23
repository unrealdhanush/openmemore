"""What a client may store (`writes` under [clients.<name>]): everything (the default),
memories only (none of its conversations are kept, by any route), or nothing (it only
reads). Reading is the same at every level."""

from contextlib import closing

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from conftest import connect, insert, plain
from openmemore import Memory, OpenMemoreError, config, db, repo
from openmemore.capture import SOURCES, Source, capture
from openmemore.cli import app, initialize
from openmemore.setup import plan
from test_otel import CHAT, send, traces_json
from test_setup import ctx, run  # noqa: F401 - the ctx fixture


def set_writes(home, client, writes):
    config.set_writes(home / "config.toml", client, writes)


@pytest.fixture
def store(home):
    initialize(home)
    with closing(db.connect(home / "openmemore.db")) as conn:
        memory = insert(conn, "Prefers metric units.")
    return home, memory


def conversations(home):
    with closing(db.connect(home / "openmemore.db")) as conn:
        return repo.list_conversations(conn, scopes=None)


# --- config -------------------------------------------------------------------------------


def test_writes_comes_from_the_client_then_defaults(home):
    initialize(home)
    cfg = config.load_config(home / "config.toml")
    assert config.resolve_client(cfg, "anyone").writes == "all"
    cfg.defaults = config.ClientScopes(read_scopes=("personal",), write_scope="personal", writes="memories")
    cfg.clients["cursor-vscode"] = config.ClientOverride(writes="none")
    assert config.resolve_client(cfg, "anyone").writes == "memories"
    assert config.resolve_client(cfg, "cursor-vscode").writes == "none"
    with pytest.raises(ValidationError):
        config.ClientOverride(writes="some")


def test_setting_it_back_to_all_leaves_no_empty_table(home):
    initialize(home)
    set_writes(home, "claude-code", "memories")
    assert config.load_config(home / "config.toml").clients["claude-code"].writes == "memories"
    set_writes(home, "claude-code", "all")
    assert "claude-code" not in config.load_config(home / "config.toml").clients


# --- the MCP tools ------------------------------------------------------------------------


@pytest.mark.anyio
async def test_memories_only_reads_and_saves_memories_but_keeps_no_conversations(store):
    home, memory = store
    set_writes(home, "claude-code", "memories")
    async with connect(home, "claude-code") as client:
        ping = (await client.call_tool("ping", {})).structured_content
        assert ping["writes"] == "memories"
        found = (await client.call_tool("search_context", {"query": "metric units"})).structured_content
        assert [r["id"] for r in found["results"]] == [memory]
        recorded = await client.call_tool("record_exchange", {"question": "Plan my garden.", "answer_summary": "Tomatoes."})
        assert not recorded.is_error and recorded.structured_content["conversation_id"] is None
        assert "aren't kept" in recorded.structured_content["note"]
        saved = await client.call_tool("save_memories", {"items": [{"kind": "preference", "content": "Likes tea."}]})
        assert not saved.is_error and len(saved.structured_content["saved"]) == 1
    assert conversations(home) == []


@pytest.mark.anyio
async def test_none_only_reads(store):
    home, memory = store
    set_writes(home, "claude-code", "none")
    async with connect(home, "claude-code") as client:
        assert (await client.call_tool("search_context", {"query": "metric units"})).structured_content["results"]
        assert (await client.call_tool("get_profile", {})).structured_content["facts"]
        for tool, args in [
            ("save_memories", {"items": [{"kind": "preference", "content": "Likes tea."}]}),
            ("update_memory", {"id": memory, "content": "Prefers imperial units."}),
            ("forget", {"id": memory}),
            ("propose_memories", {"items": [{"kind": "fact", "content": "Has a garden."}], "done": []}),
        ]:
            result = await client.call_tool(tool, args)
            assert result.is_error and 'writes = "none"' in result.content[0].text, tool
        recorded = await client.call_tool("record_exchange", {"question": "Plan my garden.", "answer_summary": "Tomatoes."})
        assert recorded.structured_content["conversation_id"] is None
        assert "save_memories" not in recorded.structured_content["note"]  # it can't, so it isn't told to
    with closing(db.connect(home / "openmemore.db")) as conn:
        assert [m.content for m in repo.list_memories(conn, scopes=None)] == ["Prefers metric units."]
    assert conversations(home) == []


@pytest.mark.anyio
async def test_other_clients_are_unaffected(store):
    home, _ = store
    set_writes(home, "claude-code", "none")
    async with connect(home, "cursor") as client:
        recorded = await client.call_tool("record_exchange", {"question": "Plan my garden.", "answer_summary": "Tomatoes."})
        assert recorded.structured_content["conversation_id"] is not None


# --- capture, OpenTelemetry and the library ------------------------------------------------


def test_transcripts_of_a_client_that_keeps_no_conversations_are_not_read(home, monkeypatch, tmp_path):
    initialize(home)
    read = []
    monkeypatch.setitem(SOURCES, "claude-code", Source(lambda h: read.append(h) or [], lambda p, o: (None, o),
                                                       app="claude-code"))
    cfg = config.load_config(home / "config.toml")
    cfg.capture.sources = ("claude-code",)
    cfg.clients["claude-code"] = config.ClientOverride(writes="memories")
    with closing(db.connect(home / "openmemore.db")) as conn:
        report = capture(conn, cfg, user_home=tmp_path)
    assert report.not_recorded == ["claude-code"] and read == []


def test_opentelemetry_from_a_client_that_keeps_no_conversations_is_dropped(home):
    initialize(home)
    set_writes(home, "garden-agent", "memories")
    cfg = config.load_config(home / "config.toml")
    with closing(db.connect(home / "openmemore.db")) as conn:
        report = send(conn, cfg, traces_json(*CHAT))
    assert report.messages == 0 and any("aren't kept" in s for s in report.skipped)
    assert conversations(home) == []


def test_the_library_follows_the_setting(home):
    initialize(home)
    set_writes(home, "trip-planner", "memories")
    memory = Memory("trip-planner")
    assert memory.record("Plan a trip to Lisbon.", "Three days, by train.") is None
    assert memory.save("Prefers trains.", kind="preference") is not None
    set_writes(home, "trip-planner", "none")
    with pytest.raises(OpenMemoreError, match='writes = "none"'):
        memory.save("Prefers window seats.", kind="preference")
    assert [h.text for h in memory.search("trains")] == ["Prefers trains."]
    assert conversations(home) == []


# --- setup and the CLI ---------------------------------------------------------------------


def test_setup_with_memories_only_skips_capture_and_says_not_to_record(ctx):
    steps = plan("claude-code", ctx, writes="memories")
    assert not any("transcript capture" in s.description for s in steps)
    run(steps)  # adding the server fails here (no claude command): the rest still happens
    cfg = config.load_config(ctx.base / "config.toml")
    assert cfg.clients["claude-code"].writes == "memories" and cfg.capture.sources == ()
    pasted = next(s.manual for s in steps if s.manual and "Use the openmemore tools" in s.manual)
    assert "Don't call record_exchange: conversations here aren't kept." in pasted


def test_setup_again_keeps_the_setting_unless_told(ctx):
    run(plan("claude-code", ctx, writes="none"))
    steps = plan("claude-code", ctx)  # e.g. to fix the server's path
    assert not any("writes" in s.description or "transcript capture" in s.description for s in steps)
    run(steps)
    assert config.load_config(ctx.base / "config.toml").clients["claude-code"].writes == "none"
    run(plan("claude-code", ctx, writes="all"))
    cfg = config.load_config(ctx.base / "config.toml")
    assert "claude-code" not in cfg.clients and cfg.capture.sources == ("claude-code",)


def test_setup_names_the_client_by_what_the_app_reports(ctx):
    run(plan("cursor", ctx, writes="memories"))
    assert config.load_config(ctx.base / "config.toml").clients["cursor-vscode"].writes == "memories"


def test_setup_needs_a_client_to_limit(home):
    initialize(home)
    result = CliRunner().invoke(app, ["setup", "--writes", "none", "--dry-run", "--dev"], env={"COLUMNS": "200"})
    assert result.exit_code == 2 and "name the client" in plain(result.output)


@pytest.mark.anyio
async def test_clients_shows_what_each_one_stores(home):
    initialize(home)
    set_writes(home, "claude-code", "memories")
    async with connect(home, "claude-code") as client:
        await client.call_tool("ping", {})
    out = plain(CliRunner().invoke(app, ["clients"], env={"COLUMNS": "200"}).output)
    assert "memories only" in out
