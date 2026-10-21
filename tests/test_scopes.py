"""Every tool honors scopes. One application writes 'personal' data through
every write path; a client configured to read only 'work' tries every read and write
path and sees or changes none of it, and can't tell its ids from ids that don't exist."""

import json

import pytest

from conftest import connect
from openmemore import config
from openmemore.cli import initialize

HOME_APP, WORK_APP = "home-app", "work-app"


@pytest.fixture
def scoped(home, fake_embedder):
    """'home-app' reads and writes personal; 'work-app' reads and writes only work."""
    initialize(home)
    path = home / "config.toml"
    cfg = config.load_config(path)
    cfg.clients[HOME_APP] = config.ClientOverride(read_scopes=("personal",), write_scope="personal")
    cfg.clients[WORK_APP] = config.ClientOverride(read_scopes=("work",), write_scope="work")
    config.save_config(path, cfg)
    return home


async def call(client, tool, args=None):
    result = await client.call_tool(tool, args or {})
    return result, result.structured_content


async def personal_data(home):
    async with connect(home, HOME_APP) as client:
        _, saved = await call(client, "save_memories", {"items": [
            {"kind": "preference", "content": "Prefers dark roast coffee."},
            {"kind": "project", "content": "Planning a garden for spring."},
        ]})
        _, chat = await call(client, "record_exchange", {
            "question": "Which coffee beans for a moka pot?", "answer_summary": "Dark roast, medium-fine grind.",
        })
    return saved["saved"], chat["conversation_id"]


@pytest.mark.anyio
async def test_a_work_only_client_cant_see_personal_data(scoped):
    memory_ids, conversation_id = await personal_data(scoped)

    async with connect(scoped, WORK_APP) as work:
        _, ping = await call(work, "ping")
        assert (ping["client"], ping["read_scopes"], ping["write_scope"]) == (WORK_APP, ["work"], "work")

        for query in ("coffee", "dark roast coffee preference", "moka pot beans", "garden"):
            _, found = await call(work, "search_context", {"query": query})
            assert found["results"] == [], query  # keyword and vector paths
        _, profile = await call(work, "get_profile")
        assert all(profile[group] == [] for group in ("preferences", "facts", "active_projects", "recent_decisions"))
        resource = json.loads((await work.read_resource("openmemore://profile")).contents[0].text)
        assert resource["preferences"] == [] and resource["active_projects"] == []
        _, recent = await call(work, "recent_conversations")
        assert recent["conversations"] == []


@pytest.mark.anyio
async def test_a_work_only_client_cant_change_personal_data_or_tell_it_exists(scoped):
    memory_ids, conversation_id = await personal_data(scoped)
    missing_memory, missing_conversation = 9999, 9999

    async with connect(scoped, WORK_APP) as work:
        for tool, args, missing_args in [
            ("update_memory", {"id": memory_ids[0], "content": "Prefers tea."}, {"id": missing_memory, "content": "x"}),
            ("forget", {"id": memory_ids[0]}, {"id": missing_memory}),
            ("recent_conversations", {"conversation_id": conversation_id}, {"conversation_id": missing_conversation}),
            ("record_exchange", {"question": "q", "answer_summary": "a", "conversation_id": conversation_id},
             {"question": "q", "answer_summary": "a", "conversation_id": missing_conversation}),
        ]:
            real, _ = await call(work, tool, args)
            fake, _ = await call(work, tool, missing_args)
            assert real.is_error and fake.is_error, tool
            # The same error for an out-of-scope id as for one that doesn't exist.
            assert real.content[0].text.replace(str(args.get("id", args.get("conversation_id"))), "N") == \
                fake.content[0].text.replace(str(missing_args.get("id", missing_args.get("conversation_id"))), "N"), tool

        refused, _ = await call(work, "save_memories", {"items": [{"kind": "fact", "content": "x"}], "scope": "personal"})
        assert refused.is_error and "can't write to scope 'personal'" in refused.content[0].text

        # A near-duplicate of a personal memory isn't reported to a client that can't read it.
        _, dup = await call(work, "save_memories", {"items": [{"kind": "preference", "content": "Prefers dark roast coffee."}]})
        assert dup["possible_duplicates"] == [] and len(dup["saved"]) == 1

    async with connect(scoped, HOME_APP) as home_client:  # everything is still there for its owner
        _, found = await call(home_client, "search_context", {"query": "coffee"})
        assert {h["type"] for h in found["results"]} == {"memory", "chat"}
        _, recent = await call(home_client, "recent_conversations", {"conversation_id": conversation_id})
        assert len(recent["conversations"][0]["messages"]) == 2


@pytest.mark.anyio
async def test_the_cli_is_the_owner_and_sees_every_scope(scoped):
    """The CLI acts for the user, who owns everything: it lists and exports all scopes."""
    from typer.testing import CliRunner
    from openmemore.cli import app

    await personal_data(scoped)
    async with connect(scoped, WORK_APP) as work:
        await call(work, "save_memories", {"items": [{"kind": "fact", "content": "Uses Go at work."}]})
    out = CliRunner().invoke(app, ["memories", "list"], env={"COLUMNS": "200"}).output
    assert "Prefers dark roast coffee." in out and "Uses Go at work." in out
