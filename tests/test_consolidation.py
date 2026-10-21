"""consolidation_batch and propose_memories. A client reads conversations not
yet consolidated and proposes memories, which wait for the user's review; it can't read,
propose from, or mark conversations outside its scopes."""

from contextlib import closing

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from conftest import caller, connect, insert
from openmemore import config, db, repo
from openmemore.capture.common import is_recall_tool
from openmemore.cli import initialize
from openmemore.ingest import store_conversation
from openmemore.models import DoneConversation, ParsedConversation, ParsedMessage, ProposalInput
from openmemore.proposals import reject
from openmemore.tools import consolidation
from openmemore.tools.consolidation import consolidation_batch, propose_memories

ME = caller("claude-ai", read=["personal", "work"], write="personal")


def chat(conn, external_id, turns, *, scope="personal", provider="vscode-copilot", recall=()):
    """A captured conversation: turns are (role, text); indexes in `recall` are recaps."""
    conversation_id, _ = store_conversation(conn, ParsedConversation(
        provider=provider, external_id=external_id, title=f"chat {external_id}", metadata={"capture": "transcript"},
        messages=[
            ParsedMessage(external_id=f"{external_id}-{i}", role=role, content=text, recall=i in recall)
            for i, (role, text) in enumerate(turns)
        ],
    ), scope)
    return conversation_id


def done_for(batch):
    return [DoneConversation(conversation_id=c.id, through_message_id=c.through_message_id) for c in batch.conversations]


def item(content, *sources, kind="preference", **extra):
    return ProposalInput(kind=kind, content=content, conversation_ids=list(sources), **extra)


def test_a_batch_has_unread_messages_newest_first_without_recaps(conn):
    older = chat(conn, "a", [("user", "I like tea."), ("assistant", "Noted.")])
    newer = chat(conn, "b", [("user", "What did I ask before?"), ("assistant", "You asked about tea."),
                             ("user", "Plan my garden."), ("assistant", "Here's a plan.")], recall={0, 1})
    batch = consolidation_batch(conn, ME, limit=5)
    assert [c.id for c in batch.conversations] == [newer, older]
    assert [len(c.messages) for c in batch.conversations] == [2, 2]  # the recap turn is left out
    assert "Plan my garden." in batch.conversations[0].messages[0].content
    assert batch.conversations[0].messages[0].content.startswith("<stored_context ")
    assert batch.remaining == 0 and "last batch" in batch.note and "openmemore review" in batch.note


def test_done_conversations_aren_t_read_again_until_they_grow(conn):
    conversation_id = chat(conn, "a", [("user", "I like tea."), ("assistant", "Noted.")])
    batch = consolidation_batch(conn, ME, limit=5)
    result = propose_memories(conn, ME, [item("Likes tea.", conversation_id)], done_for(batch))
    assert result.conversations_done == 1
    assert consolidation_batch(conn, ME, limit=5).conversations == []

    chat(conn, "a", [("user", "I like tea."), ("assistant", "Noted."), ("user", "Also coffee in winter.")])
    [again] = consolidation_batch(conn, ME, limit=5).conversations
    assert [m.content.count("coffee") for m in again.messages] == [1] and again.earlier_messages == 2


def test_a_long_conversation_continues_in_the_next_batch(conn, monkeypatch):
    monkeypatch.setattr(consolidation, "CONVERSATION_BATCH_CHARS", 30)
    conversation_id = chat(conn, "a", [("user", "x" * 20), ("assistant", "y" * 20), ("user", "z" * 20)])
    first = consolidation_batch(conn, ME, limit=5)
    [c] = first.conversations
    assert len(c.messages) == 1 and c.more_in_next_batch and first.remaining == 1
    propose_memories(conn, ME, [], done_for(first))
    [c] = consolidation_batch(conn, ME, limit=5).conversations
    assert c.id == conversation_id and c.earlier_messages == 1 and "y" * 20 in c.messages[0].content


def test_marking_done_never_goes_past_the_conversation_or_backwards(conn):
    conversation_id = chat(conn, "a", [("user", "I like tea.")])
    propose_memories(conn, ME, [], [DoneConversation(conversation_id=conversation_id, through_message_id=10**9)])
    chat(conn, "a", [("user", "I like tea."), ("user", "And coffee.")])  # a later message isn't skipped
    [c] = consolidation_batch(conn, ME, limit=5).conversations
    assert len(c.messages) == 1
    propose_memories(conn, ME, [], [DoneConversation(conversation_id=conversation_id, through_message_id=1)])
    assert repo.get_conversation(conn, conversation_id).consolidated_through == c.messages[0].id - 1


def test_a_proposal_takes_its_sources_scope(conn):
    work = chat(conn, "w", [("user", "We deploy on Fridays.")], scope="work")
    result = propose_memories(conn, ME, [
        item("Deploys happen on Fridays.", work, kind="fact"),
        item("Prefers short answers."),
    ], [])
    first, second = (repo.get_proposal(conn, i) for i in result.proposed)
    assert (first.scope, first.conversation_ids, first.status) == ("work", [work], "pending")
    assert (second.scope, second.conversation_ids) == ("personal", [])
    assert repo.list_memories(conn, scopes=None) == []  # nothing is a memory until approved


def test_sources_in_different_scopes_are_refused(conn):
    work = chat(conn, "w", [("user", "a")], scope="work")
    home = chat(conn, "h", [("user", "b")])
    with pytest.raises(ToolError, match="different scopes"):
        propose_memories(conn, ME, [item("Mixed.", work, home)], [])


def test_duplicates_secrets_and_rejections(conn):
    insert(conn, "Prefers dark mode.", kind="preference")
    result = propose_memories(conn, ME, [
        item("Prefers dark mode."),  # already a memory
        item("Likes jazz."),
        item("likes jazz"),  # the same statement, normalized
        item("Token is ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789", kind="fact"),
    ], [])
    assert (len(result.proposed), result.skipped_duplicates, [r.index for r in result.rejected]) == (1, 2, [3])
    reject(conn, result.proposed[0])
    again = propose_memories(conn, ME, [item("Likes jazz.")], [])
    assert (again.proposed, again.skipped_duplicates) == ([], 1)  # rejected stays rejected


def test_a_near_duplicate_of_a_memory_is_noted(conn, fake_embedder):
    from openmemore.tools.memories import save_memories
    from openmemore.models import MemoryInput

    [existing] = save_memories(conn, ME, [MemoryInput(kind="preference", content="Prefers green tea in the morning")]).saved
    result = propose_memories(conn, ME, [item("prefers green tea in the morning daily")], [])
    [similar] = result.similar_to_memories
    assert (similar.memory_id, similar.proposal_id) == (existing, result.proposed[0])
    assert repo.get_proposal(conn, result.proposed[0]).similar_memory_id == existing


def test_deleting_a_conversation_deletes_its_undecided_proposals(conn):
    from openmemore.proposals import approve

    conversation_id = chat(conn, "a", [("user", "I like tea."), ("user", "And jazz.")])
    pending, rejected, approved = propose_memories(conn, ME, [
        item("Likes tea.", conversation_id), item("Likes jazz.", conversation_id), item("Likes cats.", conversation_id),
    ], []).proposed
    reject(conn, rejected)
    approve(conn, approved)
    with db.transaction(conn):
        repo.delete_conversation(conn, conversation_id)
    assert repo.get_proposal(conn, pending) is None and repo.get_proposal(conn, rejected) is None
    kept = repo.get_proposal(conn, approved)
    assert kept.status == "approved" and kept.conversation_ids == []


def test_turns_that_consolidated_are_recaps_in_captured_transcripts():
    assert is_recall_tool("mcp_openmemore_consolidation_batch")


# --- through MCP, with scopes ------------------------------------------------------


@pytest.fixture
def scoped(home):
    """'home-app' reads personal; 'work-app' reads and writes only work."""
    initialize(home)
    path = home / "config.toml"
    cfg = config.load_config(path)
    cfg.clients["work-app"] = config.ClientOverride(read_scopes=("work",), write_scope="work")
    config.save_config(path, cfg)
    with closing(db.connect(home / "openmemore.db")) as conn:
        personal = chat(conn, "p", [("user", "I'm vegetarian."), ("assistant", "Got it.")])
        work = chat(conn, "w", [("user", "Our API uses gRPC."), ("assistant", "Noted.")], scope="work")
    return home, personal, work


@pytest.mark.anyio
async def test_a_client_consolidates_only_what_it_can_read(scoped):
    home, personal, work = scoped
    async with connect(home, "work-app") as client:
        batch = (await client.call_tool("consolidation_batch", {})).structured_content
        assert [c["id"] for c in batch["conversations"]] == [work]
        for args in (
            {"items": [{"kind": "fact", "content": "Is vegetarian.", "conversation_ids": [personal]}], "done": []},
            {"items": [], "done": [{"conversation_id": personal, "through_message_id": 1}]},
        ):
            refused = await client.call_tool("propose_memories", args)
            missing = await client.call_tool("propose_memories", {
                k: [{**v[0], **({"conversation_ids": [999]} if "conversation_ids" in v[0] else {"conversation_id": 999})}]
                if v else v for k, v in args.items()})
            assert refused.is_error and refused.content[0].text == missing.content[0].text.replace("999", str(personal))
        result = await client.call_tool("propose_memories", {
            "items": [{"kind": "fact", "content": "The team's API uses gRPC.", "conversation_ids": [work]}],
            "done": [{"conversation_id": work, "through_message_id": batch["conversations"][0]["through_message_id"]}],
        })
        assert len(result.structured_content["proposed"]) == 1
        assert (await client.call_tool("consolidation_batch", {})).structured_content["conversations"] == []

    async with connect(home, "home-app") as client:  # the personal chat is still due for a client that can read it
        batch = (await client.call_tool("consolidation_batch", {})).structured_content
        assert [c["id"] for c in batch["conversations"]] == [personal]


@pytest.mark.anyio
async def test_the_consolidate_prompt(scoped):
    home, _, _ = scoped
    async with connect(home, "home-app") as client:
        text = (await client.get_prompt("consolidate", {"batches": "3"})).messages[0].content.text
    assert "consolidation_batch" in text and "propose_memories" in text and "at most 3 batches" in text
    assert "openmemore review" in text and "Don't call record_exchange" in text


def test_a_proposal_can_name_the_memory_it_updates(conn):
    old = insert(conn, "Choosing between two project ideas.", kind="project", project="Garden")
    work = insert(conn, "Team uses Go.", kind="fact", scope="work")
    [proposal_id] = propose_memories(conn, ME, [item("Picked idea 2.", kind="project", project="Garden", replaces=old)], []).proposed
    p = repo.get_proposal(conn, proposal_id)
    assert (p.replaces_memory_id, p.scope, p.similar_memory_id) == (old, "personal", None)

    home = chat(conn, "h", [("user", "b")])
    with pytest.raises(ToolError, match="different scopes"):
        propose_memories(conn, ME, [item("Uses Go and Rust.", home, kind="fact", replaces=work)], [])
    [into_work] = propose_memories(conn, ME, [item("Uses Go and Rust.", kind="fact", replaces=work)], []).proposed
    assert repo.get_proposal(conn, into_work).scope == "work"  # the memory's scope, not the write scope


def test_naming_a_memory_the_client_can_t_read_looks_like_naming_none(conn):
    hidden = insert(conn, "Personal thing.", kind="fact")
    at_work = caller("cursor-vscode", read=["work"], write="work")
    with pytest.raises(ToolError) as refused:
        propose_memories(conn, at_work, [item("x", kind="fact", replaces=hidden)], [])
    with pytest.raises(ToolError) as missing:
        propose_memories(conn, at_work, [item("x", kind="fact", replaces=9999)], [])
    assert str(refused.value).replace(str(hidden), "N") == str(missing.value).replace("9999", "N")


def test_naming_an_old_version_points_to_the_current_one(conn):
    from openmemore.tools.memories import update_memory

    old = insert(conn, "Likes tea.", kind="preference")
    new = update_memory(conn, ME, old, "Likes green tea.").new_id
    with pytest.raises(ToolError, match=f"already replaced by memory {new}"):
        propose_memories(conn, ME, [item("Likes oolong.", replaces=old)], [])
