"""SPEC.md M5: approving and rejecting proposals, merging near-duplicates, and the
`openmcp review` page, which only answers its own token, on its own host."""

from contextlib import closing

import pytest
from starlette.testclient import TestClient

from conftest import caller, connect
from openmcp import db, repo
from openmcp.cli import initialize
from openmcp.models import MemoryInput, ProposalInput
from openmcp.proposals import Approval, ProposalError, approve, find_duplicates, reject
from openmcp.review import CHANGE_HEADER, build_app
from openmcp.tools.consolidation import propose_memories
from openmcp.tools.memories import forget, save_memories, update_memory

ME = caller("claude-ai", read=["personal", "work"], write="personal")


def propose(conn, *contents, kind="preference", **extra):
    items = [ProposalInput(kind=kind, content=c, **extra) for c in contents]
    return propose_memories(conn, ME, items, []).proposed


def test_approving_makes_a_memory_with_its_provenance(conn):
    [proposal_id] = propose(conn, "Prefers tabs.", project="Garden", reason="said so twice")
    memory_id = approve(conn, proposal_id)
    m = repo.get_memory(conn, memory_id)
    assert (m.content, m.project, m.scope, m.source_client) == ("Prefers tabs.", "Garden", "personal", "claude-ai")
    assert m.metadata == {"proposal": proposal_id}
    p = repo.get_proposal(conn, proposal_id)
    assert (p.status, p.memory_id) == ("approved", memory_id)
    with pytest.raises(ProposalError, match="already approved"):
        approve(conn, proposal_id)


def test_approving_with_edits_and_the_secret_filter(conn):
    [proposal_id] = propose(conn, "Prefers tabs.")
    with pytest.raises(ProposalError, match="not saved"):
        approve(conn, proposal_id, Approval(content="Token ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789", kind="fact"))
    memory_id = approve(conn, proposal_id, Approval(content="Prefers tabs, width 4.", subject="indentation", project=None))
    m = repo.get_memory(conn, memory_id)
    assert (m.content, m.subject, m.kind) == ("Prefers tabs, width 4.", "indentation", "preference")


def test_approving_as_a_replacement_for_the_similar_memory(conn, fake_embedder):
    [old] = save_memories(conn, ME, [MemoryInput(kind="preference", content="Prefers green tea in the morning")]).saved
    [proposal_id] = propose(conn, "prefers green tea in the morning daily")
    new = approve(conn, proposal_id, Approval(replace_similar=True))
    assert repo.get_memory(conn, old).superseded_by == new
    assert repo.get_memory(conn, new).metadata["replaced"] == [old]
    assert [m.id for m in repo.memory_chain(conn, new)] == [old, new]


def test_approving_what_was_saved_meanwhile_links_the_existing_memory(conn):
    [proposal_id] = propose(conn, "Likes jazz.")
    [saved] = save_memories(conn, ME, [MemoryInput(kind="preference", content="Likes jazz.")]).saved
    assert approve(conn, proposal_id) == saved
    assert len(repo.list_memories(conn, scopes=None)) == 1


def test_rejecting(conn):
    [proposal_id] = propose(conn, "Likes jazz.")
    reject(conn, proposal_id)
    assert repo.get_proposal(conn, proposal_id).status == "rejected"
    assert repo.list_memories(conn, scopes=None) == []
    with pytest.raises(ProposalError, match="already rejected"):
        approve(conn, proposal_id)


@pytest.fixture
def near_duplicates(conn, fake_embedder):
    """Three memories saying nearly the same thing, and one that doesn't."""
    fake_embedder.vectors = {
        "Prefers dark mode.": [1.0, 0.0], "Likes a dark theme.": [0.99, 0.05], "Uses dark mode everywhere.": [0.98, 0.1],
        "Lives in Lisbon.": [0.0, 1.0],
    }
    ids = save_memories(conn, ME, [MemoryInput(kind="preference", content=c) for c in
                                   ("Prefers dark mode.", "Likes a dark theme.", "Uses dark mode everywhere.")]).saved
    save_memories(conn, ME, [MemoryInput(kind="fact", content="Lives in Lisbon.")])
    return ids


def test_near_duplicates_become_one_merge_proposal(conn, near_duplicates):
    assert find_duplicates(conn) == 1
    [merge] = repo.list_proposals(conn)
    assert (merge.action, merge.memory_ids, merge.source_client) == ("merge", near_duplicates, "openmcp")
    assert merge.content == "Uses dark mode everywhere."  # the newest one's text
    assert find_duplicates(conn) == 0  # already waiting for review

    merged = approve(conn, merge.id, Approval(content="Prefers dark mode everywhere."))
    assert all(repo.get_memory(conn, i).superseded_by == merged for i in near_duplicates)
    assert [m.id for m in repo.memory_chain(conn, merged)] == [*near_duplicates, merged]
    assert {m.content for m in repo.list_memories(conn, scopes=None)} == {"Prefers dark mode everywhere.", "Lives in Lisbon."}
    assert forget(conn, ME, merged).deleted == 4  # the whole history goes


def test_a_rejected_merge_isn_t_proposed_again(conn, near_duplicates):
    find_duplicates(conn)
    reject(conn, repo.list_proposals(conn)[0].id)
    assert find_duplicates(conn) == 0


def test_a_merge_of_changed_memories_can_t_be_approved(conn, near_duplicates):
    find_duplicates(conn)
    [merge] = repo.list_proposals(conn)
    update_memory(conn, ME, near_duplicates[0], "Prefers light mode now.")
    with pytest.raises(ProposalError, match="changed since"):
        approve(conn, merge.id)


def test_forgetting_a_memory_drops_merges_that_include_it(conn, near_duplicates):
    find_duplicates(conn)
    forget(conn, ME, near_duplicates[1])
    assert repo.list_proposals(conn) == []


def test_duplicates_need_vectors(conn):
    save_memories(conn, ME, [MemoryInput(kind="preference", content=c) for c in ("Prefers dark mode.", "Prefers dark mode!!")])
    assert find_duplicates(conn) == 0  # not embedded: nothing to compare


@pytest.mark.anyio
async def test_an_approved_proposal_reaches_other_clients(home):
    initialize(home)
    async with connect(home, "claude-ai") as client:
        result = await client.call_tool("propose_memories", {
            "items": [{"kind": "preference", "content": "Prefers metric units."}], "done": []})
        [proposal_id] = result.structured_content["proposed"]
    async with connect(home, "cursor-vscode") as client:
        assert (await client.call_tool("get_profile", {})).structured_content["preferences"] == []
    with closing(db.connect(home / "openmcp.db")) as conn:
        approve(conn, proposal_id)
    async with connect(home, "cursor-vscode") as client:
        [pref] = (await client.call_tool("get_profile", {})).structured_content["preferences"]
    assert "Prefers metric units." in pref["content"] and pref["source"] == "claude-ai"


# --- the page --------------------------------------------------------------------------

PORT, TOKEN = 54321, "test-token"
CHANGE = {CHANGE_HEADER: "1"}


@pytest.fixture
def page(home):
    initialize(home)
    client = TestClient(build_app(home, TOKEN, PORT), base_url=f"http://127.0.0.1:{PORT}", follow_redirects=False)
    return client


def signed_in(page):
    response = page.get(f"/?token={TOKEN}")
    assert response.status_code == 303 and response.headers["location"] == "/"
    assert "httponly" in response.headers["set-cookie"].lower() and "samesite=strict" in response.headers["set-cookie"].lower()
    return page


def test_the_page_needs_its_token(page):
    assert page.get("/").status_code == 403
    assert page.get("/api/state").status_code == 403
    assert page.get("/?token=wrong").status_code == 403
    signed_in(page)
    response = page.get("/")
    assert response.status_code == 200 and "OpenMCP review" in response.text
    assert "default-src 'none'" in response.headers["content-security-policy"]


def test_the_page_refuses_other_hosts(page):
    signed_in(page)
    assert page.get("/api/state", headers={"host": "evil.example"}).status_code == 403
    assert page.get("/api/state", headers={"host": f"localhost:{PORT}"}).status_code == 200


def test_changes_need_the_header(page, home):
    signed_in(page)
    with closing(db.connect(home / "openmcp.db")) as conn:
        [proposal_id] = propose(conn, "Likes jazz.")
    assert page.post(f"/api/proposals/{proposal_id}/approve", json={}).status_code == 403
    response = page.post(f"/api/proposals/{proposal_id}/approve", json={}, headers=CHANGE)
    assert response.status_code == 200
    again = page.post(f"/api/proposals/{proposal_id}/approve", json={}, headers=CHANGE)
    assert again.status_code == 400 and "already approved" in again.json()["error"]


def test_reviewing_through_the_page(page, home):
    signed_in(page)
    with closing(db.connect(home / "openmcp.db")) as conn:
        from openmcp.ingest import store_conversation
        from openmcp.models import ParsedConversation, ParsedMessage

        conversation_id, _ = store_conversation(conn, ParsedConversation(
            provider="cursor-vscode", external_id="c1", title="Units", metadata={},
            messages=[ParsedMessage(external_id="1", role="user", content="Use metric, always.")]), "personal")
        a, b, c = propose(conn, "Prefers metric units.", "Likes jazz.", "Has a cat.", conversation_ids=[conversation_id])

    proposals = page.get("/api/proposals").json()
    assert [p["id"] for p in proposals] == [a, b, c]
    assert proposals[0]["conversations"] == [{"id": conversation_id, "title": "Units", "source": "cursor-vscode"}]
    assert page.get(f"/api/conversations/{conversation_id}").json()["messages"][0]["content"] == "Use metric, always."
    assert page.get("/api/state").json()["pending_proposals"] == 3

    edited = page.post(f"/api/proposals/{a}/approve", json={"content": "Always uses metric units."}, headers=CHANGE)
    memory_id = edited.json()["memory_id"]
    result = page.post("/api/proposals/decide", json={"ids": [b, c], "decision": "reject"}, headers=CHANGE).json()
    assert result == {"done": 2, "failed": []}
    assert page.get("/api/proposals").json() == []
    assert {p["id"] for p in page.get("/api/proposals?status=rejected").json()} == {b, c}

    [m] = page.get("/api/memories?q=METRIC").json()
    assert (m["id"], m["content"]) == (memory_id, "Always uses metric units.")
    assert page.get("/api/memories?q=nothing-like-this").json() == []
    updated = page.post(f"/api/memories/{memory_id}", json={"content": "Uses metric units."}, headers=CHANGE).json()
    assert [h["id"] for h in page.get(f"/api/memories/{updated['new_id']}/history").json()] == [memory_id, updated["new_id"]]
    assert page.request("DELETE", f"/api/memories/{updated['new_id']}", headers=CHANGE).json() == {"deleted": 2}
    assert page.post("/api/memories/999", json={"content": "x"}, headers=CHANGE).status_code == 404
    assert page.post("/api/memories/1", json={"content": ""}, headers=CHANGE).status_code == 400

    audit = list(db.connect(home / "openmcp.db").execute("SELECT tool, summary FROM audit_log WHERE client = 'review'"))
    assert [tool for tool, _ in audit] == ["approve", "reject all", "update_memory", "forget", "update_memory"]
    assert audit[-1][1].startswith("error")  # failed attempts are logged too


def test_find_duplicates_from_the_page(page, home, fake_embedder):
    signed_in(page)
    fake_embedder.vectors = {"Prefers dark mode.": [1.0, 0.0], "Likes a dark theme.": [0.99, 0.05]}
    with closing(db.connect(home / "openmcp.db")) as conn:
        save_memories(conn, ME, [MemoryInput(kind="preference", content=c) for c in fake_embedder.vectors])
    assert page.post("/api/duplicates", headers=CHANGE).json() == {"proposed": 1}
    [merge] = page.get("/api/proposals").json()
    assert [m["content"] for m in merge["merging"]] == ["Prefers dark mode.", "Likes a dark theme."]
