"""Contradictions. A proposal can say it disagrees with a memory;
the user settles it. Unresolved, neither side is current truth."""

from datetime import UTC, datetime

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from conftest import caller, insert
from openmemore import consolidate, db, relations, repo
from openmemore.models import ProposalInput
from openmemore.proposals import Approval, approve
from openmemore.tools.consolidation import propose_memories
from openmemore.tools.profile import get_profile
from openmemore.tools.search import search_context

ME = caller("claude-ai", read=["personal"], write="personal")
NOW = datetime(2026, 10, 19, tzinfo=UTC)


@pytest.fixture
def disputed(conn):
    """A memory, and a proposal that disagrees with it."""
    veg = insert(conn, "Is vegetarian.", kind="fact")
    [pid] = propose_memories(conn, ME, [ProposalInput(kind="fact", content="Eats fish on weekends.",
                                                      contradicts=veg)], []).proposed
    return veg, pid


def current_ids(conn, query="vegetarian fish weekends"):
    return [h.id for h in search_context(conn, ME, query).results if h.type == "memory"]


def test_a_proposal_records_what_it_disagrees_with(conn, disputed):
    veg, pid = disputed
    assert repo.get_proposal(conn, pid).contradicts_memory_id == veg


def test_not_sure_flags_the_conflict_and_hides_both_from_current_answers(conn, disputed):
    veg, pid = disputed
    fish = approve(conn, pid)  # the default: flag it
    assert current_ids(conn) == []
    profile = get_profile(conn, ME)
    assert profile.facts == [] and profile.unresolved_conflicts == 1
    statuses = {h.id: h.status for h in search_context(conn, ME, "vegetarian fish weekends", mode="history").results}
    assert statuses == {veg: "contradicted", fish: "contradicted"}
    [link] = relations.open_contradictions(conn)
    assert (link.from_memory, link.to_memory, link.source) == (fish, veg, "review")


def test_this_one_is_right_replaces_the_other(conn, disputed):
    veg, pid = disputed
    fish = approve(conn, pid, Approval(on_conflict="replace"))
    assert current_ids(conn) == [fish] and repo.get_memory(conn, veg).superseded_by == fish
    assert relations.open_contradictions(conn) == []


def test_both_true_keeps_both_current(conn, disputed):
    veg, pid = disputed
    fish = approve(conn, pid, Approval(on_conflict="both"))
    assert sorted(current_ids(conn)) == sorted([veg, fish]) and relations.open_contradictions(conn) == []


@pytest.mark.parametrize("resolution, kept", [("keep_from", "new"), ("keep_to", "old"), ("both", "both")])
def test_settling_a_flagged_conflict_later(conn, disputed, resolution, kept):
    veg, pid = disputed
    fish = approve(conn, pid)
    [link] = relations.open_contradictions(conn)
    with db.transaction(conn):
        settled = relations.resolve(conn, link.id, resolution, NOW, source="review")
    assert settled.resolution == resolution and settled.resolved_at is not None
    expected = {"new": [fish], "old": [veg], "both": sorted([veg, fish])}[kept]
    assert sorted(current_ids(conn)) == expected
    with pytest.raises(relations.ResolveError, match="already resolved"):
        relations.resolve(conn, link.id, "both", NOW)


def test_naming_a_memory_the_client_can_t_read_looks_like_naming_none(conn):
    hidden = insert(conn, "Personal thing.", kind="fact")
    at_work = caller("cursor-vscode", read=["work"], write="work")
    for memory_id in (hidden, 9999):
        with pytest.raises(ToolError, match="not found"):
            propose_memories(conn, at_work, [ProposalInput(kind="fact", content="x", contradicts=memory_id)], [])


def test_conflicts_count_only_in_the_clients_scopes(conn, disputed):
    veg, pid = disputed
    approve(conn, pid)
    assert get_profile(conn, caller("cursor-vscode", read=["work"], write="work")).unresolved_conflicts == 0


def test_the_nightly_model_can_raise_a_conflict_only_about_what_it_was_shown():
    text = ('{"memories": [{"kind": "fact", "content": "Eats fish.", "contradicts": 4},'
            ' {"kind": "fact", "content": "Has a cat.", "contradicts": 99}]}')
    shown, hidden = consolidate.parse(text, conversation_id=1, shown={4})[0]
    assert (shown.contradicts, hidden.contradicts) == (4, None)


def test_review_page_conflicts(home):
    from contextlib import closing

    from starlette.testclient import TestClient

    from openmemore.cli import initialize
    from openmemore.review import CHANGE_HEADER, build_app

    initialize(home)
    with closing(db.connect(home / "openmemore.db")) as conn:
        veg = insert(conn, "Is vegetarian.", kind="fact")
        [pid] = propose_memories(conn, ME, [ProposalInput(kind="fact", content="Eats fish.", contradicts=veg)], []).proposed
    page = TestClient(build_app(home, "t", 1), base_url="http://127.0.0.1:1")
    page.get("/?token=t")
    [view] = page.get("/api/proposals").json()
    assert view["contradicts"]["id"] == veg
    page.post(f"/api/proposals/{pid}/approve", json={"on_conflict": "flag"}, headers={CHANGE_HEADER: "1"})
    assert page.get("/api/state").json()["conflicts"] == 1
    [conflict] = page.get("/api/conflicts").json()
    assert {conflict["from"]["content"], conflict["to"]["content"]} == {"Is vegetarian.", "Eats fish."}
    r = page.post(f"/api/conflicts/{conflict['id']}/resolve", json={"resolution": "keep_to"}, headers={CHANGE_HEADER: "1"})
    assert r.status_code == 200 and page.get("/api/conflicts").json() == []
    again = page.post(f"/api/conflicts/{conflict['id']}/resolve", json={"resolution": "both"}, headers={CHANGE_HEADER: "1"})
    assert again.status_code == 400 and "already resolved" in again.json()["error"]
    with closing(db.connect(home / "openmemore.db")) as conn:
        assert [r[0] for r in conn.execute("SELECT tool FROM audit_log WHERE client = 'review'")] == [
            "approve", "resolve conflict", "resolve conflict"]
