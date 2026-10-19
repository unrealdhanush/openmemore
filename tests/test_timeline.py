"""SPEC.md M7.7 part 1: two timelines on memories. When something was true in the user's
life (valid_from / valid_to) is kept apart from when OpenMCP learned it, and learned
otherwise (created_at / superseded_at)."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from conftest import caller
from openmcp import db, repo
from openmcp.models import MemoryInput, ProposalInput
from openmcp.proposals import Approval, approve, find_duplicates
from openmcp.tools.consolidation import propose_memories
from openmcp.tools.memories import save_memories, update_memory
from openmcp.tools.profile import get_profile
from openmcp.tools.search import search_context

ME = caller("claude-ai", read=["personal"], write="personal")
JUNE = datetime(2026, 6, 1, tzinfo=UTC)


def save(conn, content, **when):
    [memory_id] = save_memories(conn, ME, [MemoryInput(kind="preference", content=content, **when)]).saved
    return memory_id


@pytest.mark.parametrize("given, expected", [
    ("2024", datetime(2024, 1, 1, tzinfo=UTC)),
    ("2024-03", datetime(2024, 3, 1, tzinfo=UTC)),
    ("2024-03-15", datetime(2024, 3, 15, tzinfo=UTC)),
    ("2024-03-15T10:30:00+02:00", datetime(2024, 3, 15, 8, 30, tzinfo=UTC)),
])
def test_dates_as_people_give_them(given, expected):
    assert MemoryInput(kind="fact", content="x", valid_from=given).valid_from == expected


def test_a_memory_can_t_end_before_it_begins():
    with pytest.raises(ValidationError, match="before valid_from"):
        MemoryInput(kind="fact", content="x", valid_from="2024-05", valid_to="2024-01")


def test_a_new_memory_is_true_from_when_it_was_learned_unless_told(conn):
    plain = repo.get_memory(conn, save(conn, "Prefers tea."))
    assert plain.valid_from == plain.created_at and plain.valid_to is None
    dated = repo.get_memory(conn, save(conn, "Lives in Berlin.", valid_from="2022-09"))
    assert dated.valid_from == datetime(2022, 9, 1, tzinfo=UTC)


def test_something_that_ended_is_history_not_current(conn):
    ended = save(conn, "Lived in Paris.", valid_from="2015", valid_to="2020")
    current = save(conn, "Lives in Berlin.", valid_from="2020")
    assert [m.id for m in repo.list_memories(conn, scopes=None)] == [current]
    assert {m.id for m in repo.list_memories(conn, scopes=None, include_superseded=True)} == {ended, current}
    assert [h.id for h in search_context(conn, ME, "Paris Berlin lives").results] == [current]
    assert [i.id for i in get_profile(conn, ME).preferences] == [current]
    assert not repo.get_memory(conn, ended).current() and repo.get_memory(conn, current).current()


def test_a_correction_closes_the_old_memory_where_the_new_one_starts(conn):
    tea = save(conn, "Prefers tea.", valid_from="2025-01")
    coffee = update_memory(conn, ME, tea, "Prefers coffee.", valid_from=JUNE).new_id
    old, new = repo.get_memory(conn, tea), repo.get_memory(conn, coffee)
    assert (old.superseded_by, old.valid_to, new.valid_from) == (coffee, JUNE, JUNE)
    assert old.superseded_at is not None and old.superseded_at >= old.created_at  # learned now, true since June


def test_a_correction_never_ends_the_old_memory_before_it_began(conn):
    tea = save(conn, "Prefers tea.", valid_from="2026-08")
    update_memory(conn, ME, tea, "Prefers coffee.", valid_from=JUNE)  # dated before tea was even true
    old = repo.get_memory(conn, tea)
    assert old.valid_to == old.valid_from == datetime(2026, 8, 1, tzinfo=UTC)


def test_a_proposal_is_true_from_its_conversation_not_from_approval(conn):
    from openmcp.ingest import store_conversation
    from openmcp.models import ParsedConversation, ParsedMessage

    tea = save(conn, "Prefers tea.", valid_from="2026-01")
    conversation_id, _ = store_conversation(conn, ParsedConversation(
        provider="vscode-copilot", external_id="c1", updated_at=JUNE,
        messages=[ParsedMessage(external_id="1", role="user", content="I've switched to coffee.", created_at=JUNE)],
    ), "personal")
    [proposal_id] = propose_memories(conn, ME, [ProposalInput(
        kind="preference", content="Prefers coffee.", conversation_ids=[conversation_id], replaces=tea)], []).proposed
    coffee = repo.get_memory(conn, approve(conn, proposal_id))  # approved now, months later
    assert coffee.valid_from == JUNE and repo.get_memory(conn, tea).valid_to == JUNE


def test_a_model_given_date_wins(conn):
    [proposal_id] = propose_memories(conn, ME, [ProposalInput(
        kind="fact", content="Started running.", valid_from="2025-04")], []).proposed
    assert repo.get_memory(conn, approve(conn, proposal_id)).valid_from == datetime(2025, 4, 1, tzinfo=UTC)


def test_a_merge_is_true_since_the_earliest_version(conn, fake_embedder):
    fake_embedder.vectors = {"Prefers dark mode.": [1.0, 0.0], "Likes a dark theme.": [0.99, 0.05]}
    first = save(conn, "Prefers dark mode.", valid_from="2024-02")
    save(conn, "Likes a dark theme.", valid_from="2025-07")
    find_duplicates(conn)
    [merge] = repo.list_proposals(conn)
    merged = repo.get_memory(conn, approve(conn, merge.id, Approval(content="Prefers dark mode everywhere.")))
    assert merged.valid_from == datetime(2024, 2, 1, tzinfo=UTC)
    assert repo.get_memory(conn, first).superseded_by == merged.id


def test_export_and_import_keep_the_timeline(conn, tmp_path):
    import json
    from contextlib import closing

    from openmcp.export import export_document
    from openmcp.ingest.jobs import import_export

    save(conn, "Lived in Paris.", valid_from="2015", valid_to="2020")
    doc = export_document(conn)
    assert (doc["memories"][0]["valid_from"], doc["memories"][0]["valid_to"]) == (
        "2015-01-01T00:00:00.000Z", "2020-01-01T00:00:00.000Z")
    path = tmp_path / "export.json"
    path.write_text(json.dumps(doc))
    with closing(db.open_db(tmp_path / "fresh.db")) as fresh:
        import_export(fresh, path, scope="personal")
        [m] = repo.list_memories(fresh, scopes=None, include_superseded=True)
        assert (m.valid_from, m.valid_to) == (datetime(2015, 1, 1, tzinfo=UTC), datetime(2020, 1, 1, tzinfo=UTC))


def test_existing_history_gets_a_timeline_when_upgrading(tmp_path, monkeypatch):
    """Memories saved before migration 016: true from when they were saved, and a
    replaced one until its replacement was saved."""
    path = tmp_path / "t.db"
    every = db.migrations()
    monkeypatch.setattr(db, "migrations", lambda: [m for m in every if m[0] < 16])
    conn = db.open_db(path)
    t1, t2, t3 = "2026-01-01T00:00:00.000Z", "2026-03-01T00:00:00.000Z", "2026-03-02T00:00:00.000Z"
    conn.execute("INSERT INTO memories (id, kind, content, scope, source_client, content_hash, created_at, updated_at) "
                 "VALUES (1, 'preference', 'Prefers tea.', 'personal', 'x', 'h1', ?, ?)", (t1, t1))
    conn.execute("INSERT INTO memories (id, kind, content, scope, source_client, content_hash, created_at, updated_at) "
                 "VALUES (2, 'preference', 'Prefers coffee.', 'personal', 'x', 'h2', ?, ?)", (t2, t2))
    conn.execute("UPDATE memories SET superseded_by = 2, updated_at = ? WHERE id = 1", (t3,))
    monkeypatch.undo()
    db.migrate(conn)
    old, new = repo.get_memory(conn, 1), repo.get_memory(conn, 2)
    assert (old.valid_from, old.valid_to, old.superseded_at) == (
        datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC), datetime(2026, 3, 2, tzinfo=UTC))
    assert (new.valid_from, new.valid_to) == (datetime(2026, 3, 1, tzinfo=UTC), None)


def test_the_library_takes_dates(home):
    from openmcp import Memory
    from openmcp.cli import initialize

    initialize(home)
    memory = Memory("agent", home=home)
    old = memory.save("Lived in Paris.", kind="fact", valid_from="2015", valid_to="2020")
    assert memory.search("Paris", chats=False) == []  # ended: not current
    tea = memory.save("Prefers tea.", kind="preference", valid_from="2025")
    memory.update(tea, "Prefers coffee.", valid_from="2026-06")
    with db.connect(home / "openmcp.db") as conn:
        assert repo.get_memory(conn, tea).valid_to == JUNE and repo.get_memory(conn, old).valid_to is not None
