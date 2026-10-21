"""Relations between memories, applied through openmemore.relations
in the same transaction as their effect."""

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime

import pytest

from conftest import caller
from openmemore import db, relations, repo
from openmemore.ingest import store_conversation
from openmemore.models import MemoryInput, ParsedConversation, ParsedMessage, ProposalInput
from openmemore.proposals import Approval, approve, find_duplicates
from openmemore.tools.consolidation import propose_memories
from openmemore.tools.memories import forget, save_memories, update_memory

ME = caller("claude-ai", read=["personal"], write="personal")


def save(conn, content):
    return save_memories(conn, ME, [MemoryInput(kind="preference", content=content)]).saved[0]


def chat(conn, external_id, text):
    cid, _ = store_conversation(conn, ParsedConversation(
        provider="vscode-copilot", external_id=external_id,
        messages=[ParsedMessage(external_id="1", role="user", content=text)]), "personal")
    return cid


def shape(link):
    return (link.type, link.from_memory, link.to_memory, link.to_conversation, link.source)


def test_a_correction_is_a_supersedes_link(conn):
    tea = save(conn, "Prefers tea.")
    coffee = update_memory(conn, ME, tea, "Prefers coffee.").new_id
    [link] = relations.links(conn, tea)
    assert shape(link) == ("supersedes", coffee, tea, None, "claude-ai")
    assert relations.links(conn, coffee) == [link]


def test_an_approved_proposal_links_to_what_it_came_from_and_replaced(conn):
    tea = save(conn, "Prefers tea.")
    a, b = chat(conn, "a", "I've switched to coffee."), chat(conn, "b", "Coffee every morning now.")
    [pid] = propose_memories(conn, ME, [ProposalInput(kind="preference", content="Prefers coffee.",
                                                      conversation_ids=[a, b], replaces=tea)], []).proposed
    coffee = approve(conn, pid)
    assert [shape(l) for l in relations.links(conn, coffee)] == [
        ("supersedes", coffee, tea, None, "review"),
        ("derived_from", coffee, None, a, "claude-ai"),
        ("derived_from", coffee, None, b, "claude-ai"),
    ]
    assert relations.sources(conn, coffee) == [a, b]
    assert repo.get_memory(conn, coffee).metadata == {"proposal": pid}  # provenance lives in links now


def test_a_merge_supersedes_every_member(conn, fake_embedder):
    fake_embedder.vectors = {"Prefers dark mode.": [1.0, 0.0], "Likes a dark theme.": [0.99, 0.05]}
    first, second = save(conn, "Prefers dark mode."), save(conn, "Likes a dark theme.")
    find_duplicates(conn)
    merged = approve(conn, repo.list_proposals(conn)[0].id, Approval())
    assert sorted(l.to_memory for l in relations.links(conn, merged, "supersedes")) == [first, second]


def test_deleting_removes_links_with_what_they_point_at(conn):
    tea = save(conn, "Prefers tea.")
    cid = chat(conn, "a", "I like coffee now.")
    [pid] = propose_memories(conn, ME, [ProposalInput(kind="preference", content="Prefers coffee.",
                                                      conversation_ids=[cid], replaces=tea)], []).proposed
    coffee = approve(conn, pid)
    with db.transaction(conn):
        repo.delete_conversation(conn, cid)
    assert [l.type for l in relations.links(conn, coffee)] == ["supersedes"]  # the memory stays; its source is gone
    forget(conn, ME, coffee)
    assert conn.execute("SELECT COUNT(*) FROM memory_links").fetchone()[0] == 0


def test_a_link_points_at_a_memory_or_a_conversation_never_both(conn):
    tea, cid = save(conn, "Prefers tea."), chat(conn, "a", "hi")
    for to_memory, to_conversation, type_ in [(tea, cid, "derived_from"), (None, None, "supports"),
                                              (None, cid, "supersedes")]:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO memory_links (type, from_memory, to_memory, to_conversation, created_at) "
                         "VALUES (?, ?, ?, ?, '2026-01-01T00:00:00.000Z')", (type_, tea, to_memory, to_conversation))


def test_existing_history_becomes_links_when_upgrading(tmp_path, monkeypatch):
    path = tmp_path / "t.db"
    every = db.migrations()
    monkeypatch.setattr(db, "migrations", lambda: [m for m in every if m[0] < 17])
    conn = db.open_db(path)
    cid = chat(conn, "a", "I switched to coffee.")
    t = "2026-01-01T00:00:00.000Z"
    for memory_id, content, metadata in [(1, "Prefers tea.", {}),
                                         (2, "Prefers coffee.", {"proposal": 7, "conversations": [cid, 999], "replaced": [1]})]:
        conn.execute("INSERT INTO memories (id, kind, content, scope, source_client, content_hash, metadata, "
                     "created_at, updated_at) VALUES (?, 'preference', ?, 'personal', 'x', ?, ?, ?, ?)",
                     (memory_id, content, f"h{memory_id}", json.dumps(metadata), t, t))
    conn.execute("UPDATE memories SET superseded_by = 2 WHERE id = 1")
    monkeypatch.undo()
    db.migrate(conn)
    assert [shape(l) for l in relations.links(conn, 2)] == [
        ("supersedes", 2, 1, None, "migration"), ("derived_from", 2, None, cid, "migration")]
    assert repo.get_memory(conn, 2).metadata == {"proposal": 7}


def test_export_and_import_keep_where_memories_came_from(conn, tmp_path):
    from openmemore.export import export_document
    from openmemore.ingest.jobs import import_export

    cid = chat(conn, "chat-1", "I've switched to coffee.")
    [pid] = propose_memories(conn, ME, [ProposalInput(kind="preference", content="Prefers coffee.",
                                                      conversation_ids=[cid])], []).proposed
    approve(conn, pid)
    doc = export_document(conn)
    assert doc["memories"][0]["derived_from"] == [{"provider": "vscode-copilot", "id": "chat-1"}]
    path = tmp_path / "export.json"
    path.write_text(json.dumps(doc))
    with closing(db.open_db(tmp_path / "fresh.db")) as fresh:
        import_export(fresh, path, scope="personal")
        [m] = repo.list_memories(fresh, scopes=None)
        [source] = relations.sources(fresh, m.id)
        assert repo.get_conversation(fresh, source).external_id == "chat-1"
