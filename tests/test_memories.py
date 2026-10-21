import pytest
from mcp.server.mcpserver.exceptions import ToolError

from conftest import caller, insert
from openmemore import repo
from openmemore.models import MemoryInput
from openmemore.tools.memories import content_hash, forget, save_memories, update_memory
from openmemore.tools.search import search_context


def items(*contents, kind="preference", **extra):
    return [MemoryInput(kind=kind, content=c, **extra) for c in contents]


def active_count(conn):
    return conn.execute("SELECT COUNT(*) FROM memories WHERE superseded_by IS NULL").fetchone()[0]


# --- save_memories -------------------------------------------------------------------


def test_save_records_scope_source_and_fields(conn):
    result = save_memories(conn, caller("cursor", read=["work"], write="work"),
                           items("Uses ruff.", subject="tooling", project="OpenMemore", confidence=0.8))
    m = repo.get_memory(conn, result.saved[0])
    assert (m.kind, m.scope, m.source_client, m.subject, m.project, m.confidence) == (
        "preference", "work", "cursor", "tooling", "OpenMemore", 0.8,
    )
    assert m.created_at.tzinfo is not None and m.metadata == {}


def test_exact_duplicates_are_skipped_after_normalization(conn):
    save_memories(conn, caller(), items("Prefers Python."))
    result = save_memories(conn, caller(), items("  prefers   PYTHON ", "Prefers Python!"))
    assert result.saved == [] and result.skipped_duplicates == 2
    assert active_count(conn) == 1


def test_duplicates_within_one_batch_are_skipped(conn):
    result = save_memories(conn, caller(), items("Likes tea.", "likes tea"))
    assert len(result.saved) == 1 and result.skipped_duplicates == 1


def test_same_text_in_another_scope_is_not_a_duplicate(conn):
    insert(conn, "Prefers Python.", scope="work")
    result = save_memories(conn, caller(), items("Prefers Python."))
    assert len(result.saved) == 1


def test_secrets_are_rejected_by_index_and_never_stored(conn):
    secret = "deploy key sk-proj-AbCdEfGhIjKlMnOpQrStUvWx1234"
    result = save_memories(conn, caller(), [
        MemoryInput(kind="fact", content="Prefers Python."),
        MemoryInput(kind="fact", content=secret),
        MemoryInput(kind="fact", content="fine", subject="password: hunter22"),
    ])
    assert len(result.saved) == 1
    assert [r.index for r in result.rejected] == [1, 2]
    assert all("sk-proj" not in r.reason and "hunter22" not in r.reason for r in result.rejected)
    stored = " ".join(r[0] for r in conn.execute("SELECT content || coalesce(subject, '') FROM memories"))
    assert "sk-proj" not in stored and "hunter22" not in stored


def test_cannot_save_to_a_scope_the_client_does_not_write(conn):
    with pytest.raises(ToolError, match="can't write to scope 'work'"):
        save_memories(conn, caller(write="personal"), items("x"), scope="work")
    assert active_count(conn) == 0


# --- update_memory -------------------------------------------------------------------


def test_update_supersedes_and_keeps_history(conn):
    [old_id] = save_memories(conn, caller(), items("Prefers tabs.", subject="style", project="P")).saved

    result = update_memory(conn, caller("cursor"), old_id, "Prefers spaces.")

    old, new = repo.get_memory(conn, old_id), repo.get_memory(conn, result.new_id)
    assert result.previous_id == old_id
    assert old.superseded_by == new.id and new.active
    assert (new.subject, new.project, new.kind, new.scope) == ("style", "P", "preference", "personal")
    assert new.source_client == "cursor"
    assert [m.id for m in repo.memory_chain(conn, old_id)] == [old_id, new.id]


def test_update_rejects_secrets(conn):
    [mid] = save_memories(conn, caller(), items("x")).saved
    with pytest.raises(ToolError, match="not saved: contains a GitHub token"):
        update_memory(conn, caller(), mid, "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789")
    assert repo.get_memory(conn, mid).active


def test_update_of_a_replaced_version_points_to_the_current_one(conn):
    [mid] = save_memories(conn, caller(), items("v1")).saved
    new_id = update_memory(conn, caller(), mid, "v2").new_id
    with pytest.raises(ToolError, match=f"replaced by memory {new_id}"):
        update_memory(conn, caller(), mid, "v3")


def test_update_hides_unreadable_scopes_and_refuses_unwritable_ones(conn):
    personal = insert(conn, "personal thing", scope="personal")
    work = insert(conn, "work thing", scope="work")
    cursor = caller("cursor", read=["work"], write="work")
    desktop = caller("claude-desktop", read=["personal", "work"], write="personal")

    with pytest.raises(ToolError, match=f"memory {personal} not found"):
        update_memory(conn, cursor, personal, "changed")
    with pytest.raises(ToolError, match="can't write to"):
        update_memory(conn, desktop, work, "changed")
    with pytest.raises(ToolError, match="memory 999 not found"):
        update_memory(conn, desktop, 999, "changed")


def test_search_index_follows_updates(conn):
    [mid] = save_memories(conn, caller(), items("Prefers vim.")).saved
    update_memory(conn, caller(), mid, "Prefers helix.")
    assert search_context(conn, caller(), "vim").results == []
    assert [h.content.split("\n")[1] for h in search_context(conn, caller(), "helix").results] == ["Prefers helix."]


# --- forget --------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["oldest", "newest"])
def test_forget_deletes_the_whole_history_chain(conn, target):
    [v1] = save_memories(conn, caller(), items("v1")).saved
    v2 = update_memory(conn, caller(), v1, "v2").new_id
    v3 = update_memory(conn, caller(), v2, "v3").new_id
    keep = insert(conn, "unrelated")

    result = forget(conn, caller(), v1 if target == "oldest" else v3)

    assert result.deleted == 3
    ids = [r[0] for r in conn.execute("SELECT id FROM memories")]
    assert ids == [keep]
    assert conn.execute("SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'v1 OR v2 OR v3'").fetchone()[0] == 0


def test_forget_respects_scopes(conn):
    personal = insert(conn, "personal thing", scope="personal")
    with pytest.raises(ToolError, match="not found"):
        forget(conn, caller("cursor", read=["work"], write="work"), personal)
    assert repo.get_memory(conn, personal) is not None


def test_content_hash_normalization():
    assert content_hash("Prefers Python.") == content_hash("  prefers\tpython ")
    assert content_hash("Prefers Python") != content_hash("Prefers Rust")
