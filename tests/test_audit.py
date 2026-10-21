import pytest

from openmemore import db
from openmemore.audit import audited


def test_successful_call_records_summary_and_count(tmp_path):
    conn = db.open_db(tmp_path / "t.db")
    with audited(conn, "cursor", "search_context") as call:
        call.summary = "q='python' " + "x" * 500
        call.result_count = 3

    row = conn.execute("SELECT * FROM audit_log").fetchone()
    assert (row["client"], row["tool"], row["result_count"]) == ("cursor", "search_context", 3)
    assert len(row["summary"]) == 200  # truncated, never the full payload
    assert row["duration_ms"] >= 0


def test_failed_call_is_still_logged_and_reraised(tmp_path):
    conn = db.open_db(tmp_path / "t.db")
    with pytest.raises(ValueError):
        with audited(conn, "cursor", "save_memories"):
            raise ValueError("boom")

    row = conn.execute("SELECT tool, summary FROM audit_log").fetchone()
    assert (row["tool"], row["summary"]) == ("save_memories", "error: ValueError")
