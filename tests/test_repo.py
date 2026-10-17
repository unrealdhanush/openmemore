from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from openmcp import db, repo
from openmcp.models import AuditEntry


def test_insert_audit_round_trip(tmp_path):
    conn = db.open_db(tmp_path / "t.db")
    entry = AuditEntry(
        ts=datetime(2026, 10, 17, 9, 30, 0, 123456, tzinfo=timezone(timedelta(hours=-7))),
        client="cursor",
        tool="ping",
        summary="ok",
        result_count=1,
        duration_ms=1.5,
    )

    row_id = repo.insert_audit(conn, entry)

    row = conn.execute("SELECT * FROM audit_log WHERE id = ?", (row_id,)).fetchone()
    assert row["ts"] == "2026-10-17T16:30:00.123Z"  # normalized to UTC
    assert (row["client"], row["tool"], row["summary"], row["result_count"], row["duration_ms"]) == (
        "cursor", "ping", "ok", 1, 1.5,
    )


@pytest.mark.parametrize("bad", [
    {"ts": datetime(2026, 1, 1)},                 # naive datetime: timezone required
    {"duration_ms": -1},
    {"summary": "x" * 201},                       # summaries stay short, never full payloads
])
def test_audit_entry_validation(bad):
    fields = {"ts": datetime.now(UTC), "client": "c", "tool": "t", "duration_ms": 0} | bad
    with pytest.raises(ValidationError):
        AuditEntry(**fields)
