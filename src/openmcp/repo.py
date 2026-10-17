"""Repository: the only module that reads and writes application data with SQL.

Functions take a connection and Pydantic models, and return models (or ids).
Schema setup and migrations live in db.py.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from openmcp.models import AuditEntry


def to_iso(dt: datetime) -> str:
    """ISO 8601 UTC with millisecond precision and a Z suffix, so stored timestamps sort as text."""
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def insert_audit(conn: sqlite3.Connection, entry: AuditEntry) -> int:
    cur = conn.execute(
        "INSERT INTO audit_log (ts, client, tool, summary, result_count, duration_ms) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (to_iso(entry.ts), entry.client, entry.tool, entry.summary, entry.result_count, entry.duration_ms),
    )
    return cur.lastrowid
