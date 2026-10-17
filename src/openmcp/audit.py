"""Audit logging: every tool call writes one audit_log row, including its duration."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

from openmcp import repo
from openmcp.models import SUMMARY_MAX, AuditEntry


@dataclass
class AuditCall:
    """Filled in by the tool while it runs; written to the log when it finishes."""

    summary: str | None = None
    result_count: int | None = None


@contextmanager
def audited(conn: sqlite3.Connection, client: str, tool: str) -> Iterator[AuditCall]:
    """Time a tool call and log it, whether it succeeds or raises."""
    call = AuditCall()
    ts = datetime.now(UTC)
    start = time.perf_counter()
    try:
        yield call
    except Exception as e:
        call.summary = f"error: {type(e).__name__}"
        raise
    finally:
        summary = call.summary[:SUMMARY_MAX] if call.summary else None
        entry = AuditEntry(
            ts=ts,
            client=client,
            tool=tool,
            summary=summary,
            result_count=call.result_count,
            duration_ms=(time.perf_counter() - start) * 1000,
        )
        repo.insert_audit(conn, entry)
