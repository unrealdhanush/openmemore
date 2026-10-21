"""Audit logging: every tool call writes one audit_log row, including its duration."""

from __future__ import annotations

import math
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

from openmemore import repo
from openmemore.models import SUMMARY_MAX, AuditEntry


@dataclass
class AuditCall:
    """Filled in by the tool while it runs; written to the log when it finishes."""

    summary: str | None = None
    result_count: int | None = None


@dataclass(frozen=True)
class ToolStats:
    tool: str
    calls: int
    p50_ms: float
    p95_ms: float
    max_ms: float


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100) of a non-empty list."""
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def stats(durations: dict[str, list[float]]) -> list[ToolStats]:
    """p50/p95 per tool, for checking the performance budget (scripts/benchmark.py)."""
    return [
        ToolStats(tool, len(ms), percentile(ms, 50), percentile(ms, 95), max(ms))
        for tool, ms in sorted(durations.items())
        if ms
    ]


@contextmanager
def audited(conn: sqlite3.Connection, client: str, tool: str, app: str | None = None) -> Iterator[AuditCall]:
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
            app=app[:SUMMARY_MAX] if app else None,
        )
        repo.insert_audit(conn, entry)
