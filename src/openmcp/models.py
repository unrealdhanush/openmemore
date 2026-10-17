"""Pydantic models for every boundary: MCP tool arguments/results, CLI inputs, stored rows.

Nothing reaches the repository (repo.py) without passing through one of these.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints

ClientName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"),
]
"""Name a client connects as (`openmcp serve --client <name>`); recorded in the audit log."""

Scope = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]

SUMMARY_MAX = 200


class AuditEntry(BaseModel):
    """One row of audit_log: a single tool call by a single client."""

    model_config = ConfigDict(frozen=True)

    ts: AwareDatetime
    client: str
    tool: str
    summary: str | None = Field(default=None, max_length=SUMMARY_MAX)
    result_count: int | None = Field(default=None, ge=0)
    duration_ms: float = Field(ge=0)


class PingResult(BaseModel):
    """Result of the `ping` tool."""

    ok: bool
    version: str
    client: str
    read_scopes: list[str]
    write_scope: str
    schema_version: int
