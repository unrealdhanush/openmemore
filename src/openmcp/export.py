"""`openmcp export` (SPEC.md M4): everything stored, in OpenMCP's own format
(openmcp.ingest.openmcp_format), which `openmcp import` reads back.

Memories include their history (earlier versions, linked by `superseded_by`);
conversations include every message with its scope. Embeddings and the audit log are
left out: embeddings are rebuilt from the text, and the log is about this machine.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from openmcp import repo
from openmcp.ingest.openmcp_format import FORMAT, VERSION

EVERYTHING = 1_000_000_000


def _iso(value: datetime | None) -> str | None:
    return repo.to_iso(value) if value else None


def export_document(conn: sqlite3.Connection, scopes: Sequence[str] | None = None) -> dict[str, Any]:
    """The whole store (or these scopes) as an openmcp.export document, oldest first."""
    memories = sorted(
        repo.list_memories(conn, scopes=scopes, include_superseded=True, limit=EVERYTHING), key=lambda m: m.id
    )
    conversations = sorted(repo.list_conversations(conn, scopes=scopes, limit=EVERYTHING), key=lambda c: c.id)
    return {
        "format": FORMAT,
        "version": VERSION,
        "exported_at": repo.to_iso(datetime.now(UTC)),
        "memories": [
            {
                "id": m.id, "kind": m.kind, "content": m.content, "subject": m.subject, "project": m.project,
                "scope": m.scope, "source": m.source_client, "confidence": m.confidence, "metadata": m.metadata,
                "created_at": _iso(m.created_at), "updated_at": _iso(m.updated_at), "superseded_by": m.superseded_by,
                "valid_from": _iso(m.valid_from), "valid_to": _iso(m.valid_to), "superseded_at": _iso(m.superseded_at),
            }
            for m in memories
        ],
        "conversations": [
            {
                "provider": c.provider, "id": c.external_id, "title": c.title, "scope": c.scope,
                "created_at": _iso(c.created_at), "updated_at": _iso(c.updated_at), "metadata": c.metadata,
                "messages": [
                    {
                        # Messages captured or imported keep their source's id; recorded ones are
                        # identified by their content hash, which importing again matches.
                        "id": m.external_id or f"sha256:{m.content_hash}", "role": m.role, "content": m.content,
                        "created_at": _iso(m.created_at), "recall": m.recall,
                    }
                    for m in repo.conversation_messages(conn, c.id)
                ],
            }
            for c in conversations
        ],
    }
