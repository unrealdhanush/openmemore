"""get_profile: a deterministic summary of active memories, grouped by kind."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from openmcp import repo
from openmcp.models import ACTIVE_PROJECT_DAYS, PROFILE_GROUP_CAP, Kind, Memory, Profile, ProfileItem
from openmcp.safety.wrap import wrap
from openmcp.tools import Caller


def _item(m: Memory) -> ProfileItem:
    return ProfileItem(
        id=m.id,
        content=wrap(m.content, source=m.source_client, id=m.id),
        subject=m.subject,
        project=m.project,
        source=m.source_client,
        updated_at=m.updated_at,
    )


def get_profile(conn: sqlite3.Connection, caller: Caller, project: str | None = None) -> Profile:
    """With `project`, each group holds that project's memories first, then general
    (untagged) ones, since general preferences still apply inside a project."""

    def group(kind: Kind, updated_since: datetime | None = None) -> list[ProfileItem]:
        memories = repo.list_memories(
            conn,
            scopes=caller.scopes.read_scopes,
            kinds=[kind],
            project=project,
            include_general=True,
            updated_since=updated_since,
            limit=PROFILE_GROUP_CAP,
        )
        return [_item(m) for m in memories]

    return Profile(
        preferences=group("preference"),
        facts=group("fact"),
        active_projects=group("project", datetime.now(UTC) - timedelta(days=ACTIVE_PROJECT_DAYS)),
        recent_decisions=group("decision"),
    )
