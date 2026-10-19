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


MIN_PARTIAL = 3
"""Shortest name matched as part of another ('renovation' in 'Kitchen Renovation 2026')."""


def match_projects(asked: str, labels: list[str]) -> list[str]:
    """The stored project labels a model's `project` means. Models guess the name ('cabinets',
    'renovation', 'openmcp'), so: the same name ignoring case; otherwise labels containing
    it, or contained in it."""
    key = asked.casefold().strip()
    same = [label for label in labels if label.casefold() == key]
    if same or len(key) < MIN_PARTIAL:
        return same
    return [
        label for label in labels
        if key in label.casefold() or (len(label) >= MIN_PARTIAL and label.casefold() in key)
    ]


def get_profile(conn: sqlite3.Connection, caller: Caller, project: str | None = None) -> Profile:
    """With `project`, each group holds that project's memories first, then general
    (untagged) ones, since general preferences still apply inside a project. A project
    that matches no stored one gives the whole profile, and the projects there are, rather
    than an empty answer a model might take as "nothing stored"."""
    note, known, projects = None, None, None
    if project is not None:
        labels = repo.project_labels(conn, caller.scopes.read_scopes)
        projects = match_projects(project, labels)
        if not projects:
            projects = None
            known = labels
            note = f"No memories are tagged with project {project!r}, so this is the whole profile. " + (
                "Projects with memories: " + ", ".join(repr(label) for label in labels) + "."
                if labels else "No memories are tagged with any project."
            )
        elif projects != [project]:
            note = "Showing project " + ", ".join(repr(label) for label in projects) + f" for {project!r}."

    def group(kind: Kind, updated_since: datetime | None = None) -> list[ProfileItem]:
        memories = repo.list_memories(
            conn,
            scopes=caller.scopes.read_scopes,
            kinds=[kind],
            project=projects,
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
        project_note=note,
        known_projects=known,
        unresolved_conflicts=repo.count_open_contradictions(conn, caller.scopes.read_scopes),
    )
