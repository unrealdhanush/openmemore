from datetime import UTC, datetime, timedelta

from conftest import caller, insert
from openmemore.models import PROFILE_GROUP_CAP
from openmemore.safety.wrap import NOTICE
from openmemore.tools.profile import get_profile


def ids(items):
    return [i.id for i in items]


def test_groups_active_memories_by_kind(conn):
    pref = insert(conn, "Prefers Python.", kind="preference")
    fact = insert(conn, "Lives in Berlin.", kind="fact")
    proj = insert(conn, "OpenMemore: M1 underway.", kind="project", project="OpenMemore")
    dec = insert(conn, "Chose SQLite.", kind="decision")
    insert(conn, "Random note.", kind="note")

    profile = get_profile(conn, caller())

    assert profile.notice == NOTICE
    assert (ids(profile.preferences), ids(profile.facts), ids(profile.active_projects), ids(profile.recent_decisions)) == (
        [pref], [fact], [proj], [dec],
    )
    assert profile.preferences[0].content.startswith("<stored_context ")


def test_active_projects_are_those_updated_in_the_last_30_days(conn):
    now = datetime.now(UTC)
    recent = insert(conn, "recent project", kind="project", when=now - timedelta(days=29))
    insert(conn, "stale project", kind="project", when=now - timedelta(days=31))
    assert ids(get_profile(conn, caller()).active_projects) == [recent]


def test_groups_are_capped_newest_first(conn):
    now = datetime.now(UTC)
    made = [insert(conn, f"pref {i}", kind="preference", when=now - timedelta(minutes=i)) for i in range(20)]
    prefs = get_profile(conn, caller()).preferences
    assert ids(prefs) == made[:PROFILE_GROUP_CAP]


def test_project_filter_puts_project_items_first_and_keeps_general_ones(conn):
    now = datetime.now(UTC)
    general = insert(conn, "Prefers terse answers.", kind="preference", when=now)
    tagged = insert(conn, "Use tabs in this repo.", kind="preference", project="A", when=now - timedelta(days=1))
    insert(conn, "Other repo uses spaces.", kind="preference", project="B")
    assert ids(get_profile(conn, caller(), project="A").preferences) == [tagged, general]


def test_excludes_superseded_and_unreadable_memories(conn):
    old = insert(conn, "old pref", kind="preference")
    new = insert(conn, "new pref", kind="preference")
    conn.execute("UPDATE memories SET superseded_by = ? WHERE id = ?", (new, old))
    insert(conn, "personal pref", kind="preference", scope="personal")
    work = insert(conn, "work pref", kind="preference", scope="work")

    assert ids(get_profile(conn, caller("cursor", read=["work"], write="work")).preferences) == [work]
    assert old not in ids(get_profile(conn, caller()).preferences)


def test_a_project_is_matched_ignoring_case_and_by_part_of_its_name(conn):
    kitchen = insert(conn, "Renovating the kitchen.", kind="project", project="Kitchen Renovation 2026")
    other = insert(conn, "Garden plan.", kind="project", project="Garden")
    for asked in ("kitchen renovation 2026", "renovation", "my Kitchen Renovation 2026 budget"):
        profile = get_profile(conn, caller(), project=asked)
        assert ids(profile.active_projects) == [kitchen], asked
        assert profile.known_projects is None
    assert get_profile(conn, caller(), project="Kitchen Renovation 2026").project_note is None
    assert "Showing project 'Kitchen Renovation 2026' for 'renovation'" in get_profile(conn, caller(), project="renovation").project_note
    assert ids(get_profile(conn, caller(), project="garden").active_projects) == [other]


def test_a_project_matching_nothing_gives_the_whole_profile_and_the_projects_there_are(conn):
    """A model guessed a name ('cabinets') that matches no stored project and got an empty
    profile (live test, 2026-10-19)."""
    kitchen = insert(conn, "Renovating the kitchen.", kind="project", project="Kitchen Renovation 2026")
    pref = insert(conn, "Prefers terse answers.", kind="preference")
    insert(conn, "Work thing.", kind="project", project="Billing", scope="work")
    profile = get_profile(conn, caller("cursor-vscode", read=["personal"], write="personal"), project="cabinets")
    assert (ids(profile.active_projects), ids(profile.preferences)) == ([kitchen], [pref])
    assert profile.known_projects == ["Kitchen Renovation 2026"]  # not the work project it can't read
    assert "No memories are tagged with project 'cabinets'" in profile.project_note


def test_a_short_project_name_is_matched_only_exactly(conn):
    insert(conn, "x", kind="project", project="Garden")
    assert get_profile(conn, caller(), project="ar").known_projects == ["Garden"]
