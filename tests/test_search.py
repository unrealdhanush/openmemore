from datetime import UTC, datetime, timedelta

import pytest

from conftest import caller, insert
from openmcp.safety.wrap import NOTICE
from openmcp.tools.search import fts_query, search_context


def ids(result):
    return [h.id for h in result.results]


def test_natural_language_question_finds_preference_via_stemming(conn):
    pref = insert(conn, "Prefers Python and terse answers.", kind="preference")
    insert(conn, "Lives in Berlin.")
    result = search_context(conn, caller(), "what do you know about my preferences?")
    assert ids(result) == [pref]


def test_results_are_wrapped_and_carry_metadata(conn):
    mid = insert(conn, "Uses uv for packaging.", kind="preference", project="OpenMCP", source="cursor")
    result = search_context(conn, caller(), "uv packaging")
    assert result.notice == NOTICE
    hit = result.results[0]
    assert hit.content.startswith(f'<stored_context source="cursor" id="{mid}">')
    assert (hit.type, hit.kind, hit.project, hit.source) == ("memory", "preference", "OpenMCP", "cursor")
    assert hit.score > 0


def test_subject_matches_rank_above_content_matches(conn):
    in_content = insert(conn, "Mentions rust once in passing among many other words here.")
    in_subject = insert(conn, "Chose it for the CLI.", subject="rust")
    assert ids(search_context(conn, caller(), "rust")) == [in_subject, in_content]


@pytest.mark.parametrize("query", ['"unbalanced', "a AND", "NEAR(x y", "col:value", "* ^ -", "' OR 1=1 --"])
def test_fts_syntax_in_queries_is_treated_literally(conn, query):
    insert(conn, "value of x and y")
    search_context(conn, caller(), query)  # must not raise


def test_query_with_only_punctuation_returns_nothing(conn):
    insert(conn, "anything")
    assert search_context(conn, caller(), "?!").results == []


def test_only_stopwords_still_searches(conn):
    mid = insert(conn, "The Who is a band they like.")
    assert ids(search_context(conn, caller(), "the who")) == [mid]


def test_only_readable_scopes_are_searched(conn):
    insert(conn, "Personal python note.", scope="personal")
    work = insert(conn, "Work python note.", scope="work")
    assert ids(search_context(conn, caller("cursor", read=["work"], write="work"), "python")) == [work]


def test_filters_by_kind_project_and_limit(conn):
    decision = insert(conn, "python decision", kind="decision", project="A")
    insert(conn, "python fact", kind="fact", project="A")
    insert(conn, "python decision elsewhere", kind="decision", project="B")
    assert ids(search_context(conn, caller(), "python", kinds=["decision"], project="A")) == [decision]
    for _ in range(5):
        insert(conn, f"python {_}")
    assert len(search_context(conn, caller(), "python", limit=3).results) == 3


def test_superseded_versions_are_not_returned(conn):
    old = insert(conn, "Uses python 3.11")
    new = insert(conn, "Uses python 3.12")
    conn.execute("UPDATE memories SET superseded_by = ? WHERE id = ?", (new, old))
    assert ids(search_context(conn, caller(), "python")) == [new]


def test_equal_scores_prefer_recent(conn):
    now = datetime.now(UTC)
    older = insert(conn, "likes python", when=now - timedelta(days=10))
    newer = insert(conn, "likes python too", when=now)
    conn.execute("UPDATE memories SET content = 'likes python' WHERE id = ?", (newer,))
    assert ids(search_context(conn, caller(), "python")) == [newer, older]


def test_fts_query_builder():
    assert fts_query("What do you know about my Preferences?") == '"preferences"'
    assert fts_query('python "NEAR" python*') == '"python" OR "near"'
    assert fts_query("...") is None
