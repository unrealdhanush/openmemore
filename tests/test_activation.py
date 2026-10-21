"""SPEC.md M7.7 part 7: activation (ACT-R base-level learning) instead of a fixed recency
half-life: recent and frequently presented items rank a little higher."""

import math
from contextlib import closing
from datetime import UTC, datetime, timedelta

import pytest

from conftest import caller, connect, insert
from openmemore import db, relations, repo
from openmemore.cli import initialize
from openmemore.tools.search import activation, search_context

NOW = datetime(2026, 10, 19, tzinfo=UTC)
ME = caller("claude-ai", read=["personal"], write="personal")


def days_ago(n):
    return NOW - timedelta(days=n)


def exact(ages, d=0.5):
    return math.log(sum(a ** -d for a in ages))


def test_newer_and_more_often_presented_is_more_active():
    assert activation(days_ago(1), 0, None, NOW) > activation(days_ago(30), 0, None, NOW) > activation(days_ago(365), 0, None, NOW)
    assert activation(days_ago(30), 5, days_ago(1), NOW) > activation(days_ago(30), 1, days_ago(1), NOW) \
        > activation(days_ago(30), 0, None, NOW)
    assert activation(days_ago(30), 3, days_ago(1), NOW) > activation(days_ago(30), 3, days_ago(20), NOW)


def test_it_is_a_power_law_not_a_half_life():
    # Exponential decay would make a year-old item ~2^12 times less weighty than a month-old one.
    month, year = activation(days_ago(30), 0, None, NOW), activation(days_ago(365), 0, None, NOW)
    assert month - year == pytest.approx(0.5 * math.log(365 / 30))


@pytest.mark.parametrize("first, last, n", [(60, 2, 4), (365, 30, 10), (10, 9, 1)])
def test_petrovs_approximation_is_close_to_the_exact_sum(first, last, n):
    """Presentations spread evenly between the first and the last, as the approximation assumes."""
    ages = [first] + [last + (first - last) * i / n for i in range(n)]
    approx = activation(days_ago(first), n, days_ago(last), NOW)
    assert approx == pytest.approx(exact(ages), abs=0.15)


def test_an_item_without_a_date_counts_as_a_year_old():
    assert activation(None, 0, None, NOW) == pytest.approx(-0.5 * math.log(365))


def test_activation_breaks_ties_between_equal_matches(conn, monkeypatch):
    """The same words: the more active one ranks first. (A weight large enough to outweigh
    one rank step; the shipped weight is tuned with the benchmark.)"""
    from openmemore.tools import search

    monkeypatch.setattr(search, "MEMORY_ACTIVATION_WEIGHT", 0.01)
    older = insert(conn, "Prefers green tea.", kind="preference", when=days_ago(400))
    newer = insert(conn, "Prefers green tea.", kind="preference", when=days_ago(2))

    def first():
        return search_context(conn, ME, "green tea", now=NOW).results[0].id

    assert first() == newer
    for _ in range(30):  # shown often lately, the older one catches up
        repo.record_presented(conn, [older], [], days_ago(1))
    assert first() == older


def test_confirmation_counts_as_a_presentation(conn):
    memory_id = insert(conn, "Prefers metric units.", kind="preference", source="claude-ai")
    relations.support(conn, memory_id, NOW, source="cursor-vscode")
    m = repo.get_memory(conn, memory_id)
    assert (m.presented, m.last_presented_at) == (1, NOW)


def test_the_search_function_alone_records_nothing(conn):
    """Evaluation calls it directly: benchmark questions mustn't reinforce each other."""
    memory_id = insert(conn, "Prefers green tea.", kind="preference")
    search_context(conn, ME, "green tea")
    assert repo.get_memory(conn, memory_id).presented == 0


@pytest.mark.anyio
async def test_what_clients_are_shown_is_counted(home):
    from openmemore import Memory

    initialize(home)
    with closing(db.connect(home / "openmemore.db")) as conn:
        memory_id = insert(conn, "Prefers green tea.", kind="preference")
    async with connect(home, "claude-ai") as client:
        await client.call_tool("search_context", {"query": "green tea"})
        await client.call_tool("get_profile", {})
    Memory("agent", home=home).search("green tea")
    with closing(db.connect(home / "openmemore.db")) as conn:
        assert repo.get_memory(conn, memory_id).presented == 3


def test_chats_get_no_age_weighting(conn):
    """Tuned with the benchmark: old conversations are as much evidence as new ones."""
    from openmemore.ingest import store_conversation
    from openmemore.models import ParsedConversation, ParsedMessage
    from openmemore.tools import search

    for name, age in (("old", 400), ("new", 2)):
        store_conversation(conn, ParsedConversation(provider="vscode-copilot", external_id=name, messages=[
            ParsedMessage(external_id="1", role="user", content="Plan a vegetable garden.", created_at=days_ago(age))]),
            "personal")
    hits = search_context(conn, ME, "vegetable garden", now=NOW, explain=True).results
    assert search.CHAT_ACTIVATION_WEIGHT == 0 and all(h.explain["activation"] == 0 for h in hits)
