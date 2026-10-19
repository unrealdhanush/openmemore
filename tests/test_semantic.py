"""Semantic search (SPEC.md M2). Most tests use FakeEmbedder (fast, offline, exact
similarities); `real_model` tests use bge-small itself, for the milestone's done-when."""

import pytest
from mcp import Client

from conftest import FakeEmbedder, caller, insert
from openmcp import embed, repo
from openmcp.cli import initialize
from openmcp.models import MemoryInput
from openmcp.server import build_server
from openmcp.tools.conversations import record_exchange
from openmcp.tools.memories import save_memories, update_memory
from openmcp.tools.search import SIMILARITY_FLOOR, search_context

ME = caller("claude-desktop", read=["personal", "work"], write="personal")
WORK_ONLY = caller("cursor", read=["work"], write="work")


def unit(*xs):
    """A pinned vector: FakeEmbedder pads and normalizes it."""
    return list(xs)


def vector_count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}_vec").fetchone()[0]


def save(conn, *texts, kind="preference", who=ME):
    return save_memories(conn, who, [MemoryInput(kind=kind, content=t) for t in texts])


# --- embedding on write --------------------------------------------------------------


def test_saved_memories_are_embedded(conn, fake_embedder):
    [mid] = save(conn, "Prefers Python.").saved
    assert vector_count(conn, "memories") == 1
    assert repo.count_pending_embeddings(conn) == 0
    assert conn.execute("SELECT embedded FROM memories WHERE id = ?", (mid,)).fetchone()[0] == 1


def test_without_a_model_rows_wait_and_embed_pending_catches_up(conn):
    save(conn, "Prefers Python.", "Prefers tea.")
    record_exchange(conn, ME, "Flask question", "Flask answer")
    assert vector_count(conn, "memories") == 0 and repo.count_pending_embeddings(conn) == 4

    report = embed.embed_pending(conn, FakeEmbedder())
    assert (report.memories, report.chunks, report.remaining) == (2, 2, 0)
    assert (vector_count(conn, "memories"), vector_count(conn, "chunks")) == (2, 2)


def test_embed_pending_respects_its_budget(conn):
    save(conn, "Prefers Python.")
    report = embed.embed_pending(conn, FakeEmbedder(), budget_ms=-1)
    assert (report.memories, report.remaining) == (0, 1)


def test_record_exchange_embeds_its_chunks(conn, fake_embedder):
    record_exchange(conn, ME, "How do I rate-limit Flask?", "Use Flask-Limiter.")
    assert vector_count(conn, "chunks") == 2


def test_vectors_follow_their_rows(conn, fake_embedder):
    [mid] = save(conn, "Prefers Python.").saved
    new_id = update_memory(conn, ME, mid, "Prefers Python 3.12.").new_id
    assert repo.count_pending_embeddings(conn) == 0
    assert [m.id for m, _ in repo.knn_memories(conn, fake_embedder.query("python"), k=5, scopes=None)] == [new_id]

    conn.execute("DELETE FROM memories")
    assert vector_count(conn, "memories") == 0

    cid = record_exchange(conn, ME, "q", "a").conversation_id
    repo.delete_conversation(conn, cid)
    assert vector_count(conn, "chunks") == 0


def test_storing_a_vector_twice_replaces_it(conn, fake_embedder):
    """Two servers may embed the same pending row at once."""
    [mid] = save(conn, "Prefers Python.").saved
    v = fake_embedder.query("something else")
    assert repo.store_vectors(conn, "memories", [(mid, v)]) == 1
    assert vector_count(conn, "memories") == 1
    assert repo.store_vectors(conn, "memories", [(999, v)]) == 0  # deleted meanwhile


# --- near-duplicates -----------------------------------------------------------------


@pytest.fixture
def pinned(monkeypatch):
    """Exact similarities: a~b 0.95 (near-duplicate), a~c 0.8 (related, not a duplicate)."""
    fake = FakeEmbedder(vectors={
        "Prefers Python for scripting.": unit(1, 0),
        "Likes Python for scripts.": unit(0.95, 0.3122),
        "Prefers Go for services.": unit(0.8, 0.6),
        "Prefers Rust for CLIs.": unit(0, 0, 1),
        "Likes Rust for command-line tools.": unit(0, 0, 0.95, 0.3122),
    })
    monkeypatch.setattr(embed, "get_embedder", lambda allow_download=False: fake)
    return fake


def test_near_duplicates_are_saved_and_reported(conn, pinned):
    [first] = save(conn, "Prefers Python for scripting.").saved
    result = save(conn, "Prefers Go for services.", "Likes Python for scripts.")

    assert len(result.saved) == 2
    [dup] = result.possible_duplicates
    assert (dup.new_index, dup.existing_id) == (1, first)
    assert dup.existing_content.startswith('<stored_context source="claude-desktop"')
    assert "Prefers Python for scripting." in dup.existing_content


def test_near_duplicates_within_one_call_are_reported(conn, pinned):
    result = save(conn, "Prefers Python for scripting.", "Likes Python for scripts.")
    assert [(d.new_index, d.existing_id) for d in result.possible_duplicates] == [(1, result.saved[0])]


def test_near_duplicates_need_the_same_kind(conn, pinned):
    save(conn, "Prefers Python for scripting.", kind="fact")
    assert save(conn, "Likes Python for scripts.", kind="preference").possible_duplicates == []


def test_near_duplicates_in_unreadable_scopes_are_not_reported(conn, pinned):
    save(conn, "Prefers Rust for CLIs.", who=caller("x", read=["secret"], write="secret"))
    assert save(conn, "Likes Rust for command-line tools.").possible_duplicates == []


# --- hybrid search -------------------------------------------------------------------


def test_vectors_find_what_keywords_miss(conn, monkeypatch):
    fake = FakeEmbedder(vectors={"Orange lazy cat: Marmalade.": unit(1, 0.2), "kitten ideas": unit(1, 0)})
    save(conn, "Orange lazy cat: Marmalade.", kind="note")
    assert search_context(conn, ME, "kitten ideas").results == []  # no shared words, no model

    monkeypatch.setattr(embed, "get_embedder", lambda allow_download=False: fake)
    embed.embed_pending(conn, fake)
    [hit] = search_context(conn, ME, "kitten ideas").results
    assert "Marmalade" in hit.content


def test_weak_vector_matches_are_dropped(conn, monkeypatch):
    texts = {"strong match": unit(1, 0.1), "weak match": unit(0.5, 0.866), "query words": unit(1, 0)}
    fake = FakeEmbedder(vectors=texts)
    monkeypatch.setattr(embed, "get_embedder", lambda allow_download=False: fake)
    save(conn, "strong match", "weak match", kind="note")

    hits = search_context(conn, ME, "query words").results
    assert [h.content.split("\n")[1] for h in hits] == ["strong match"]
    assert SIMILARITY_FLOOR > 0.5


def test_results_found_both_ways_rank_first(conn, monkeypatch):
    fake = FakeEmbedder(vectors={
        "Flask rate limits with Redis.": unit(1, 0.12),  # keyword and vector match
        "Throttling the API gateway.": unit(1, 0.1),     # vector only (nearer than the one above)
        "Flask templates.": unit(0, 1),                  # keyword only
        "flask": unit(1, 0),
    })
    monkeypatch.setattr(embed, "get_embedder", lambda allow_download=False: fake)
    save(conn, "Flask templates.", "Throttling the API gateway.", "Flask rate limits with Redis.", kind="note")

    hits = [h.content.split("\n")[1] for h in search_context(conn, ME, "flask").results]
    assert hits[0] == "Flask rate limits with Redis."
    assert set(hits) == {"Flask rate limits with Redis.", "Throttling the API gateway.", "Flask templates."}


def test_recency_breaks_ties(conn, monkeypatch):
    from datetime import UTC, datetime, timedelta
    old = insert(conn, "Uses vim.", when=datetime.now(UTC) - timedelta(days=200))
    new = insert(conn, "Uses vim daily.", when=datetime.now(UTC))
    # Same keyword rank pattern either way: only recency separates them.
    ids = [h.id for h in search_context(conn, ME, "vim").results]
    assert ids.index(new) < ids.index(old)


def test_vector_search_respects_scopes(conn, fake_embedder):
    save(conn, "Prefers Python.")
    record_exchange(conn, ME, "Python packaging?", "Use uv.")
    assert search_context(conn, WORK_ONLY, "python").results == []


def test_kind_and_project_filters_apply_to_vector_hits(conn, fake_embedder):
    save(conn, "Prefers Python.")
    save(conn, "Python is installed.", kind="fact")
    assert {h.kind for h in search_context(conn, ME, "python", kinds=["fact"]).results} == {"fact"}
    assert search_context(conn, ME, "python", project="Nope").results == []


# --- surface -------------------------------------------------------------------------


@pytest.mark.anyio
async def test_ping_reports_semantic_search(home, monkeypatch):
    initialize(home)
    async with Client(build_server("claude-desktop", home)) as client:
        off = (await client.call_tool("ping", {})).structured_content["semantic_search"]
        monkeypatch.setattr(embed, "get_embedder", lambda allow_download=False: FakeEmbedder())
        on = (await client.call_tool("ping", {})).structured_content["semantic_search"]
    assert off.startswith("unavailable") and "openmcp embed" in off
    assert on == f"ready ({embed.MODEL})"


# --- the real model (SPEC M2 done-when) ----------------------------------------------

CORPUS = [
    ("note", "Orange lazy cat names: Marmalade, Pumpkin, Dozer."),
    ("preference", "Prefers Python for scripting and terse answers."),
    ("fact", "Celsius to Fahrenheit: F = C × 9/5 + 32."),
    ("fact", "Running club meets Saturdays at 7am at Lake Merritt."),
    ("preference", "Uses Neovim with the LazyVim distribution and a dark theme."),
    ("fact", "Allergic to peanuts; avoid recipes with peanut oil."),
    ("project", "Trip to Lisbon planned for March; booked an apartment in Alfama."),
    ("decision", "Kubernetes deploys go through Argo CD; production needs manual approval."),
]


@pytest.fixture
def corpus(conn):
    save_memories(conn, ME, [MemoryInput(kind=k, content=c) for k, c in CORPUS])
    return conn


@pytest.mark.real_model
@pytest.mark.parametrize("query, expected", [
    ("what should I call my new kitten", "Marmalade"),
    ("temperature conversion formula", "Fahrenheit"),
    ("what language do I like for small automation", "Prefers Python"),
    ("when is my jogging group", "Lake Merritt"),
    ("my upcoming vacation in Portugal", "Lisbon"),
    ("any food restrictions I mentioned", "peanut"),
])
def test_m2_done_when_a_paraphrase_finds_what_keywords_miss(corpus, monkeypatch, real_embedder, query, expected):
    """SPEC M2 'done when': each query shares no meaningful word with its memory."""
    monkeypatch.setattr(embed, "get_embedder", lambda allow_download=False: None)
    keyword_only = search_context(corpus, ME, query).results
    assert not any(expected in h.content for h in keyword_only)

    monkeypatch.setattr(embed, "get_embedder", lambda allow_download=False: real_embedder)
    embed.embed_pending(corpus, real_embedder)
    hits = search_context(corpus, ME, query).results
    assert hits and expected in hits[0].content


@pytest.mark.real_model
def test_unrelated_questions_find_little(corpus, real_embedder):
    for query in ("who won the 1998 world cup", "how tall is mount everest", "how do solar panels work"):
        assert len(search_context(corpus, ME, query).results) <= 1, query


@pytest.mark.real_model
def test_real_near_duplicates(conn, real_embedder):
    save(conn, "Running club meets Saturdays at 7am at Lake Merritt.", kind="fact")
    reworded = save(conn, "The running club meets at Lake Merritt on Saturdays at 7am.", kind="fact")
    different = save(conn, "Allergic to shellfish.", kind="fact")
    assert len(reworded.possible_duplicates) == 1
    assert different.possible_duplicates == []


@pytest.mark.real_model
@pytest.mark.xfail(strict=True, reason="bge-small doesn't link 'fitness app throttling' to Strava 429s (SPEC.md 14)")
def test_known_miss_throttling_paraphrase(corpus, real_embedder):
    save(corpus, "Strava returns 429 when the 15-minute rate limit is hit; the poller skips to the next window.",
         kind="decision")
    hits = search_context(corpus, ME, "how was I planning to deal with the fitness app throttling me").results
    assert hits and "Strava" in hits[0].content


# --- keeping keyword search within budget --------------------------------------------


def test_stems_match_the_index(conn):
    assert repo.stems(conn, ["deploys", "Retries", "???"]) == ["deploi", "retri", None]


def test_the_most_common_words_are_left_out_of_chat_keyword_search(conn, monkeypatch):
    from openmcp.tools import search
    for i in range(6):
        record_exchange(conn, ME, f"code question {i}", "code answer" + (" redis" if i == 0 else ""))
    monkeypatch.setattr(search, "MAX_CHUNK_KEYWORD_MATCHES", 3)
    # 'code' is in all 12 chunks, 'redis' in one: only 'redis' is searched.
    assert search.chunk_keyword_query(conn, ["code", "redis"]) == '"redis"'
    assert search.chunk_keyword_query(conn, ["code"]) is None
    assert search.chunk_keyword_query(conn, ["unseen", "redis"]) == '"unseen" OR "redis"'
    hits = search_context(conn, ME, "code redis").results
    assert hits and all("redis" in h.content for h in hits)
