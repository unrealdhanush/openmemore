"""The evaluation harness, on synthetic datasets in both formats."""

import json
from datetime import UTC, datetime

import httpx
import pytest

from bench import datasets, runner
from bench.datasets import Question
from bench.runner import Retrieved


def write_longmemeval(path, questions):
    path.write_text(json.dumps(questions))
    return path


def lme_question(qid, question, answer_session, sessions, qtype="single-session-user"):
    ids = [sid for sid, _ in sessions]
    return {
        "question_id": qid, "question_type": qtype, "question": question, "answer": "x",
        "question_date": "2023/05/30 (Tue) 10:00", "answer_session_ids": [answer_session],
        "haystack_session_ids": ids, "haystack_dates": ["2023/05/2%d (Sat) 09:00" % i for i in range(len(ids))],
        "haystack_sessions": [turns for _, turns in sessions],
    }


SESSIONS = [
    ("s_garden", [{"role": "user", "content": "I'm planting tomatoes and basil this spring.", "has_answer": True},
                  {"role": "assistant", "content": "Great combination for a vegetable garden."}]),
    ("s_car", [{"role": "user", "content": "My car needs new brake pads."},
               {"role": "assistant", "content": "Get them checked at a garage."}]),
    ("s_trip", [{"role": "user", "content": "We're flying to Lisbon in June."},
                {"role": "assistant", "content": "June is a lovely time for Portugal."}]),
]


def test_longmemeval_format(tmp_path):
    path = write_longmemeval(tmp_path / "lme.json", [
        lme_question("q1", "What am I planting this spring?", "s_garden", SESSIONS),
        lme_question("q2_abs", "What is my dog's name?", "s_car", SESSIONS),
    ])
    first, abstention = datasets.longmemeval(path)
    q = first.questions[0]
    assert (q.evidence_sessions, q.evidence_turns, q.date) == (
        {"s_garden"}, {"s_garden#0"}, datetime(2023, 5, 30, 10, 0, tzinfo=UTC))
    assert [s.id for s in first.sessions] == ["s_garden", "s_car", "s_trip"]
    assert abstention.questions[0].type == "single-session-user-abstention" and not abstention.questions[0].evidence_sessions


def test_locomo_format(tmp_path):
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps([{
        "sample_id": "conv-1",
        "conversation": {
            "speaker_a": "Ana", "speaker_b": "Ben",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [{"speaker": "Ana", "dia_id": "D1:1", "text": "I adopted a cat named Miso."},
                          {"speaker": "Ben", "dia_id": "D1:2", "text": "Cute!"}],
            "session_2_date_time": "10:00 am on 9 June, 2023",
            "session_2": [{"speaker": "Ben", "dia_id": "D2:1", "text": "I started rock climbing."}],
        },
        "qa": [{"question": "What is Ana's cat called?", "answer": "Miso", "evidence": ["D1:1"], "category": 4},
               {"question": "When did Ben start climbing?", "answer": "June 2023", "evidence": ["D2:1; D1:2"],
                "category": 2}],
    }]))
    [h] = datasets.locomo(path)
    assert [s.date for s in h.sessions] == [datetime(2023, 5, 8, 13, 56, tzinfo=UTC), datetime(2023, 6, 9, 10, 0, tzinfo=UTC)]
    assert h.sessions[0].turns[0].text == "Ana: I adopted a cat named Miso." and h.sessions[0].turns[1].role == "assistant"
    single, temporal = h.questions
    assert (single.type, single.evidence_turns, single.evidence_sessions) == ("single-hop", {"D1:1"}, {"session_1"})
    assert temporal.evidence_turns == {"D2:1", "D1:2"} and temporal.evidence_sessions == {"session_1", "session_2"}


def test_metrics():
    q = Question("q", "?", "x", "t", None, evidence_sessions={"a", "b"}, evidence_turns={"a#0", "b#1"})
    r = Retrieved(sessions=["c", "a", "a", "d", "b"], turns=["c#0", "a#0", "a#2", "d#0", "b#1"])
    assert runner.session_recall(r, q, 2) == 0.5 and runner.session_recall(r, q, 5) == 1.0
    assert runner.any_hit(r, q, 1) == 0.0 and runner.any_hit(r, q, 2) == 1.0
    assert runner.turn_recall(r, q, 5) == 1.0 and runner.turn_recall(r, q, 3) == 0.5
    perfect = Retrieved(sessions=["a", "b"], turns=["a#0", "b#1"])
    assert runner.ndcg(perfect, q, 5) == pytest.approx(1.0)
    assert 0 < runner.ndcg(r, q, 5) < 1


def test_a_run_finds_the_evidence_and_reports_by_type(tmp_path):  # keyword search: conftest's no_model
    path = write_longmemeval(tmp_path / "lme.json", [
        lme_question("q1", "What am I planting in the garden this spring?", "s_garden", SESSIONS),
        lme_question("q2", "Which city are we flying to in June?", "s_trip", SESSIONS, qtype="temporal-reasoning"),
        lme_question("q3_abs", "What is my dog's name?", "s_car", SESSIONS),
    ])
    report = runner.run("synthetic", datasets.longmemeval(path), (1, 5), embedder_name="none")
    summary = report["summary"]
    assert summary["all"]["n"] == 3 and summary["all"]["any_hit@5"] == 1.0
    assert summary["temporal-reasoning"]["session_recall@1"] == 1.0
    assert "any_hit@5" not in summary["single-session-user-abstention"]  # no evidence: not scored
    assert report["latency"]["query_ms_p50"] is not None
    assert all(set(row) <= {"id", "type", "evidence"} | {f"{m}@{k}" for m in runner.METRICS for k in (1, 5)}
               for row in report["questions_detail"])  # ids and numbers only, no dataset text


def test_sampling_spreads_over_types():
    def haystack(i, t):
        return datasets.Haystack(f"h{i}", [], [Question(f"q{i}", "?", "", t, None)])
    pool = [haystack(i, "a") for i in range(10)] + [haystack(i, "b") for i in range(10, 12)]
    picked = runner.sample(pool, 4)
    assert sorted(h.questions[0].type for h in picked) == ["a", "a", "b", "b"]


def test_judged_answers():
    from bench.judge import answerer
    from openmemore import llm

    replies = iter(["Tomatoes and basil.", "correct", "I don't know.", "incorrect"])

    def respond(request):
        text = next(replies)
        return httpx.Response(200, json={"status": "completed", "output": [
            {"type": "message", "content": [{"type": "output_text", "text": text}]}]})

    real = llm.complete
    model = llm.Model(endpoint="http://127.0.0.1:9/v1", model="m")
    answer = answerer(model, 5)
    q = Question("q", "What am I planting?", "tomatoes and basil", "t", datetime(2023, 5, 30, tzinfo=UTC))
    r = Retrieved(sessions=["s"], turns=["s#0"], texts=["I'm planting tomatoes and basil."])
    llm.complete = lambda m, i, p, transport=None: real(m, i, p, transport=httpx.MockTransport(respond))
    try:
        assert answer(q, r) == {"correct": 1.0}
        assert answer(q, r) == {"correct": 0.0}
    finally:
        llm.complete = real


def test_the_embedding_cache_reuses_vectors(tmp_path):
    from conftest import FakeEmbedder

    inner = FakeEmbedder()
    cache = runner.CachingEmbedder(inner, tmp_path / "v.sqlite")
    first = cache.passages(["a b", "c d"])
    again = cache.passages(["c d", "a b", "e f"])
    assert again[:2] == [first[1], first[0]] and inner.calls == 3
    assert cache.query("q") == cache.query("q") and inner.calls == 4
