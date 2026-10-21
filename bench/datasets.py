"""Long-term memory benchmarks (SPEC.md M7.6), read into one shape.

- LongMemEval (Wu et al., MIT; github.com/xiaowu0162/LongMemEval): each question has its
  own history ("haystack") of sessions, with the sessions and turns holding the answer
  marked. `longmemeval-s` is the cleaned S split (about 50 sessions per question).
- LoCoMo (Maharana et al., github.com/snap-research/locomo; non-commercial terms): ten
  long two-person conversations, each with many questions whose evidence turns are
  marked ("D3:12" is session 3, turn 12).

Datasets are downloaded to the cache on first use and never committed.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

CACHE = Path(os.environ.get("OPENMEMORE_BENCH_DIR") or Path.home() / ".cache" / "openmemore-bench")
SOURCES = {
    "longmemeval-s": ("longmemeval_s_cleaned.json",
                      "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"),
    "longmemeval-oracle": ("longmemeval_oracle.json",
                           "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json"),
    "locomo": ("locomo10.json", "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"),
}
LOCOMO_CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


@dataclass
class Turn:
    id: str
    role: str
    text: str


@dataclass
class Session:
    id: str
    date: datetime | None
    turns: list[Turn]


@dataclass
class Question:
    id: str
    text: str
    answer: str
    type: str
    date: datetime | None
    evidence_sessions: set[str] = field(default_factory=set)
    evidence_turns: set[str] = field(default_factory=set)


@dataclass
class Haystack:
    """A history to store, and the questions asked about it."""

    id: str
    sessions: list[Session]
    questions: list[Question]


def path_for(name: str, cache: Path = CACHE) -> Path:
    """The dataset file, downloaded to the cache if it isn't there yet."""
    filename, url = SOURCES[name]
    path = cache / filename
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        print(f"downloading {name} from {url} to {path}")
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        partial.replace(path)
    return path


_LME_DATE = re.compile(r"(\d{4})/(\d{2})/(\d{2})\s*(?:\(\w+\))?\s*(\d{1,2}):(\d{2})")


def _lme_date(value: str | None) -> datetime | None:
    m = _LME_DATE.search(value or "")
    return datetime(*map(int, m.groups()), tzinfo=UTC) if m else None


def longmemeval(path: Path) -> Iterator[Haystack]:
    for q in json.loads(path.read_text(encoding="utf-8")):
        sessions = []
        evidence_turns = set()
        for sid, date, turns in zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]):
            kept = []
            for i, t in enumerate(turns):
                turn_id = f"{sid}#{i}"
                if t.get("has_answer"):
                    evidence_turns.add(turn_id)
                if (t.get("content") or "").strip():
                    kept.append(Turn(turn_id, "assistant" if t.get("role") == "assistant" else "user", t["content"]))
            sessions.append(Session(sid, _lme_date(date), kept))
        abstention = q["question_id"].endswith("_abs")
        yield Haystack(q["question_id"], sessions, [Question(
            id=q["question_id"], text=q["question"], answer=str(q["answer"]),
            type=q["question_type"] + ("-abstention" if abstention else ""), date=_lme_date(q.get("question_date")),
            evidence_sessions=set() if abstention else set(q.get("answer_session_ids") or ()),
            evidence_turns=set() if abstention else evidence_turns,
        )])


_LOCOMO_DATE = re.compile(r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s+(\d{4})", re.I)
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
     "december"], start=1)}


def _locomo_date(value: str | None) -> datetime | None:
    m = _LOCOMO_DATE.search(value or "")
    if not m or m.group(5).lower() not in _MONTHS:
        return None
    hour, minute, half, day, month, year = m.groups()
    hour = int(hour) % 12 + (12 if half.lower() == "pm" else 0)
    return datetime(int(year), _MONTHS[month.lower()], int(day), hour, int(minute), tzinfo=UTC)


def locomo(path: Path) -> Iterator[Haystack]:
    for sample in json.loads(path.read_text(encoding="utf-8")):
        conv = sample["conversation"]
        numbers = sorted(int(k.split("_")[1]) for k in conv if re.fullmatch(r"session_\d+", k))
        speakers = [conv.get("speaker_a"), conv.get("speaker_b")]
        sessions, last = [], None
        for n in numbers:
            date = _locomo_date(conv.get(f"session_{n}_date_time"))
            last = date or last
            turns = [
                Turn(t["dia_id"], "user" if t.get("speaker") == speakers[0] else "assistant",
                     f"{t.get('speaker')}: {t['text']}")
                for t in conv[f"session_{n}"] if (t.get("text") or "").strip()
            ]
            sessions.append(Session(f"session_{n}", date, turns))
        questions = []
        for i, qa in enumerate(sample["qa"]):
            turns = {e for raw in qa.get("evidence") or () for e in re.findall(r"D\d+:\d+", raw)}
            questions.append(Question(
                id=f"{sample['sample_id']}:{i}", text=qa["question"], answer=str(qa.get("answer", "")),
                type=LOCOMO_CATEGORIES.get(qa.get("category"), str(qa.get("category"))), date=last,
                evidence_sessions={f"session_{int(t[1:].split(':')[0])}" for t in turns}, evidence_turns=turns,
            ))
        yield Haystack(sample["sample_id"], sessions, questions)


READERS = {"longmemeval-s": longmemeval, "longmemeval-oracle": longmemeval, "locomo": locomo}


def load(name: str, cache: Path = CACHE) -> list[Haystack]:
    return list(READERS[name](path_for(name, cache)))
