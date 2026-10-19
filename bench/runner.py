"""Run OpenMCP's retrieval against a benchmark (SPEC.md M7.6).

Each history is stored the way capture stores conversations (one conversation per
session, one message per turn, with its date), embedded, and searched with
`search_context` for each question, with "now" set to the question's date. Results
are scored against the sessions and turns that hold the answer.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench.datasets import CACHE, Haystack, Question
from openmcp import __version__, db, embed
from openmcp.config import ClientScopes
from openmcp.ingest import store_conversation
from openmcp.models import ParsedConversation, ParsedMessage
from openmcp.safety.wrap import unwrap
from openmcp.tools import Caller
from openmcp.tools.search import search_context

READER = Caller(client="bench", scopes=ClientScopes(read_scopes=None, write_scope="personal"))
PROVIDER = "bench"


class CachingEmbedder:
    """The real embedder, with vectors cached by text (sessions repeat across
    LongMemEval questions, and reruns repeat everything)."""

    def __init__(self, inner: embed.Embedder, path: Path, model: str = embed.MODEL) -> None:
        self.inner, self.model = inner, model
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
        self.hits = self.misses = 0

    def _key(self, kind: str, text: str) -> str:
        return hashlib.sha256(f"{self.model}\0{kind}\0{text}".encode()).hexdigest()

    def passages(self, texts: Sequence[str]) -> list[bytes]:
        keys = [self._key("p", t) for t in texts]
        found = dict(self.conn.execute(
            f"SELECT key, vector FROM vectors WHERE key IN ({','.join('?' * len(keys))})", keys).fetchall()) if keys else {}
        missing = [i for i, k in enumerate(keys) if k not in found]
        self.hits += len(keys) - len(missing)
        self.misses += len(missing)
        if missing:
            vectors = self.inner.passages([texts[i] for i in missing])
            with self.conn:
                self.conn.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?)",
                                      [(keys[i], v) for i, v in zip(missing, vectors)])
            found.update((keys[i], v) for i, v in zip(missing, vectors))
        return [found[k] for k in keys]

    def query(self, text: str) -> bytes:
        key = self._key("q", text)
        row = self.conn.execute("SELECT vector FROM vectors WHERE key = ?", (key,)).fetchone()
        if row:
            self.hits += 1
            return row[0]
        self.misses += 1
        vector = self.inner.query(text)
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO vectors VALUES (?, ?)", (key, vector))
        return vector


def use_embedder(embedder: embed.Embedder | None) -> None:
    """Make OpenMCP's code use this embedder (None: keyword search only)."""
    embed._embedder, embed._unavailable = embedder, None if embedder else "benchmark: keyword search only"
    if embedder is None:
        embed.get_embedder = lambda allow_download=False: None  # type: ignore[assignment]


def store_haystack(conn: sqlite3.Connection, haystack: Haystack) -> None:
    for session in haystack.sessions:
        if not session.turns:
            continue
        store_conversation(conn, ParsedConversation(
            provider=PROVIDER, external_id=session.id, title=None, created_at=session.date, updated_at=session.date,
            messages=[ParsedMessage(external_id=t.id, role=t.role, content=t.text, created_at=session.date)
                      for t in session.turns],
        ), "personal")


@dataclass
class Retrieved:
    """One question's ranked results, mapped back to the dataset."""

    sessions: list[str]
    """Each result's session, in result order."""
    turns: list[str]
    """Each result's turn."""
    texts: list[str] = field(default_factory=list)


def retrieve(conn: sqlite3.Connection, question: Question, k: int) -> Retrieved:
    result = search_context(conn, READER, question.text, include_chats=True, limit=k, now=question.date)
    sessions, turns, texts = [], [], []
    for hit in result.results:
        if hit.type != "chat":
            continue
        row = conn.execute(
            "SELECT c.external_id, m.external_id FROM chunks k JOIN messages m ON m.id = k.message_id "
            "JOIN conversations c ON c.id = m.conversation_id WHERE k.id = ?", (hit.id,)).fetchone()
        if row is None:
            continue
        sessions.append(row[0])
        turns.append(row[1])
        texts.append(unwrap(hit.content))
    return Retrieved(sessions, turns, texts)


# --- metrics -------------------------------------------------------------------------


def session_recall(retrieved: Retrieved, question: Question, k: int) -> float:
    found = set(_top_sessions(retrieved, k)) & question.evidence_sessions
    return len(found) / len(question.evidence_sessions)


def any_hit(retrieved: Retrieved, question: Question, k: int) -> float:
    return float(bool(set(_top_sessions(retrieved, k)) & question.evidence_sessions))


def ndcg(retrieved: Retrieved, question: Question, k: int) -> float:
    """Over the unique sessions in the top k results; each evidence session counts once."""
    gains = [1.0 if s in question.evidence_sessions else 0.0 for s in _top_sessions(retrieved, k)]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(question.evidence_sessions), k)))
    return dcg / ideal if ideal else 0.0


def turn_recall(retrieved: Retrieved, question: Question, k: int) -> float | None:
    if not question.evidence_turns:
        return None
    return len(set(retrieved.turns[:k]) & question.evidence_turns) / len(question.evidence_turns)


def _top_sessions(retrieved: Retrieved, k: int) -> list[str]:
    """The distinct sessions among the top k results (k results, not k sessions)."""
    return list(dict.fromkeys(retrieved.sessions[:k]))


METRICS: dict[str, Callable[[Retrieved, Question, int], float | None]] = {
    "session_recall": session_recall, "any_hit": any_hit, "ndcg": ndcg, "turn_recall": turn_recall,
}


# --- a run ---------------------------------------------------------------------------


def commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], capture_output=True, text=True).stdout
        return sha.stdout.strip() + ("+uncommitted" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run(dataset: str, haystacks: Iterable[Haystack], ks: Sequence[int] = (5, 10), *, embedder_name: str,
        answer: Callable[[Question, Retrieved], dict[str, Any]] | None = None,
        progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Store each history in a fresh database, ask its questions, and score them."""
    haystacks = list(haystacks)
    k_max = max(ks)
    per_question: list[dict[str, Any]] = []
    ingest_s: list[float] = []
    query_ms: list[float] = []
    with tempfile.TemporaryDirectory(prefix="openmcp-bench-") as tmp:
        for n, haystack in enumerate(haystacks, start=1):
            with closing(db.open_db(Path(tmp) / f"{n}.db")) as conn:
                start = time.perf_counter()
                store_haystack(conn, haystack)
                embedder = embed.get_embedder()
                if embedder is not None:
                    embed.embed_pending(conn, embedder)
                ingest_s.append(time.perf_counter() - start)
                for question in haystack.questions:
                    start = time.perf_counter()
                    retrieved = retrieve(conn, question, k_max)
                    query_ms.append((time.perf_counter() - start) * 1000)
                    row: dict[str, Any] = {"id": question.id, "type": question.type,
                                           "evidence": bool(question.evidence_sessions)}
                    if question.evidence_sessions:
                        for k in ks:
                            for name, metric in METRICS.items():
                                value = metric(retrieved, question, k)
                                if value is not None:
                                    row[f"{name}@{k}"] = value
                    if answer is not None:
                        row.update(answer(question, retrieved))
                    per_question.append(row)
            if progress:
                progress(n, len(haystacks))
    return {
        "dataset": dataset, "commit": commit(), "openmcp": __version__, "embedder": embedder_name,
        "ks": list(ks), "ran_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "haystacks": len(haystacks), "questions": len(per_question),
        "summary": summarize(per_question),
        "latency": {
            "ingest_s_mean": round(statistics.fmean(ingest_s), 3) if ingest_s else None,
            "query_ms_p50": round(statistics.median(query_ms), 1) if query_ms else None,
            "query_ms_p95": round(sorted(query_ms)[int(0.95 * (len(query_ms) - 1))], 1) if query_ms else None,
        },
        "questions_detail": per_question,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Mean of every metric, overall and by question type. Questions without evidence
    (abstention) count only towards answer accuracy."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[row["type"]].append(row)
    out = {}
    for name, members in sorted(groups.items(), key=lambda g: (g[0] != "all", g[0])):
        metrics = sorted({k for r in members for k in r if "@" in k or k == "correct"})
        out[name] = {"n": len(members), **{
            m: round(statistics.fmean([r[m] for r in members if m in r]), 4)
            for m in metrics if any(m in r for r in members)
        }}
    return out


def sample(haystacks: list[Haystack], limit: int | None) -> list[Haystack]:
    """Up to `limit` histories, spread evenly over question types (deterministic)."""
    if limit is None or limit >= len(haystacks):
        return haystacks
    by_type: dict[str, list[Haystack]] = defaultdict(list)
    for h in haystacks:
        by_type[h.questions[0].type].append(h)
    picked: list[Haystack] = []
    queues = [sorted(v, key=lambda h: h.id) for _, v in sorted(by_type.items())]
    while len(picked) < limit and any(queues):
        for queue in queues:
            if queue and len(picked) < limit:
                picked.append(queue.pop(0))
    return picked


def cache_path() -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / "vectors.sqlite"
