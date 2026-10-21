"""Check the performance budget on a synthetic store: search_context under 100ms p95,
and save_memories with 20 items under 300ms p95, once warm, on 10k memories and 100k
conversation chunks. (The third budget, `openmemore serve` ready in under 500ms, is
tests/test_server.py's.)

    uv run python scripts/benchmark.py [--memories 10000] [--chunks 100000]

Builds a throwaway store of synthetic memories and conversation chunks, then times
search_context and save_memories (20 items) the way a server runs them: a fresh
connection per call, with the real embedding model, warm. Stored vectors are random
unit vectors (embedding 110k texts would take most of an hour, and search time
doesn't depend on what the vectors contain); queries and saved items use the model.

Exits non-zero if a p95 is over budget. Needs the model: run `openmemore embed` first,
or set OPENMEMORE_MODEL_DIR to a directory that has it.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import tempfile
import time
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from openmemore import db, embed, repo
from openmemore.audit import percentile
from openmemore.config import ClientScopes
from openmemore.models import MemoryInput, NewConversation, NewMemory, NewMessage
from openmemore.tools import Caller
from openmemore.tools.memories import content_hash, save_memories
from openmemore.tools.search import search_context

BUDGET_MS = {"search_context": 100.0, "save_memories (20 items)": 300.0}
TOPIC_WORDS = """
python rust go typescript flask django fastapi sqlite postgres redis kafka docker kubernetes
deploy staging production retry backoff jitter cache latency throughput budget invoice payment
running club marathon recipe peanut allergy vacation lisbon tokyo apartment flight hotel garden
plants window lamp sofa editor neovim theme keyboard monitor laptop battery backup restore
migration schema index query vector embedding search memory profile decision preference project
""".split()
# Natural text has a long-tailed vocabulary: a few words are everywhere, most are rare
# (Zipf's law). Sampling from 20k pseudo-words with Zipf weights gives the index
# realistic term frequencies; a small uniform vocabulary would put every word in
# nearly every chunk, a worst case no real store has.
VOCABULARY = TOPIC_WORDS + [f"w{i}x" for i in range(20_000)]
ZIPF_WEIGHTS = [1 / (rank + 10) for rank in range(len(VOCABULARY))]
QUERIES = [
    "what did I decide about retries", "my editor setup", "where am I travelling next",
    "how do deploys reach production", "food allergies", "database migration plan",
    "how do I like answers formatted", "the running club schedule", "cache invalidation approach",
    "laptop battery issue", "which vector index did we pick", "garden plants for the window",
]
CALLER = Caller(client="bench", scopes=ClientScopes(read_scopes=("personal",), write_scope="personal"))


def sentence(rng: random.Random, n: int) -> str:
    return " ".join(rng.choices(VOCABULARY, weights=ZIPF_WEIGHTS, k=n)).capitalize() + "."


def unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.standard_normal((n, embed.DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def build(path: Path, memories: int, chunks: int) -> None:
    rng, nprng = random.Random(0), np.random.default_rng(0)
    now = datetime.now(UTC)
    with closing(db.open_db(path)) as conn, db.transaction(conn):
        for i in range(memories):
            text = f"{sentence(rng, 12)} ({i})"
            when = now - timedelta(days=rng.randint(0, 365))
            repo.insert_memory(conn, NewMemory(
                kind=rng.choice(["fact", "preference", "project", "decision", "note"]), content=text,
                scope="personal", source_client="bench", content_hash=content_hash(text),
                created_at=when, updated_at=when,
            ))
        vectors = unit_vectors(nprng, memories)
        repo.store_vectors(conn, "memories", [(i + 1, vectors[i].tobytes()) for i in range(memories)])

        per_conversation, chunk_id = 20, 0
        vectors = unit_vectors(nprng, chunks)
        for c in range(chunks // per_conversation):
            when = now - timedelta(days=rng.randint(0, 365))
            cid = repo.insert_conversation(conn, NewConversation(
                provider="bench", external_id=str(c), title=sentence(rng, 5), scope="personal",
                created_at=when, updated_at=when,
            ))
            for m in range(per_conversation):
                text = sentence(rng, 120)
                role = "user" if m % 2 == 0 else "assistant"
                mid, _ = repo.insert_message(conn, NewMessage(
                    conversation_id=cid, role=role, content=text, created_at=when, content_hash=f"{c}-{m}",
                ))
                [new_chunk] = repo.insert_chunks(conn, mid, [text])
                repo.store_vectors(conn, "chunks", [(new_chunk, vectors[chunk_id].tobytes())])
                chunk_id += 1


def timed(path: Path, runs: int, call) -> list[float]:
    out = []
    for i in range(runs):
        start = time.perf_counter()
        with closing(db.connect(path)) as conn:  # a fresh connection per call, as in the server
            call(conn, i)
        out.append((time.perf_counter() - start) * 1000)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--memories", type=int, default=10_000)
    parser.add_argument("--chunks", type=int, default=100_000)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    embedder = embed.get_embedder()
    if embedder is None:
        print(f"The embedding model isn't in {embed.model_dir()}: run `openmemore embed` first.", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bench.db"
        start = time.perf_counter()
        build(path, args.memories, args.chunks)
        print(f"Built {args.memories:,} memories and {args.chunks:,} chunks in {time.perf_counter() - start:.0f}s")

        embedder.query("warm up")
        rng = random.Random(1)
        results = {
            "search_context": timed(path, args.runs, lambda conn, i: search_context(conn, CALLER, QUERIES[i % len(QUERIES)])),
            "save_memories (20 items)": timed(path, args.runs, lambda conn, i: save_memories(
                conn, CALLER, [MemoryInput(kind="note", content=f"{sentence(rng, 10)} {i}-{j}") for j in range(20)]
            )),
        }

    failed = False
    print(f"\n{'call':<26}{'p50 ms':>9}{'p95 ms':>9}{'budget':>9}")
    for name, ms in results.items():
        p95 = percentile(ms, 95)
        ok = p95 < BUDGET_MS[name]
        failed |= not ok
        print(f"{name:<26}{statistics.median(ms):>9.1f}{p95:>9.1f}{BUDGET_MS[name]:>9.0f}  {'ok' if ok else 'OVER BUDGET'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
