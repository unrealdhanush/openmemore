"""Embeddings for semantic search: fastembed with BAAI/bge-small-en-v1.5.

The model never loads at server startup: importing fastembed
alone takes ~200ms. It loads on the first call that needs it, once per process.

Servers never download the model, since a download is far too slow for a tool call.
Without it on disk they fall back to keyword search and leave new rows pending;
`openmemore embed` (also run by `setup` and `capture`) downloads it and embeds what's
pending.

Vectors are unit-length float32, stored as raw bytes in the sqlite-vec tables.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from openmemore import config, db, repo
from openmemore.models import VECTOR_TABLES, VectorTable

MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384
BATCH = 32
WRITE_BUDGET_MS = 200
"""How long a tool call that wrote rows spends embedding pending ones, newest first."""


class Embedder(Protocol):
    def passages(self, texts: Sequence[str]) -> list[bytes]:
        """Vectors for stored text (memories, chunks)."""
        ...

    def query(self, text: str) -> bytes:
        """The vector for a search query (bge models embed queries differently from passages)."""
        ...


class FastEmbedder:
    def __init__(self, model_dir: Path, allow_download: bool) -> None:
        from fastembed import TextEmbedding  # ~200ms; only when embeddings are needed

        self._model = TextEmbedding(MODEL, cache_dir=str(model_dir), local_files_only=not allow_download)

    def passages(self, texts: Sequence[str]) -> list[bytes]:
        return [v.astype("float32").tobytes() for v in self._model.passage_embed(list(texts), batch_size=BATCH)]

    def query(self, text: str) -> bytes:
        return next(iter(self._model.query_embed([text]))).astype("float32").tobytes()


def model_dir() -> Path:
    """Where the model is cached: $OPENMEMORE_MODEL_DIR, else <data dir>/models."""
    env = os.environ.get("OPENMEMORE_MODEL_DIR")
    return Path(env).expanduser() if env else config.home() / "models"


_lock = threading.Lock()
_embedder: Embedder | None = None
_unavailable: str | None = None


def get_embedder(allow_download: bool = False) -> Embedder | None:
    """The process-wide embedder, or None if the model can't be loaded (not downloaded
    yet, or broken). A failure is remembered, so a server checks the disk only once."""
    global _embedder, _unavailable
    with _lock:
        if _embedder is None and (_unavailable is None or allow_download):
            try:
                _embedder = FastEmbedder(model_dir(), allow_download)
                _unavailable = None
            except Exception as e:  # noqa: BLE001 - any failure means "keyword search only"
                _unavailable = f"{type(e).__name__}: {e}"
                print(f"openmemore: semantic search unavailable ({_unavailable}); run `openmemore embed`", file=sys.stderr)
        return _embedder


def status() -> str:
    """For `ping`: whether semantic search is available in this process."""
    if get_embedder() is not None:
        return f"ready ({MODEL})"
    return "unavailable: the embedding model isn't downloaded; run `openmemore embed`"


def memory_text(subject: str | None, content: str) -> str:
    """What gets embedded for a memory: the subject adds context to short statements."""
    return f"{subject}: {content}" if subject else content


def similarity(distance: float) -> float:
    """sqlite-vec's cosine distance -> cosine similarity."""
    return 1.0 - distance


@dataclass
class EmbedReport:
    memories: int = 0
    chunks: int = 0
    remaining: int = 0
    """Rows still pending: the budget ran out, or there was no model."""


def embed_pending(
    conn: sqlite3.Connection,
    embedder: Embedder | None,
    *,
    budget_ms: float | None = None,
    progress: Callable[[int], None] | None = None,
    tables: Sequence[VectorTable] = VECTOR_TABLES,
) -> EmbedReport:
    """Embed memories and chunks that don't have vectors yet, newest first, in short
    transactions. With `budget_ms`, stop starting new batches once that much time has passed."""
    start = time.perf_counter()
    report = EmbedReport()
    for table in tables:
        while embedder is not None:
            if budget_ms is not None and (time.perf_counter() - start) * 1000 > budget_ms:
                break
            rows = repo.pending_embeddings(conn, table, BATCH)
            if not rows:
                break
            vectors = embedder.passages([memory_text(subject, text) for _, subject, text in rows])
            with db.transaction(conn):
                stored = repo.store_vectors(conn, table, [(row[0], v) for row, v in zip(rows, vectors)])
            if table == "memories":
                report.memories += stored
            else:
                report.chunks += stored
            if progress:
                progress(len(rows))
    report.remaining = repo.count_pending_embeddings(conn)
    return report
