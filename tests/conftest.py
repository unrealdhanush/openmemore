import os
import re
import zlib
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from openmcp import db, embed, repo
from openmcp.config import ClientScopes
from openmcp.models import NewMemory
from openmcp.tools import Caller
from openmcp.tools.memories import content_hash


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """CLI output without terminal styling (rich colors output on CI, e.g. GitHub Actions)."""
    return ANSI_ESCAPE.sub("", text)


@pytest.fixture(autouse=True)
def isolated_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets an empty home directory, on every OS, so nothing (transcript
    capture, setup) can read or edit the real user's files. Path.home() reads HOME on
    macOS/Linux and USERPROFILE on Windows; app data dirs come from APPDATA and
    XDG_CONFIG_HOME."""
    user_home = tmp_path / "user-home"
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setenv("USERPROFILE", str(user_home))
    for var in ("APPDATA", "XDG_CONFIG_HOME", "CLAUDE_CONFIG_DIR", "OPENMCP_HOME"):
        monkeypatch.delenv(var, raising=False)
    return user_home


# Where the real embedding model is cached for tests marked `real_model`: outside the
# per-test home directories, so it downloads once per machine (CI caches it).
MODEL_CACHE = Path(os.environ.get("OPENMCP_TEST_MODEL_DIR") or Path.home() / ".cache" / "openmcp-test-models")


class FakeEmbedder:
    """Deterministic stand-in for the model: a bag of words hashed into DIM dimensions.
    Same words, same vector; shared words, similar vectors. `vectors` pins a text's
    vector exactly, for tests that need a specific similarity."""

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.vectors = vectors or {}
        self.calls = 0

    def _vector(self, text: str) -> bytes:
        self.calls += 1
        if text in self.vectors:
            v = np.array(self.vectors[text] + [0.0] * (embed.DIM - len(self.vectors[text])), dtype=np.float32)
        else:
            v = np.zeros(embed.DIM, dtype=np.float32)
            for word in re.findall(r"\w+", text.casefold()):
                v[zlib.crc32(word.encode()) % embed.DIM] += 1
        norm = np.linalg.norm(v)
        return (v / norm if norm else v).astype(np.float32).tobytes()

    def passages(self, texts):
        return [self._vector(t) for t in texts]

    def query(self, text):
        return self._vector(text)


@pytest.fixture(autouse=True)
def no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """By default there is no embedding model (keyword search only), as on a machine
    that hasn't run `openmcp embed`. Tests opt in with `fake_embedder` or
    `real_embedder`, whose patch runs after this one and wins."""
    monkeypatch.setattr(embed, "get_embedder", lambda allow_download=False: None)


@pytest.fixture
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> FakeEmbedder:
    fake = FakeEmbedder()
    monkeypatch.setattr(embed, "get_embedder", lambda allow_download=False: fake)
    return fake


@pytest.fixture
def real_embedder(monkeypatch: pytest.MonkeyPatch):
    """The real model (downloaded to MODEL_CACHE on first use). Skipped when it can't be
    downloaded, unless OPENMCP_REQUIRE_MODEL is set (CI), where that's a failure."""
    try:
        model = embed.FastEmbedder(MODEL_CACHE, allow_download=True)
    except Exception as e:  # noqa: BLE001
        if os.environ.get("OPENMCP_REQUIRE_MODEL"):
            raise
        pytest.skip(f"embedding model unavailable: {e}")
    monkeypatch.setattr(embed, "get_embedder", lambda allow_download=False: model)
    return model


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated OPENMCP_HOME so tests never touch ~/.openmcp."""
    base = tmp_path / "openmcp-home"
    monkeypatch.setenv("OPENMCP_HOME", str(base))
    return base


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def conn(tmp_path: Path):
    with closing(db.open_db(tmp_path / "t.db")) as c:
        yield c


def caller(client: str = "claude-desktop", read=("personal", "work"), write="personal") -> Caller:
    return Caller(client=client, scopes=ClientScopes(read_scopes=tuple(read), write_scope=write))


def insert(conn, content: str, *, kind="fact", scope="personal", project=None, subject=None,
           when: datetime | None = None, source="test") -> int:
    """Insert a memory directly, e.g. with a backdated timestamp."""
    when = when or datetime.now(UTC)
    return repo.insert_memory(conn, NewMemory(
        kind=kind, content=content, subject=subject, project=project, scope=scope, source_client=source,
        content_hash=content_hash(content), created_at=when, updated_at=when,
    ))
