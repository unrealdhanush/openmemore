import os
import re
import zlib
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from mcp import Client
from mcp.types import Implementation

from openmemore import config, db, embed, repo
from openmemore import setup as client_setup
from openmemore.config import ClientScopes
from openmemore.models import NewMemory
from openmemore.tools import Caller
from openmemore.tools.memories import content_hash


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
    for var in ("APPDATA", "XDG_CONFIG_HOME", "CLAUDE_CONFIG_DIR", "OPENMEMORE_HOME"):
        monkeypatch.delenv(var, raising=False)
    return user_home


# Where the real embedding model is cached for tests marked `real_model`: outside the
# per-test home directories, so it downloads once per machine (CI caches it).
MODEL_CACHE = Path(os.environ.get("OPENMEMORE_TEST_MODEL_DIR") or Path.home() / ".cache" / "openmemore-test-models")


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
    that hasn't run `openmemore embed`. Tests opt in with `fake_embedder` or
    `real_embedder`, whose patch runs after this one and wins."""
    monkeypatch.setattr(embed, "get_embedder", lambda allow_download=False: None)


REAL_DEVELOPMENT_CHECKOUT = client_setup.development_checkout
REAL_COPY_TO_CLIPBOARD = client_setup.copy_to_clipboard


@pytest.fixture(autouse=True)
def clipboard(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Never touch the real clipboard: `openmemore setup` copies the standing instruction.
    Copies are recorded here, and report success."""
    copied: list[str] = []
    monkeypatch.setattr(client_setup, "copy_to_clipboard", lambda text: copied.append(text) or True)
    return copied


@pytest.fixture(autouse=True)
def installed_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests run from a development checkout, so `openmemore setup` would hand off to the
    machine's installed copy. By default, behave like an installed copy instead."""
    monkeypatch.setattr(client_setup, "development_checkout", lambda: None)


@pytest.fixture
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> FakeEmbedder:
    fake = FakeEmbedder()
    monkeypatch.setattr(embed, "get_embedder", lambda allow_download=False: fake)
    return fake


@pytest.fixture
def real_embedder(monkeypatch: pytest.MonkeyPatch):
    """The real model (downloaded to MODEL_CACHE on first use). Skipped when it can't be
    downloaded, unless OPENMEMORE_REQUIRE_MODEL is set (CI), where that's a failure."""
    try:
        model = embed.FastEmbedder(MODEL_CACHE, allow_download=True)
    except Exception as e:  # noqa: BLE001
        if os.environ.get("OPENMEMORE_REQUIRE_MODEL"):
            raise
        pytest.skip(f"embedding model unavailable: {e}")
    monkeypatch.setattr(embed, "get_embedder", lambda allow_download=False: model)
    return model


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated OPENMEMORE_HOME so tests never touch ~/.openmemore."""
    base = tmp_path / "openmemore-home"
    monkeypatch.setenv("OPENMEMORE_HOME", str(base))
    return base


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def conn(tmp_path: Path):
    with closing(db.open_db(tmp_path / "t.db")) as c:
        yield c


def connect(home: Path, app: str = "claude-desktop") -> Client:
    """An in-process MCP session with a server for `home`, as the application `app`
    (which is also the client name, since app names are already config-friendly here)."""
    from openmemore.server import build_server

    return Client(build_server(home), client_info=Implementation(name=app, version="test"))


def example_clients(home: Path) -> None:
    """Scopes for the scope tests (once the shipped defaults): 'claude-desktop' reads
    personal and work, 'cursor' reads only work."""
    path = home / "config.toml"
    cfg = config.load_config(path)
    cfg.clients["claude-desktop"] = config.ClientOverride(read_scopes=("personal", "work"), write_scope="personal")
    cfg.clients["cursor"] = config.ClientOverride(read_scopes=("work",), write_scope="work")
    config.save_config(path, cfg)


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


@pytest.fixture(autouse=True)
def consent_answers(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Never show a real permission dialog or prompt: OpenMemore's own consent question
    (openmemore.consent.ask) answers 'denied' unless a test appends another answer here."""
    from openmemore import consent

    answers: list[str] = []
    monkeypatch.setattr(consent, "ask", lambda *a, **k: answers.pop(0) if answers else "denied")
    return answers
