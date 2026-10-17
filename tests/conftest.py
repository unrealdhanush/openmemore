from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest

from openmcp import db, repo
from openmcp.config import ClientScopes
from openmcp.models import NewMemory
from openmcp.tools import Caller
from openmcp.tools.memories import content_hash


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
