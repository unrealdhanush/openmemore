from pathlib import Path

import pytest


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated OPENMCP_HOME so tests never touch ~/.openmcp."""
    base = tmp_path / "openmcp-home"
    monkeypatch.setenv("OPENMCP_HOME", str(base))
    return base


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
