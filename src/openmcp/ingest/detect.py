"""Opening an export (SPEC.md Section 10): the .zip a provider sends, the
conversations.json inside it, or a folder it was unpacked into. The provider is
recognized from the contents, not the file name."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from openmcp.ingest import chatgpt, claude
from openmcp.models import ExportProvider

PARSERS: dict[ExportProvider, ModuleType] = {"chatgpt": chatgpt, "claude": claude}


class ExportError(ValueError):
    """Not an export this version can read; the message says why."""


@dataclass
class Export:
    provider: ExportProvider
    conversations: list[Any]
    """Raw conversations as exported; the provider's parser reads one at a time."""
    file_hash: str


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _conversations_file(path: Path) -> tuple[bytes, Path]:
    """The bytes of conversations.json, and the file whose hash identifies the export."""
    if path.is_dir():
        candidate = path / "conversations.json"
        if not candidate.is_file():
            raise ExportError(f"{path} has no conversations.json")
        return candidate.read_bytes(), candidate
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.rsplit("/", 1)[-1] == "conversations.json"]
            if not names:
                raise ExportError(f"{path.name} has no conversations.json: is it a ChatGPT or Claude data export?")
            return z.read(min(names, key=len)), path
    return path.read_bytes(), path


def detect(conversations: list[Any]) -> ExportProvider:
    for raw in conversations:
        if isinstance(raw, dict):
            if "mapping" in raw:
                return "chatgpt"
            if "chat_messages" in raw:
                return "claude"
    raise ExportError("conversations.json doesn't look like a ChatGPT or Claude export")


def load(path: Path) -> Export:
    if not path.exists():
        raise ExportError(f"{path} doesn't exist")
    data, identity = _conversations_file(path)
    try:
        conversations = json.loads(data)
    except ValueError as e:
        raise ExportError(f"conversations.json isn't valid JSON: {e}") from None
    if not isinstance(conversations, list):
        raise ExportError("conversations.json should hold a list of conversations")
    return Export(provider=detect(conversations), conversations=conversations, file_hash=file_hash(identity))
