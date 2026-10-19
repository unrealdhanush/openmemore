"""Opening an export (SPEC.md Section 10): the .zip a provider sends, the
conversations.json inside it, an unpacked folder, or any JSON file in a registered
format. The format is recognized from the contents by the importer registry, never from
the file name."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openmcp.ingest import registry
from openmcp.ingest.registry import Importer
from openmcp.models import ParsedMemory


class ExportError(ValueError):
    """Not an export this version can read; the message says why."""


@dataclass
class Export:
    importer: Importer
    conversations: list[Any]
    """Raw conversations as exported; the importer parses one at a time."""
    file_hash: str
    memories: list[ParsedMemory] = field(default_factory=list)
    memory_errors: list[str] = field(default_factory=list)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _document_file(path: Path) -> tuple[bytes, Path]:
    """The bytes of the export's JSON document, and the file whose hash identifies it."""
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


def recognize(doc: Any) -> Importer:
    for importer in registry.importers():
        if importer.matches(doc):
            return importer
    known = ", ".join(i.display for i in registry.importers())
    raise ExportError(f"this doesn't look like a format OpenMCP can import (it reads: {known})")


def load(path: Path) -> Export:
    if not path.exists():
        raise ExportError(f"{path} doesn't exist")
    data, identity = _document_file(path)
    try:
        doc = json.loads(data)
    except ValueError as e:
        raise ExportError(f"it isn't valid JSON: {e}") from None
    if not isinstance(doc, (list, dict)):
        raise ExportError("it should hold a list of conversations, or an OpenMCP conversations document")
    importer = recognize(doc)
    try:
        conversations = importer.items(doc)
        memories, memory_errors = importer.memories(doc) if importer.memories else ([], [])
    except ValueError as e:
        raise ExportError(str(e)) from None
    return Export(
        importer=importer, conversations=conversations, file_hash=file_hash(identity),
        memories=memories, memory_errors=memory_errors,
    )
