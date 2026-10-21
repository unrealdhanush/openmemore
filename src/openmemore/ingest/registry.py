"""The importer registry: every export format OpenMemore can read.

An importer recognizes its format from the parsed contents (never the file name), lists
its raw conversations, and turns one into a ParsedConversation. Adding a provider means
writing an Importer (with synthetic fixtures modeled on a real export) and registering
it: built-in ones are listed in importers(), others call register(). Detection,
resumable jobs, newest-first ordering, and embedding work unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openmemore.models import ParsedConversation, ParsedMemory


@dataclass(frozen=True)
class Importer:
    name: str
    """Recorded on the import job, e.g. 'chatgpt'."""
    display: str
    matches: Callable[[Any], bool]
    """Whether a parsed JSON document is this format."""
    items: Callable[[Any], list[Any]]
    """The document's raw conversations."""
    updated: Callable[[Any], float]
    """A raw conversation's last activity, in seconds (0 if unknown), for newest-first order."""
    identity: Callable[[Any], tuple[str, str] | None]
    """A raw conversation's (provider, id), as stored."""
    conversation: Callable[[Any], ParsedConversation | None]
    """A raw conversation, parsed; None if there's nothing to keep."""
    memories: Callable[[Any], tuple[list[ParsedMemory], list[str]]] | None = None
    """The document's memories, and why any were skipped (OpenMemore exports have them)."""


_registered: list[Importer] = []


def register(importer: Importer) -> None:
    """Add an importer beyond the built-in ones (e.g. from a plugin)."""
    if any(i.name == importer.name for i in importers()):
        raise ValueError(f"an importer named {importer.name!r} is already registered")
    _registered.append(importer)


def importers() -> list[Importer]:
    """Every importer, built-in first; detection uses the first whose `matches` is true."""
    from openmemore.ingest import chatgpt, claude, openmemore_format

    return [chatgpt.IMPORTER, claude.IMPORTER, openmemore_format.IMPORTER, *_registered]


def importer_named(name: str) -> Importer | None:
    return next((i for i in importers() if i.name == name), None)
