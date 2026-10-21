"""ChatGPT data export: conversations.json, a list of conversations (SPEC.md Section 10).

Each conversation stores its messages as a tree in `mapping` (node id -> {message,
parent, children}): editing a message or regenerating an answer starts a new branch.
`current_node` is the last message of the branch the user ended on; walking its
parents up to the root and reversing gives that branch, and abandoned branches are
left out.

Kept: user and assistant messages addressed to everyone, as text. Skipped: system and
tool messages, hidden scaffolding, assistant calls to tools (recipient isn't 'all'),
custom instructions, memory and reasoning blocks, images. Formats change without
notice, so anything unrecognized is skipped rather than fatal.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from openmemore.ingest.registry import Importer
from openmemore.models import ParsedConversation, ParsedMessage

PROVIDER = "chatgpt"
TEXT_CONTENT = ("text", "multimodal_text")


def _when(seconds: Any) -> datetime | None:
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds <= 0:
        return None
    try:
        return datetime.fromtimestamp(seconds, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def updated(raw: Any) -> float:
    """Sort key: last activity, in seconds (0 if unknown)."""
    if not isinstance(raw, dict):
        return 0.0
    for key in ("update_time", "create_time"):
        if isinstance(raw.get(key), (int, float)) and not isinstance(raw.get(key), bool):
            return float(raw[key])
    return 0.0


def external_id(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    value = raw.get("conversation_id") or raw.get("id")
    return value if isinstance(value, str) and value else None


def _branch(mapping: dict[str, Any], current: Any) -> list[dict[str, Any]]:
    """Nodes from the root to `current`. Without a usable current_node, the most recent
    leaf stands in for it."""
    if not isinstance(current, str) or current not in mapping:
        leaves = [n for n in mapping.values() if isinstance(n, dict) and not n.get("children")]
        if not leaves:
            return []
        current = max(leaves, key=lambda n: (_message_time(n) or 0.0)).get("id")
    path, seen = [], set()
    while isinstance(current, str) and current in mapping and current not in seen:
        seen.add(current)
        node = mapping[current]
        if not isinstance(node, dict):
            break
        path.append(node)
        current = node.get("parent")
    return path[::-1]


def _message_time(node: dict[str, Any]) -> float | None:
    message = node.get("message")
    t = message.get("create_time") if isinstance(message, dict) else None
    return float(t) if isinstance(t, (int, float)) and not isinstance(t, bool) else None


def _role(message: dict[str, Any]) -> str | None:
    author = message.get("author")
    role = author.get("role") if isinstance(author, dict) else None
    return role if role in ("user", "assistant") else None


def _text(message: dict[str, Any]) -> str | None:
    """The visible text of a user or assistant message, or None if it isn't one to keep."""
    if _role(message) is None:
        return None
    if message.get("recipient", "all") != "all":  # an assistant's call to a tool (python, browser, ...)
        return None
    metadata = message.get("metadata") or {}
    if isinstance(metadata, dict) and metadata.get("is_visually_hidden_from_conversation"):
        return None
    content = message.get("content") or {}
    if not isinstance(content, dict) or content.get("content_type") not in TEXT_CONTENT:
        return None
    parts = content.get("parts")
    texts = [p for p in parts if isinstance(p, str)] if isinstance(parts, list) else []  # images are dicts
    text = "\n\n".join(t for t in texts if t.strip()).strip()
    return text or None


def conversation(raw: Any) -> ParsedConversation | None:
    """One exported conversation, or None if it has no id or no messages worth keeping."""
    conv_id = external_id(raw)
    mapping = raw.get("mapping") if conv_id else None
    if not isinstance(mapping, dict):
        return None
    messages = []
    for node in _branch(mapping, raw.get("current_node")):
        message = node.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("id") or node.get("id"), str):
            continue
        text = _text(message)
        if text is None:
            continue
        messages.append(
            ParsedMessage(
                external_id=message.get("id") or node["id"],
                role=_role(message),
                content=text,
                created_at=_when(message.get("create_time")),
            )
        )
    if not messages:
        return None
    title = raw.get("title")
    metadata: dict[str, Any] = {"capture": "import"}
    if isinstance(raw.get("default_model_slug"), str):
        metadata["model"] = raw["default_model_slug"]
    return ParsedConversation(
        provider=PROVIDER,
        external_id=conv_id,
        title=title if isinstance(title, str) and title.strip() else messages[0].content,
        created_at=_when(raw.get("create_time")) or messages[0].created_at,
        updated_at=_when(raw.get("update_time")) or messages[-1].created_at,
        metadata=metadata,
        messages=messages,
    )


def _identity(raw: Any) -> tuple[str, str] | None:
    conv_id = external_id(raw)
    return (PROVIDER, conv_id) if conv_id else None


# Lambdas look the functions up at call time, so tests can replace them.
IMPORTER = Importer(
    name=PROVIDER, display="ChatGPT",
    matches=lambda doc: isinstance(doc, list) and any(isinstance(c, dict) and "mapping" in c for c in doc),
    items=lambda doc: doc, updated=lambda raw: updated(raw), identity=lambda raw: _identity(raw),
    conversation=lambda raw: conversation(raw),
)
