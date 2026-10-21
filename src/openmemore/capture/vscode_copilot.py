"""VS Code Copilot Chat sessions: one JSON file per chat, rewritten whole as it grows.

Workspace chats live in <user dir>/workspaceStorage/<hash>/chatSessions/<id>.json and
chats with no folder open in <user dir>/globalStorage/emptyWindowChatSessions/. Each
file holds `requests`; each request has the prompt (`message.text`), a millisecond
`timestamp`, and `response` parts: markdown (`value`), inline file references, and
tool calls (skipped, except that calling OpenMemore's recall tools makes it a recall turn).

Checked against VS Code 1.99. The format is undocumented: unknown parts are ignored.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path, PurePath
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname
from typing import Any

from openmemore.capture.common import RECALL_TOOLS, is_recall_tool
from openmemore.models import ParsedConversation, ParsedMessage
from openmemore.paths import app_data_dir

PROVIDER = "vscode-copilot"


def user_dir(user_home: Path, platform: str = sys.platform) -> Path:
    return app_data_dir(user_home, platform) / "Code" / "User"


def transcript_files(user_home: Path) -> Iterator[Path]:
    base = user_dir(user_home)
    yield from base.glob("workspaceStorage/*/chatSessions/*.json")
    yield from base.glob("globalStorage/emptyWindowChatSessions/*.json")


def _when(ms: Any) -> datetime | None:
    if not isinstance(ms, (int, float)) or isinstance(ms, bool):
        return None
    try:
        return datetime.fromtimestamp(ms / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _response_text(parts: Any) -> str:
    if not isinstance(parts, list):
        return ""
    out = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        kind = part.get("kind")
        if kind is None and isinstance(part.get("value"), str):
            out.append(part["value"])
        elif kind == "markdownContent" and isinstance((part.get("content") or {}).get("value"), str):
            out.append(part["content"]["value"])
        elif kind == "inlineReference":
            ref = part.get("inlineReference") or {}
            name = part.get("name") or ref.get("name") or ref.get("path") or ref.get("fsPath")
            if isinstance(name, str) and name:
                out.append(f"`{PurePath(name).name}`")
    return "".join(out).strip()


def _calls_recall_tool(request: dict[str, Any]) -> bool:
    """From the tool calls in result.metadata.toolCallRounds ('590_recent_conversations'),
    or failing that, the 'Running `recent_conversations`' messages in the response."""
    metadata = (request.get("result") or {}).get("metadata") or {}
    rounds = metadata.get("toolCallRounds") if isinstance(metadata, dict) else None
    for round_ in rounds if isinstance(rounds, list) else []:
        calls = round_.get("toolCalls") if isinstance(round_, dict) else None
        for call in calls if isinstance(calls, list) else []:
            if isinstance(call, dict) and isinstance(call.get("name"), str) and is_recall_tool(call["name"]):
                return True
    for part in request.get("response") or []:
        if isinstance(part, dict) and part.get("kind") == "toolInvocationSerialized":
            message = _markdown(part.get("invocationMessage"))
            if message and any(f"`{tool}`" in message for tool in RECALL_TOOLS):
                return True
    return False


def _markdown(value: Any) -> str | None:
    """VS Code stores messages as plain strings for some tools (file edits) and as
    {"value": "..."} markdown objects for others."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("value"), str):
        return value["value"]
    return None


def _workspace(path: Path) -> str | None:
    """The folder or workspace a chat belongs to, from workspaceStorage/<hash>/workspace.json."""
    try:
        info = json.loads((path.parent.parent / "workspace.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(info, dict):
        return None
    value = info.get("folder") or info.get("workspace")
    return value if isinstance(value, str) else None


def folder_path(uri: str) -> str | None:
    """A local folder from VS Code's workspace URI ('file:///Users/me/app' -> '/Users/me/app').
    Remote ones (vscode-remote://...) have no local path."""
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    return url2pathname(parsed.path) if os.name == "nt" else unquote(parsed.path)


def parse(path: Path, offset: int = 0) -> tuple[ParsedConversation | None, int]:
    """Read a whole chat session (`offset` is ignored: VS Code rewrites the file each time)."""
    data = json.loads(path.read_bytes())
    if not isinstance(data, dict):
        return None, 0
    messages: list[ParsedMessage] = []
    requests = data.get("requests")
    for r in requests if isinstance(requests, list) else []:
        if not isinstance(r, dict) or not isinstance(r.get("requestId"), str):
            continue
        when = _when(r.get("timestamp"))
        recall = _calls_recall_tool(r)
        prompt = (r.get("message") or {}).get("text")
        if isinstance(prompt, str) and prompt.strip():
            messages.append(
                ParsedMessage(
                    external_id=r["requestId"], role="user", content=prompt.strip(), created_at=when, recall=recall
                )
            )
        answer = _response_text(r.get("response"))
        if answer:
            answer_id = r.get("responseId") if isinstance(r.get("responseId"), str) else f"{r['requestId']}:response"
            messages.append(
                ParsedMessage(external_id=answer_id, role="assistant", content=answer, created_at=when, recall=recall)
            )
    if not messages:
        return None, 0

    metadata: dict[str, Any] = {"capture": "transcript"}
    if workspace := _workspace(path):
        metadata["workspace"] = workspace
        if folder := folder_path(workspace):
            metadata["folder"] = folder
    title = data.get("customTitle")
    return (
        ParsedConversation(
            provider=PROVIDER,
            external_id=data["sessionId"] if isinstance(data.get("sessionId"), str) else path.stem,
            title=title if isinstance(title, str) and title.strip() else messages[0].content,
            created_at=_when(data.get("creationDate")) or messages[0].created_at,
            updated_at=_when(data.get("lastMessageDate")) or messages[-1].created_at,
            metadata=metadata,
            messages=messages,
        ),
        0,
    )
