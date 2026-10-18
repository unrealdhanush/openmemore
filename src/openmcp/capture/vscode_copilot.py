"""VS Code Copilot Chat sessions: one JSON file per chat, rewritten whole as it grows.

Workspace chats live in <user dir>/workspaceStorage/<hash>/chatSessions/<id>.json and
chats with no folder open in <user dir>/globalStorage/emptyWindowChatSessions/. Each
file holds `requests`; each request has the prompt (`message.text`), a millisecond
`timestamp`, and `response` parts: markdown (`value`), inline file references, and
tool calls (skipped).

Checked against VS Code 1.99. The format is undocumented: unknown parts are ignored.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path, PurePath
from typing import Any

from openmcp.models import ParsedConversation, ParsedMessage
from openmcp.paths import app_data_dir

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
        prompt = (r.get("message") or {}).get("text")
        if isinstance(prompt, str) and prompt.strip():
            messages.append(
                ParsedMessage(external_id=r["requestId"], role="user", content=prompt.strip(), created_at=when)
            )
        answer = _response_text(r.get("response"))
        if answer:
            answer_id = r.get("responseId") if isinstance(r.get("responseId"), str) else f"{r['requestId']}:response"
            messages.append(ParsedMessage(external_id=answer_id, role="assistant", content=answer, created_at=when))
    if not messages:
        return None, 0

    metadata: dict[str, Any] = {"capture": "transcript"}
    if workspace := _workspace(path):
        metadata["workspace"] = workspace
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
