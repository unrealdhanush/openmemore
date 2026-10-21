"""Claude Code transcripts: one JSONL file per session under ~/.claude/projects/<project>/.

Each line is an event. The ones that matter here:
- `user` with string content (or text blocks) is a prompt. `user` lines holding only
  tool_result blocks, meta lines, compaction summaries, and slash-command scaffolding are not.
- `assistant` lines each carry one content block; the text blocks between two prompts
  form that turn's answer (tool calls and thinking are skipped).
- `ai-title` / `custom-title` / `summary` carry the session's title.
- `isSidechain` lines belong to subagents and are skipped.
- A turn whose assistant lines call OpenMemore's recall tools (tool_use blocks) is a recall turn.

The format is undocumented and changes between versions: unknown lines are ignored.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from openmemore.capture.common import is_recall_tool
from openmemore.models import ParsedConversation, ParsedMessage

PROVIDER = "claude-code"

_SCAFFOLDING = re.compile(
    r"^\s*(?:<(?:command-name|command-message|command-args|local-command-stdout|local-command-stderr|"
    r"local-command-caveat|bash-input|bash-stdout|bash-stderr|task-notification)>|\[Request interrupted)"
)
_SYSTEM_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)


def transcript_files(user_home: Path) -> Iterator[Path]:
    root = Path(os.environ.get("CLAUDE_CONFIG_DIR") or user_home / ".claude") / "projects"
    return root.glob("*/*.jsonl")


def _when(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _prompt_text(line: dict[str, Any]) -> str | None:
    """The user's prompt on a `user` line, or None if the line isn't one."""
    if line.get("isMeta") or line.get("isCompactSummary") or line.get("isVisibleInTranscriptOnly"):
        return None
    content = (line.get("message") or {}).get("content")
    if isinstance(content, list):
        content = "\n\n".join(
            b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
        )
    return clean_prompt(content) if isinstance(content, str) else None


def clean_prompt(content: str) -> str | None:
    """A prompt without system reminders, or None if it's slash-command or tool
    scaffolding rather than something the user wrote. Shared with OpenTelemetry capture."""
    text = _SYSTEM_REMINDER.sub("", content).strip()
    if not text or _SCAFFOLDING.match(text):
        return None
    return text


def _calls_recall_tool(line: dict[str, Any]) -> bool:
    content = (line.get("message") or {}).get("content")
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_use" and isinstance(b.get("name"), str) and is_recall_tool(b["name"])
        for b in content
    )


def _assistant_text(line: dict[str, Any]) -> str:
    content = (line.get("message") or {}).get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return "\n\n".join(
        b["text"].strip()
        for b in content
        if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str) and b["text"].strip()
    )


def parse(path: Path, offset: int = 0) -> tuple[ParsedConversation | None, int]:
    """Read a session transcript from byte `offset`, which must be the start of a line.

    Returns the conversation (None if nothing in this part of the file is a message)
    and the offset to resume from next time: the start of the last prompt, so a turn
    still being answered is re-read in full. A last line without a newline is still
    being written and is left for next time.
    """
    messages: list[ParsedMessage] = []
    title = custom_title = None
    first_ts = last_ts = None
    metadata: dict[str, Any] = {"capture": "transcript"}
    session_id = path.stem
    resume = offset

    prompt_id: str | None = None
    prompt: ParsedMessage | None = None
    answer_parts: list[str] = []
    answer_ts: datetime | None = None
    recall = False

    def finish_turn() -> None:
        if prompt is not None:
            prompt.recall = recall
        if prompt_id is not None and answer_parts:
            messages.append(
                ParsedMessage(
                    external_id=f"{prompt_id}:answer",
                    role="assistant",
                    content="\n\n".join(answer_parts),
                    created_at=answer_ts,
                    recall=recall,
                )
            )

    with path.open("rb") as f:
        f.seek(offset)
        pos = offset
        for raw in f:
            start, pos = pos, pos + len(raw)
            if not raw.endswith(b"\n"):
                break
            try:
                line = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(line, dict) or line.get("isSidechain"):
                continue
            kind = line.get("type")
            if kind == "ai-title" and isinstance(line.get("aiTitle"), str):
                title = line["aiTitle"]
                continue
            if kind == "custom-title" and isinstance(line.get("customTitle"), str):
                custom_title = line["customTitle"]
                continue
            if kind == "summary" and isinstance(line.get("summary"), str):
                title = title or line["summary"]
                continue
            if kind not in ("user", "assistant"):
                continue

            when = _when(line.get("timestamp"))
            if when is not None:
                first_ts = first_ts or when
                last_ts = when
            if isinstance(line.get("sessionId"), str):
                session_id = line["sessionId"]
            for key, field in (("cwd", "cwd"), ("gitBranch", "git_branch")):
                if isinstance(line.get(key), str) and line[key]:
                    metadata[field] = line[key]
            if "cwd" in metadata:
                metadata["folder"] = metadata["cwd"]

            if kind == "user":
                text = _prompt_text(line)
                if text is None or not isinstance(line.get("uuid"), str):
                    continue
                finish_turn()
                prompt_id, answer_parts, answer_ts, recall, resume = line["uuid"], [], None, False, start
                prompt = ParsedMessage(external_id=prompt_id, role="user", content=text, created_at=when)
                messages.append(prompt)
            elif prompt_id is not None:
                recall = recall or _calls_recall_tool(line)
                text = _assistant_text(line)
                if text:
                    answer_parts.append(text)
                    answer_ts = answer_ts or when
        finish_turn()

    if not messages:
        return None, resume
    if offset == 0 and not (custom_title or title):
        title = messages[0].content
    return (
        ParsedConversation(
            provider=PROVIDER,
            external_id=session_id,
            title=custom_title or title,
            created_at=first_ts if offset == 0 else None,
            updated_at=last_ts,
            metadata=metadata,
            messages=messages,
        ),
        resume,
    )
