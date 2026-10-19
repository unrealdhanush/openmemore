"""Synthetic data exports of any size, for import tests (never real data)."""

import json
import zipfile
from pathlib import Path

START = 1_780_000_000  # 2026-05-28


def chatgpt_conversation(i: int, *, turns: int = 2, extra: str = "") -> dict:
    """Conversation `i` mentions the unique word f"topic{i}"; later i means more recent."""
    mapping, parent = {"root": {"id": "root", "message": None, "parent": None, "children": []}}, "root"
    t = START + i * 3600
    for turn in range(turns):
        for role, text in (("user", f"Question {turn} about topic{i}{extra}"), ("assistant", f"Answer {turn} about topic{i}")):
            node_id = f"c{i}-{role}-{turn}"
            mapping[node_id] = {
                "id": node_id, "parent": parent, "children": [],
                "message": {"id": node_id, "author": {"role": role}, "create_time": t, "recipient": "all",
                            "content": {"content_type": "text", "parts": [text]}, "metadata": {}},
            }
            mapping[parent]["children"].append(node_id)
            parent, t = node_id, t + 1
    return {"title": f"Conversation {i}", "create_time": START + i * 3600, "update_time": t,
            "mapping": mapping, "current_node": parent, "conversation_id": f"conv-{i}", "id": f"conv-{i}"}


def chatgpt_export(n: int, **kwargs) -> list[dict]:
    return [chatgpt_conversation(i, **kwargs) for i in range(n)]


def write_zip(path: Path, conversations: list[dict], folder: str = "") -> Path:
    """A zip shaped like a real export: conversations.json plus other files."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{folder}conversations.json", json.dumps(conversations))
        z.writestr(f"{folder}user.json", json.dumps({"id": "user-test"}))
        z.writestr(f"{folder}chat.html", "<html></html>")
    return path
