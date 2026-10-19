"""Shared by the transcript parsers."""

from __future__ import annotations

RECALL_TOOLS = ("recent_conversations", "search_context")
"""OpenMCP tools that look up earlier conversations. A captured turn that called one is
a recall turn (SPEC.md 7.2), ranked below the conversations it recaps."""


def is_recall_tool(name: str) -> bool:
    """Clients prefix MCP tool names with the server: 'mcp__openmcp__search_context'
    (Claude Code), '590_recent_conversations' (VS Code)."""
    return any(name == tool or name.endswith("_" + tool) for tool in RECALL_TOOLS)
