"""What OpenMCP asks connected models to do (SPEC.md Section 7.10).

The same guidance goes to two places: the MCP server instructions, and the standing
instruction `openmcp setup` installs where each client reads user instructions. The
M1 manual test showed models ignore tool descriptions and server instructions on
their own (no record_exchange calls until the user added a standing instruction).

Clients whose transcripts are captured from disk (SPEC.md M1.5) get a variant
without record_exchange: their conversations are stored without the model's help.
"""

from __future__ import annotations

STANDING_PREFIX = "Use the openmcp tools"
"""First words of every standing instruction, used to find and replace an earlier copy."""


def standing_instruction(captured: bool) -> str:
    """Written as the user, for the place a client reads the user's own instructions."""
    lines = [
        f"{STANDING_PREFIX} (my memory, shared with all my AI assistants):",
        "- At the start of each conversation, call get_profile.",
    ]
    if captured:
        lines.append("- Don't call record_exchange: conversations here are captured automatically.")
    else:
        lines.append(
            "- After every answer, call record_exchange with my question and a short summary of your answer, "
            "putting any code you wrote in its `code` field. Reuse the conversation_id it returns for the rest "
            "of the chat. Skip greetings, thanks, and turns where you only looked up earlier conversations."
        )
    lines += [
        '- When I make a decision, also save it with save_memories (kind "decision").',
        "- When I mention an earlier conversation, including one with another assistant, call "
        "recent_conversations or search_context before saying you don't know. Search for the topic itself "
        "(\"cat names\", not \"the pet question I asked Copilot\"); to narrow by assistant, use "
        "recent_conversations with source.",
        "- Never record secrets or credentials.",
    ]
    return "\n".join(lines)


def server_instructions(captured: bool) -> str:
    """The MCP `instructions` sent to every model that connects."""
    lines = [
        "OpenMCP is the user's local, user-owned memory, shared across all of their AI assistants "
        "(for example Claude, Cursor, and Gemini). What you store here, the user's other assistants "
        "can read, and vice versa.",
        "- At the start of a conversation, call get_profile to load what the user has already told "
        "their assistants.",
    ]
    if captured:
        lines.append(
            "- This client's conversations are captured automatically from its transcripts, so "
            "there is no need to record exchanges."
        )
    else:
        lines.append(
            "- After every answer, call record_exchange with the user's question and a short summary of "
            "your answer, including any code you wrote, so another assistant can pick up where you left "
            "off. Reuse the conversation_id from your first record_exchange call for later turns of the "
            "same chat. Skip greetings, thanks, and turns where you only looked up earlier conversations: "
            "a recap of a conversation is not worth storing beside the conversation itself."
        )
    lines += [
        "- When the user makes a decision, also save it with save_memories (kind 'decision'), so it "
        "outranks options that were discussed before it.",
        "- When the user refers to an earlier conversation, including one with a different assistant "
        "('what did I ask Claude earlier?', 'continue what I started in Cursor'), call "
        "recent_conversations or search_context before saying you don't know. Search for the topic itself "
        "('cat names', not 'the pet question I asked Copilot'); to narrow by assistant, use "
        "recent_conversations with source.",
        "- Save other lasting preferences, facts, and project state with save_memories.",
        "Never record secrets, credentials, or other people's private details.",
    ]
    return "\n".join(lines)
