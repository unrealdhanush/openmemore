# OpenMCP

> Working name. Rename before any public release.

A local-first MCP server that gives every AI assistant you use one shared, user-owned
context store on your own machine. See [SPEC.md](SPEC.md) for the full design.

**Status:** early development (milestones M1 and M1.5: memory store, keyword search, and capturing
conversations across assistants). Not ready for use.

## Development

Requires [uv](https://docs.astral.sh/uv/). The project pins a uv-managed Python
(3.12) because `sqlite-vec` needs SQLite extension loading, which some system
Python builds disable.

```sh
uv sync
uv run openmcp init                       # creates ~/.openmcp (override with OPENMCP_HOME)
uv run openmcp serve --client my-client   # MCP server over stdio
uv run openmcp setup claude-desktop       # connect a client (also vscode-copilot, claude-code); --dry-run first
uv run openmcp capture                    # catch up on transcripts of clients with capture on
uv run pytest
```
