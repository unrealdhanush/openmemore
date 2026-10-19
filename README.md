# OpenMCP

> Working name. Rename before any public release.

A local-first MCP server that gives every AI assistant you use one shared, user-owned
context store on your own machine. See [SPEC.md](SPEC.md) for the full design.

**Status:** early development (milestones M1 to M3: memory store, capturing conversations across
assistants, semantic search, importing ChatGPT and Claude exports). Not ready for use.

## Development

Requires [uv](https://docs.astral.sh/uv/). The project pins a uv-managed Python
(3.12) because `sqlite-vec` needs SQLite extension loading, which some system
Python builds disable.

```sh
uv sync
uv run openmcp init                       # creates ~/.openmcp (override with OPENMCP_HOME)
uv run openmcp serve --client my-client   # MCP server over stdio
uv run openmcp clients                    # clients setup knows; any other MCP client works too
uv run openmcp setup claude-desktop       # connect a client; --dry-run first; an unknown name prints a generic snippet
uv run openmcp capture                    # catch up on transcripts of clients with capture on
uv run openmcp embed                      # download the embedding model (once) and embed what's pending
uv run openmcp import ~/Downloads/export.zip  # a ChatGPT or Claude data export; resumable
uv run openmcp status                     # import jobs and what's stored
uv run pytest                             # real-model tests download ~70MB once to ~/.cache/openmcp-test-models
uv run python scripts/benchmark.py        # check the performance budget (SPEC.md 4.3)
```
