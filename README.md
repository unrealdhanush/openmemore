# OpenMCP

> Working name. Rename before any public release.

A local-first MCP server that gives every AI assistant you use one shared, user-owned
context store on your own machine. See [SPEC.md](SPEC.md) for the full design.

**Status:** early development (milestone M1: memory store, keyword search). Not ready for use.

## Development

Requires [uv](https://docs.astral.sh/uv/). The project pins a uv-managed Python
(3.12) because `sqlite-vec` needs SQLite extension loading, which some system
Python builds disable.

```sh
uv sync
uv run openmcp init                       # creates ~/.openmcp (override with OPENMCP_HOME)
uv run openmcp serve --client my-client   # MCP server over stdio
uv run pytest
```
