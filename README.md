# OpenMCP

> Working name. Rename before any public release.

One shared memory for all your AI assistants, kept on your own machine.

Ask Claude something today and ask Cursor about it tomorrow: it knows what you mean. OpenMCP is a local [MCP](https://modelcontextprotocol.io) server that every assistant you use connects to. Each one can save what it learns, record what you asked and what it answered, and read what the others stored. It's all in one SQLite file on your disk, which you can list, edit, export and delete.

**Status:** pre-release (v1 is milestones M0 to M4 of [SPEC.md](SPEC.md); M5, consolidation and review, is in progress). The repository is private, and there's no published package yet.

## How it works

- **One store:** `~/.openmcp/openmcp.db`, a single SQLite file. Each assistant starts its own `openmcp serve` process and they all share the file.
- **What assistants can do:**
  - `get_profile`: what you've told your assistants before
  - `save_memories`: preferences, facts, decisions
  - `record_exchange`: what you asked and what it answered, including code
  - `recent_conversations` and `search_context`: find earlier conversations, with any assistant, by meaning as well as by words
  - `update_memory` and `forget`
  - `consolidation_batch` and `propose_memories`: turn conversations into proposed memories, for you to review
- **Capture:** an assistant records an exchange by calling `record_exchange`, which a short instruction you paste in asks it to do. For VS Code Copilot and Claude Code, OpenMCP can instead read the conversations they already save on disk, so nothing depends on the model.
- **Search:** keyword search (SQLite FTS5) combined with meaning-based search, using a small local embedding model (`BAAI/bge-small-en-v1.5`, about 70MB, downloaded once). Nothing leaves your machine.
- **History:** import your ChatGPT and Claude data exports, or any provider's history converted to OpenMCP's own format. Export everything back out in that same format.

## Install

Requires [uv](https://docs.astral.sh/uv/). uv provides Python 3.12, so you don't need your own; some system Pythons can't load SQLite extensions. OpenMCP runs on macOS, Linux and Windows; CI tests all three.

```sh
uv tool install --managed-python --python 3.12 "openmcp @ git+https://github.com/unrealdhanush/openmcp@main"
openmcp init      # creates ~/.openmcp (set OPENMCP_HOME to put it elsewhere)
openmcp embed     # downloads the embedding model once, so search works by meaning
```

Upgrade with `uv tool upgrade openmcp`, then restart your assistants: each one loads OpenMCP when it starts.

## Connect your assistants

```sh
openmcp setup cursor          # or vscode-copilot, claude-code, or any other client's name
openmcp setup --dry-run       # see what it would do first
```

Any MCP client works, with no client-specific support needed. The server is the same for every client:

```json
{ "mcpServers": { "openmcp": { "command": "/path/to/openmcp", "args": ["serve"] } } }
```

`openmcp setup` never edits another app's files. Where the app has its own way to add an MCP server, setup uses it:

| Client | How setup adds the server |
|---|---|
| VS Code (Copilot) | `code --add-mcp`. Needs `code` on your PATH (in VS Code, run "Shell Command: Install 'code' command in PATH") |
| Cursor | Cursor's install link, which opens its own review prompt |
| Claude Code | `claude mcp add` |
| Claude Desktop, and everything else | Prints the snippet above. Paste it into the client's MCP config (Claude Desktop: Settings → Developer → Edit Config), or run `openmcp setup <client> --config <that file>` to merge it in |

**The standing instruction.** Assistants don't reliably use tools just because they're available, so setup copies a short instruction to your clipboard. Paste it wherever the client reads your own instructions, for example Claude Desktop's Settings → Profile, or Cursor's User Rules. It points to the server's own instructions, which update with OpenMCP, so you only paste it once.

Restart the client after setup. `openmcp clients` then lists each client that has connected, under the name its app reports (Claude Desktop is `claude-ai`, Cursor is `cursor-vscode`), with the scopes it gets.

## Scopes

Every memory and conversation carries a scope label, for example `personal` or `work`. Each client reads some scopes and writes one, so you can keep an assistant away from part of your data. The default is that every client reads and writes `personal`, which shares everything:

```toml
# ~/.openmcp/config.toml
ignore_scopes = false        # true: every client reads and changes everything, whatever its scope

[defaults]
read_scopes = ["personal"]
write_scope = "personal"

[clients.cursor-vscode]      # a client, by the name `openmcp clients` shows
read_scopes = ["work"]
write_scope = "work"
```

## Capture conversations from disk

`openmcp setup vscode-copilot` and `openmcp setup claude-code` also turn on transcript capture for those apps (`[capture] sources` in the config). While capture is on, each search picks up new turns from their saved chats. To catch up on older chats, run `openmcp capture`; it reads chats changed in the last 30 days (`[capture] days`).

## Consolidate and review

Captured and imported chats hold most of what your assistants have learned, but only memories make up your profile. To turn chats into memories, ask any connected assistant to **"consolidate my chats"**, or use its OpenMCP `consolidate` prompt. It reads the conversations that haven't been consolidated yet, a batch at a time, and proposes memories, each citing the conversations it came from. The model that writes the proposals is your assistant's own; OpenMCP calls no model and needs no API key.

Nothing is saved until you approve it:

```sh
openmcp review     # opens a local page in your browser
```

The review page lets you:
- approve, edit or reject each proposal, and see the conversations it came from;
- for a proposal similar to an existing memory, approve it as a replacement;
- find near-duplicate memories and merge them into one, keeping history;
- search, edit and delete memories, and see each one's history.

A rejected proposal isn't proposed again. A conversation is read once, and comes back later with only its new messages. The page listens on 127.0.0.1 only, behind a token in the link `openmcp review` prints.

## Import and export

```sh
openmcp import ~/Downloads/chatgpt-export.zip   # ChatGPT or Claude data export, or an OpenMCP export
openmcp status                                  # import progress and what's stored
openmcp export                                  # everything, to openmcp-export-<date>.json (readable only by you)
```

Imports run newest conversations first, and everything is searchable by keyword within seconds. Meaning-based search catches up as the import continues. An import is safe to interrupt: run the same command again to resume. Importing a file twice adds nothing, and conversations you deleted stay deleted.

An export contains every memory, with its earlier versions, and every conversation, each with its scope. `openmcp import` reads it back with the same scopes, unless you pass `--scope`. To bring in history from another provider, convert its export to this format; the format is documented in `src/openmcp/ingest/openmcp_format.py`.

## Commands

| Command | What it does |
|---|---|
| `openmcp init` | Create `~/.openmcp`, the database and the config |
| `openmcp setup [client]` | Connect an MCP client and copy the standing instruction |
| `openmcp clients` | Clients that have connected, and their scopes |
| `openmcp memories list\|show\|rm` | Browse and delete memories |
| `openmcp conversations list\|show\|rm` | Browse and delete conversations |
| `openmcp review` | Approve proposed memories; browse, edit and delete memories |
| `openmcp import <file>` / `openmcp export` | Bring history in, take everything out |
| `openmcp capture` | Catch up on transcripts from capture-enabled apps |
| `openmcp embed` | Download the model and embed whatever isn't embedded yet |
| `openmcp status` | Import jobs, store counts, proposals waiting for review |
| `openmcp audit [--stats]` | Every tool call: which client, which app, what, how long |
| `openmcp serve` | The MCP server; your clients run this themselves |

## Safety model

- **Local only.** There's no server and no account. Your data is in `~/.openmcp`, and the only network access is the one-time model download.
- **Secrets.** Memories that look like credentials (API keys, tokens, private keys, card numbers, passwords) are rejected. In recorded and imported conversations, those parts are replaced with `[REDACTED]`. The filter uses patterns, so it will miss some secrets.
- **Stored data is data, not instructions.** Everything returned to a model is wrapped and labelled as your stored data, and content that looks like a prompt injection is flagged. This makes injection visible and harder. It doesn't make it impossible.
- **Every call is logged** with the client, the reporting app, the tool and the timing (`openmcp audit`).
- **Scopes are about assistants, not security.** They decide which assistant sees what. They don't protect the database file: any program running as you can read it. Client names come from what each app reports about itself, and nothing verifies that.
- **Deleting really deletes** (`forget`, `rm`): the item, its history, its search index entries and its vectors.

## Limitations

- **Recording depends on the model** for clients without transcript capture. In testing, assistants rarely called `record_exchange` without the pasted instruction.
- **Transcript capture reads undocumented formats.** It's verified on VS Code 1.99 and Claude Code 2.1, and a format change will need a reader update. A broken reader is reported and skipped.
- **Web-only assistants can't connect,** because they can't start a local server. That covers ChatGPT, claude.ai and Gemini on the web. Their history can come in through exports.
- **Meaning-based search uses a small model** that misses some paraphrases. "the fitness app throttling me" doesn't find a note about Strava rate limits.
- **The ChatGPT and Claude export parsers** are tested on synthetic exports. Both formats are undocumented, and a very large export is loaded into memory whole.
- **Clients are verified on macOS:** Claude Desktop, VS Code, Cursor and Claude Code. Linux and Windows are covered by CI only.

## Development

```sh
uv sync
uv run pytest                        # real-model tests download ~70MB once to ~/.cache/openmcp-test-models
uv run python scripts/benchmark.py   # performance budget: 10k memories, 100k chunks (SPEC.md 4.3)
```

Running `openmcp setup` from a development checkout hands the job to your installed copy, so your apps never run a work-in-progress branch. `--dev` overrides that. See [SPEC.md](SPEC.md) for the design, milestones and open questions.
