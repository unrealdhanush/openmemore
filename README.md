# OpenMemore

One shared memory for all your AI assistants, kept on your own machine.

Ask Claude something today and ask Cursor about it tomorrow: it knows what you mean. OpenMemore is a local [MCP](https://modelcontextprotocol.io) server that every assistant you use connects to. Each one can save what it learns, record what you asked and what it answered, and read what the others stored. It's all in one SQLite file on your disk, which you can list, edit, export and delete.

**Status:** pre-release. The repository is private, and there's no published package yet.

## How it works

- **One store:** `~/.openmemore/openmemore.db`, a single SQLite file. Each assistant starts its own `openmemore serve` process and they all share the file.
- **What assistants can do:**
  - `get_profile`: what you've told your assistants before
  - `save_memories`: preferences, facts, decisions
  - `record_exchange`: what you asked and what it answered, including code
  - `recent_conversations` and `search_context`: find earlier conversations, with any assistant, by meaning as well as by words
  - `update_memory` and `forget`
  - `consolidation_batch` and `propose_memories`: turn conversations into proposed memories, for you to review
- **Capture:** an assistant records an exchange by calling `record_exchange`, which a short instruction you paste in asks it to do. For VS Code Copilot and Claude Code, OpenMemore can instead read the conversations they already save on disk, so nothing depends on the model.
- **Search:** keyword search (SQLite FTS5) combined with meaning-based search, using a small local embedding model (`BAAI/bge-small-en-v1.5`, about 70MB, downloaded once). Nothing leaves your machine.
- **History:** import your ChatGPT and Claude data exports, or any provider's history converted to OpenMemore's own format. Export everything back out in that same format.

## Install

Requires [uv](https://docs.astral.sh/uv/). uv provides Python 3.12, so you don't need your own; some system Pythons can't load SQLite extensions. OpenMemore runs on macOS, Linux and Windows; CI tests all three.

```sh
uv tool install --managed-python --python 3.12 "openmemore @ git+https://github.com/unrealdhanush/openmemore@main"
openmemore init      # creates ~/.openmemore (set OPENMEMORE_HOME to put it elsewhere)
openmemore embed     # downloads the embedding model once, so search works by meaning
```

Upgrade with `uv tool upgrade openmemore`, then restart your assistants: each one loads OpenMemore when it starts.

## Connect your assistants

```sh
openmemore setup cursor          # or vscode-copilot, claude-code, or any other client's name
openmemore setup --dry-run       # see what it would do first
```

Any MCP client works, with no client-specific support needed. The server is the same for every client:

```json
{ "mcpServers": { "openmemore": { "command": "/path/to/openmemore", "args": ["serve"] } } }
```

`openmemore setup` never edits another app's files. Where the app has its own way to add an MCP server, setup uses it:

| Client | How setup adds the server |
|---|---|
| VS Code (Copilot) | `code --add-mcp`. Needs `code` on your PATH (in VS Code, run "Shell Command: Install 'code' command in PATH") |
| Cursor | Cursor's install link, which opens its own review prompt |
| Claude Code | `claude mcp add` |
| Claude Desktop, and everything else | Prints the snippet above. Paste it into the client's MCP config (Claude Desktop: Settings → Developer → Edit Config), or run `openmemore setup <client> --config <that file>` to merge it in |

**The standing instruction.** Assistants don't reliably use tools just because they're available, so setup copies a short instruction to your clipboard. Paste it wherever the client reads your own instructions, for example Claude Desktop's Settings → Profile, or Cursor's User Rules. It points to the server's own instructions, which update with OpenMemore, so you only paste it once.

Restart the client after setup. `openmemore clients` then lists each client that has connected, under the name its app reports (Claude Desktop is `claude-ai`, Cursor is `cursor-vscode`), with the scopes it gets.

## Scopes

Every memory and conversation carries a scope label, for example `personal` or `work`. Each client reads some scopes and writes one, so you can keep an assistant away from part of your data. The default is that every client reads and writes `personal`, which shares everything:

```toml
# ~/.openmemore/config.toml
ignore_scopes = false        # true: every client reads and changes everything, whatever its scope

[defaults]
read_scopes = ["personal"]
write_scope = "personal"

[clients.cursor-vscode]      # a client, by the name `openmemore clients` shows
read_scopes = ["work"]
write_scope = "work"
```

A client can also be kept from storing much. With `writes = "memories"` under its `[clients.<name>]` table, it reads as usual and saves memories, but none of its conversations are kept: no `record_exchange`, no transcript capture. `writes = "none"` makes it read only. `openmemore setup claude-code --writes memories` sets this up for Claude Code.

## Capture conversations from disk

`openmemore setup vscode-copilot` and `openmemore setup claude-code` also turn on transcript capture for those apps (`[capture] sources` in the config). While capture is on, each search picks up new turns from their saved chats. To catch up on older chats, run `openmemore capture`; it reads chats changed in the last 30 days (`[capture] days`).

To capture only some projects, list folders under `[capture]`. This works for apps whose transcripts record the folder a chat was in, which Claude Code and VS Code both do. Subfolders count, and exclusions win:

```toml
[capture]
include_folders = ["~/projects"]          # only chats from these folders
exclude_folders = ["~/projects/client"]   # never chats from these
```

## Capture in the background

```sh
openmemore daemon install    # asks first; then starts at login (launchd, systemd, or a Windows logon task)
openmemore daemon status
```

Nothing runs in the background without your permission. `openmemore daemon install` first asks in a dialog (Allow / Don't Allow), or in the terminal when there's no desktop, such as over SSH. `openmemore setup` offers it when there's something to capture, and scripts can pass `--yes`. On macOS it then appears under System Settings → General → Login Items, where you can also switch it off.

The daemon reads transcripts from capture-enabled apps every few seconds and embeds what's new. Another assistant can then find a conversation moments after it happens, without waiting for the next lookup.

It's also an OpenTelemetry endpoint. Any app or library that records its conversations with OpenTelemetry's standard GenAI conventions can send them there; there's no app-specific code. That includes agents built with LangChain, LlamaIndex, or the OpenAI and Anthropic SDKs plus OpenTelemetry instrumentation. `openmemore daemon telemetry` prints the standard settings to give such an app (`OTEL_EXPORTER_OTLP_ENDPOINT` and a token header).
- **Formats:** logs and traces, in JSON or protobuf.
- **Which app:** it's filed under the app's `service.name`, which you can give scopes like any client.
- **What's kept:** only the text of what you and the assistant said. System prompts, tool output and the telemetry's other attributes are dropped.
- **Access:** the endpoint listens on 127.0.0.1 only, and only accepts requests carrying the token from your config.

Most desktop assistants don't export their conversations this way. For them, transcript capture and `record_exchange` remain the way in.

`openmemore daemon uninstall` stops it and removes the service.

## Memory over time

Memories keep two timelines: when something was true in your life, and when OpenMemore learned it. That idea comes from [LongMemory](https://github.com/CaviraOSS/LongMemory)'s design.
- **Corrections:** a correction doesn't overwrite. The old memory becomes history, true until the change, and current answers only use what's true now.
- **Questions about the past:** "what did I use before?" or "what did I prefer last year?" are answered with `search_context(mode="history")` or `as_of="2025-03"`. Every result says whether it's current, superseded, ended or contradicted.
- **Conflicts:** when two memories disagree and it isn't clear which is right, you settle it on the review page. Until then, neither is treated as current.
- **Confirmation:** something stated again by another assistant, or found again in a new conversation, gains confidence instead of being duplicated.
- **Ranking:** results favor what's recent and what you actually use (ACT-R-style activation). `explain=true` shows why each result was chosen, and `budget_tokens` keeps results within a size.

## Consolidate and review

Captured and imported chats hold most of what your assistants have learned, but only memories make up your profile. To turn chats into memories, ask any connected assistant to **"consolidate my chats"**, or use its OpenMemore `consolidate` prompt. It reads the conversations that haven't been consolidated yet, a batch at a time, and proposes memories, each citing the conversations it came from. The model that writes the proposals is your assistant's own; OpenMemore calls no model and needs no API key.

Nothing is saved until you approve it:

```sh
openmemore review     # opens a local page in your browser
```

The review page lets you:
- approve, edit or reject each proposal, and see the conversations it came from;
- for a proposal similar to an existing memory, approve it as a replacement;
- find near-duplicate memories and merge them into one, keeping history;
- search, edit and delete memories, and see each one's history.

**Automatic consolidation (optional, off by default).** OpenMemore can also consolidate on its own each night, with a model you choose. It calls any endpoint that speaks the open LLM API standard, Open Responses, such as OpenAI, Ollama, LM Studio, vLLM or OpenRouter. Endpoints that only speak OpenAI-style Chat Completions work too:

```sh
openmemore consolidate setup --endpoint http://127.0.0.1:11434/v1 --model qwen2.5:7b             # a local server
openmemore consolidate setup --endpoint https://api.openai.com/v1 --model <model> --key   # hosted: asks first
openmemore consolidate run      # now, instead of at 03:00
openmemore consolidate off
```

What OpenMemore sends and expects is in [docs/reference/consolidation-endpoint.mdx](docs/reference/consolidation-endpoint.mdx). A local endpoint never sends anything off your machine. A hosted one is used only after you allow that host in a dialog, and its key is kept in a file only you can read. The proposals land in the same review queue: the model also proposes corrections when a chat contradicts a memory, and nothing is saved until you approve it.

A rejected proposal isn't proposed again. A conversation is read once, and comes back later with only its new messages. The page listens on 127.0.0.1 only, behind a token in the link `openmemore review` prints.

## Use it from your own programs (experimental)

Agents and scripts you write can share the same memory, in-process, with no server:

```python
from openmemore import Memory

memory = Memory("trip-planner")       # a client name, scoped like any app

@memory.recall(query="request")       # related memories passed in as `context`
@memory.remember(question="request")  # each call stored as an exchange your assistants can find
def plan(request: str, context: str = "") -> str: ...
```

See [docs/guides/library.mdx](docs/guides/library.mdx).

## Import and export

```sh
openmemore import ~/Downloads/chatgpt-export.zip   # ChatGPT or Claude data export, or an OpenMemore export
openmemore status                                  # import progress and what's stored
openmemore export                                  # everything, to openmemore-export-<date>.json (readable only by you)
```

Imports run newest conversations first, and everything is searchable by keyword within seconds. Meaning-based search catches up as the import continues. An import is safe to interrupt: run the same command again to resume. Importing a file twice adds nothing, and conversations you deleted stay deleted.

An export contains every memory, with its earlier versions, and every conversation, each with its scope. `openmemore import` reads it back with the same scopes, unless you pass `--scope`. To bring in history from another provider, convert its export to this format; the format is documented in `src/openmemore/ingest/openmemore_format.py`.

## Commands

| Command | What it does |
|---|---|
| `openmemore init` | Create `~/.openmemore`, the database and the config |
| `openmemore setup [client]` | Connect an MCP client and copy the standing instruction |
| `openmemore clients` | Clients that have connected, and their scopes |
| `openmemore memories list\|show\|rm` | Browse and delete memories |
| `openmemore conversations list\|show\|rm` | Browse and delete conversations |
| `openmemore wipe --before DATE` / `--all` | Delete everything not touched since a date, or start fresh |
| `openmemore review` | Approve proposed memories; browse, edit and delete memories |
| `openmemore consolidate setup\|run\|status\|off` | Optional nightly consolidation with a model you choose |
| `openmemore import <file>` / `openmemore export` | Bring history in, take everything out |
| `openmemore capture` | Catch up on transcripts from capture-enabled apps |
| `openmemore daemon install\|status\|uninstall` | Capture in the background; `daemon telemetry` prints the OpenTelemetry settings |
| `openmemore embed` | Download the model and embed whatever isn't embedded yet |
| `openmemore status` | Import jobs, store counts, proposals waiting for review |
| `openmemore audit [--stats]` | Every tool call: which client, which app, what, how long |
| `openmemore serve` | The MCP server; your clients run this themselves |

## Safety model

- **Local only.** There's no server and no account. Your data is in `~/.openmemore`, and the only network access is the one-time model download.
- **Secrets.** Memories that look like credentials (API keys, tokens, private keys, card numbers, passwords) are rejected. In recorded and imported conversations, those parts are replaced with `[REDACTED]`. The filter uses patterns, so it will miss some secrets.
- **Stored data is data, not instructions.** Everything returned to a model is wrapped and labelled as your stored data, and content that looks like a prompt injection is flagged. This makes injection visible and harder. It doesn't make it impossible.
- **Every call is logged** with the client, the reporting app, the tool and the timing (`openmemore audit`).
- **Scopes are about assistants, not security.** They decide which assistant sees what. They don't protect the database file: any program running as you can read it. Client names come from what each app reports about itself, and nothing verifies that.
- **Deleting really deletes** (`forget`, `rm`, `wipe`): the item, its history, its search index entries and its vectors. `wipe` asks for your login password first, like sudo (not on Windows), and also erases the deleted text from the database file. It keeps a copy from just before, which it tells you to delete if you want the data gone for good.

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
uv run pytest                        # real-model tests download ~70MB once to ~/.cache/openmemore-test-models
uv run python scripts/benchmark.py   # performance budget: search p95 < 100ms, saving 20 memories < 300ms, at 10k memories and 100k chunks
uv run python -m bench locomo           # retrieval quality on LongMemEval / LoCoMo (results in bench/RESULTS.md)
```

The documentation is a [Mintlify](https://mintlify.com) site in `docs/` (`docs.json` and MDX pages). To preview it, run `npx mint dev` in `docs/`, which needs Node.js. `tests/test_docs.py` fails when a CLI command or option, a config key, or an MCP tool, prompt or resource has no reference entry, and when a page is missing from the navigation, links to a page that doesn't exist, or isn't valid MDX.

Running `openmemore setup` from a development checkout hands the job to your installed copy, so your apps never run a work-in-progress branch. `--dev` overrides that.
