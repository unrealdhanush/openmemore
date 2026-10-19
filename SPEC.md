# OpenMCP — v1 Build Spec

> Working name. Rename before any public release ("OpenMCP" reads like the protocol itself and may already be taken).

## 0. Instructions for Claude Code

- Read this whole file before writing code.
- Build **one milestone at a time** (Section 11). Stop at the end of each milestone, summarize what was built, and wait for review.
- Every milestone ships with tests. No milestone is done until `pytest` passes.
- Keep dependencies to the list in Section 4. Ask before adding anything else.
- Do **not** build anything from "Later" (Section 12) unless asked.
- If this spec contradicts itself or something turns out to be impossible, stop and flag it instead of guessing.
- Never commit real personal chat data. Tests use synthetic fixtures only.

---

## 1. What this is

A **local-first MCP server** that gives every AI assistant a user works with (Claude, Cursor, Gemini CLI, and later ChatGPT) **one shared, user-owned context store** on the user's own machine.

Any connected model can **write** what it learns (preferences, projects, decisions), **record** each substantive exchange (what the user asked, what it answered), and **read** what other models stored. Ask Claude something today, ask a different assistant about it tomorrow, and it knows what you mean. Start a project in Claude, continue it in another assistant without re-explaining anything.

Single-assistant memory of preferences is already solved inside each product. The point of OpenMCP is what no single assistant can do: carry context **across** assistants.

Models do not "learn from each other." They read and write a shared store. No weights change.

### Problem
- Power users spread their context across several assistants, and each one holds a different partial picture.
- They keep re-explaining projects, preferences, and constraints.
- Every assistant is building *import into its own silo*. None makes *export* easy.

### Principles
1. **Local-first.** All data lives on the user's disk. No hosted backend.
2. **User-owned.** The user can list, edit, export, and delete everything.
3. **Stored data is data, not instructions.** Everything returned to a model is wrapped and labeled as untrusted context (Section 8).
4. **Transparent.** Every read and write is logged with the client that made it.
5. **Legitimate access only.** No scraping, no session cookies. Context comes in through MCP writes, official data exports, and (opt-in, per client) the transcripts that clients already save on the user's own disk.

### Non-goals (v1)
- No LLM-to-LLM calling. Multi-model "consult another model" servers already exist.
- No hosted service, no accounts, no sync.
- No web or mobile clients (they need a remote server; see Section 12).
- No fine-tuning or adapters.

---

## 2. Architecture

```
 Claude Desktop     Cursor      Gemini CLI      (each spawns its own stdio process)
       |              |             |
   openmcp serve  openmcp serve  openmcp serve   --client <name>
       \              |             /
        \             |            /
         +----- ~/.openmcp/openmcp.db  (SQLite, WAL mode) -----+
         |   memories | conversations | messages | chunks      |
         |   FTS5 indexes | sqlite-vec vectors | audit_log     |
         +------------------------------------------------------+
                              ^
                              |
                 openmcp import <export.zip>   (CLI, background job)
```

Key consequences:
- Each MCP client launches its **own** `openmcp serve` process over stdio, so **several processes may write to the DB at once**. Use WAL mode plus `busy_timeout`, and keep transactions short.
- **Bulk ingestion never runs inside `serve`.** Import is a CLI command with resumable jobs, so a large import can't stall a chat. The one exception is transcript capture (M1.5): before each read, `serve` stores what changed in enabled clients' transcripts, incrementally and within a 250ms budget (anything left over waits for the next read). The first backfill runs from the CLI.

---

## 3. Data directory

```
~/.openmcp/
  openmcp.db         # SQLite database
  config.toml        # clients + scopes
  imports/           # copies of imported export zips (optional, user can delete)
  logs/
```

Overridable with the `OPENMCP_HOME` environment variable (tests use a temp dir).

---

## 4. Tech stack

| Concern | Choice |
|---|---|
| Language | Python 3.11+ |
| Packaging | `uv`, `pyproject.toml`, console script `openmcp` |
| MCP | Official `mcp` Python SDK (FastMCP), stdio transport |
| Validation / schemas | Pydantic v2 |
| Storage | SQLite (stdlib `sqlite3`), WAL mode. **No ORM.** |
| Keyword search | SQLite FTS5 |
| Vector search | `sqlite-vec` |
| Embeddings | `fastembed` with `BAAI/bge-small-en-v1.5` (384-dim, local, ONNX) |
| CLI | `typer` + `rich` (progress bars) |
| Config | `tomllib` (read), `tomli-w` (write) |
| Tests | `pytest` |
| Manual testing | MCP Inspector (`npx @modelcontextprotocol/inspector`) |

Verify current SDK APIs against the installed versions rather than relying on memory.

### 4.1 Layering

```
MCP tools / CLI  ->  Pydantic models (models.py)  ->  repository (repo.py)  ->  sqlite3
```

- **Pydantic at every boundary.** Every MCP tool's arguments and results, every CLI input, and the normalized import schema are Pydantic models. Nothing unvalidated reaches the repository.
- **Repository layer (`repo.py`)** is the only module that writes SQL. It converts Pydantic models to rows and rows back to models. Tools, CLI, and ingestion call the repository, never `sqlite3` directly.
- **Plain `sqlite3`, no ORM.** One fewer abstraction to debug; the schema is small.

### 4.2 Design decisions (settled; don't relitigate in v1)

- **SQLite over files of serialized Pydantic objects.** Several server processes write at once (one per client); SQLite in WAL mode handles concurrent writers, transactions make imports atomic and resumable, and FTS5 plus sqlite-vec give search in the same single, user-owned file.
- **Flexibility through `metadata` JSON columns, not a schemaless store.** The core schema is the contract between every model that reads and writes. Provider-specific or experimental fields go in `metadata` (queryable with SQLite's JSON functions). A field that proves useful gets promoted to a real column through a migration.
- **Python, not Rust.** Per-call latency is dominated by the LLM (seconds); SQLite and ONNX Runtime are already native code. The one real Python cost, process startup, is handled by lazy-loading the embedding model. If profiling later shows a hot spot, rewrite only that piece (e.g., via PyO3), not the server.

### 4.3 Performance budget

- `openmcp serve` ready to accept calls in under 500ms (embedding model must **not** load at startup).
- `search_context`: p95 under 100ms once warm, on a store of 10k memories and 100k chunks.
- `save_memories` with 20 items: p95 under 300ms once warm.
- Every tool call records `duration_ms` in `audit_log`; `openmcp audit --stats` prints p50/p95 per tool so these budgets are checked against real numbers.
- `scripts/benchmark.py` checks the two p95 budgets on a synthetic store (Zipf-distributed vocabulary, random unit vectors; queries and saves use the real model). Measured 2026-10-18 on an M-series Mac: `search_context` p50 72ms / p95 76ms, `save_memories` (20) p50 208ms / p95 219ms; startup ~360ms. About 60ms of each search is the exact vector scan over 100k chunks, which grows linearly with the store.

---

## 5. Repo layout

```
openmcp/
  pyproject.toml
  README.md
  SPEC.md
  src/openmcp/
    __init__.py
    cli.py              # typer app: init, serve, setup, capture, import, status, memories, conversations, audit, export
    config.py           # load/write config.toml, client scope resolution
    models.py           # Pydantic models: tool args/results, normalized import schema
    repo.py             # repository: the ONLY module that writes SQL
    db.py               # connection, pragmas, migrations
    migrations/         # 001_init.sql, ...
    server.py           # FastMCP server, tool + prompt registration
    instructions.py     # server instructions + the standing instruction for each client
    setup.py            # `openmcp setup <client>`: register the server, install the standing instruction
    tools/
      memories.py       # save, update, forget
      conversations.py  # record_exchange, recent_conversations
      search.py         # hybrid search
      profile.py        # get_profile
    safety/
      secrets.py        # secret/PII filter on writes
      wrap.py           # untrusted-data wrapping on reads
    capture/
      __init__.py       # transcript capture runner (changed files only, time budget)
      claude_code.py    # Claude Code session JSONL parser
      vscode_copilot.py # VS Code Copilot Chat session parser
    embed.py            # fastembed wrapper, lazy-loaded; embeds pending rows
    ingest/
      jobs.py           # job table, resumable runner
      normalize.py      # common conversation/message schema
      chatgpt.py        # ChatGPT export parser
      claude.py         # Claude export parser
      detect.py         # auto-detect provider from zip contents
    audit.py
  scripts/
    benchmark.py        # performance budget check (Section 4.3)
  tests/
    fixtures/           # SYNTHETIC export samples only
    test_*.py
```

---

## 6. Data model

Logical schema. Implementation details (FTS5 external-content wiring, triggers, vec0 syntax) are up to the implementer.

```sql
-- Conversation history: captured live by clients (record_exchange) or imported from exports
CREATE TABLE conversations (
  id            INTEGER PRIMARY KEY,
  provider      TEXT NOT NULL,          -- captured: the client name ('claude-desktop', ...); imported: 'chatgpt' | 'claude' | ...
  external_id   TEXT NOT NULL,          -- provider's conversation id
  title         TEXT,
  scope         TEXT NOT NULL DEFAULT 'personal',
  metadata      TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(metadata)),  -- provider-specific extras
  created_at    TEXT,                   -- ISO 8601 UTC
  updated_at    TEXT,
  UNIQUE (provider, external_id)
);

CREATE TABLE messages (
  id               INTEGER PRIMARY KEY,
  conversation_id  INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role             TEXT NOT NULL CHECK (role IN ('user','assistant','system','tool')),
  content          TEXT NOT NULL,
  created_at       TEXT,
  content_hash     TEXT NOT NULL,        -- sha256(role, created_at, content): the same message is stored once
  external_id      TEXT,                 -- the transcript's id for it (captured transcripts): re-reads update in place
  UNIQUE (conversation_id, content_hash),
  UNIQUE (conversation_id, external_id)  -- where external_id is set
);

-- Messages split for search and embedding (~1000 chars; long messages -> several chunks)
CREATE TABLE chunks (
  id          INTEGER PRIMARY KEY,
  message_id  INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
  ord         INTEGER NOT NULL,
  text        TEXT NOT NULL
);

-- The core: distilled, durable context written by models or the user
CREATE TABLE memories (
  id             INTEGER PRIMARY KEY,
  kind           TEXT NOT NULL CHECK (kind IN ('fact','preference','project','decision','note')),
  subject        TEXT,                 -- short label, e.g. 'coding style', 'OpenMCP'
  content        TEXT NOT NULL,
  project        TEXT,                 -- optional project tag
  scope          TEXT NOT NULL DEFAULT 'personal',
  source_client  TEXT NOT NULL,        -- 'claude-desktop', 'cursor', 'cli', 'import:chatgpt'
  confidence     REAL,                 -- 0..1, optional, set by writer
  content_hash   TEXT NOT NULL,        -- normalized-text hash for exact dedup
  metadata       TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(metadata)),  -- experimental fields
  created_at     TEXT NOT NULL,
  updated_at     TEXT NOT NULL,
  superseded_by  INTEGER REFERENCES memories(id)   -- history kept on update
);

-- Full-text: FTS5 over memories(subject, content) and chunks(text); an fts5vocab table over the
--            chunk index gives each term's document count (search skips very common words)
-- Vectors:   sqlite-vec vec0 tables memories_vec / chunks_vec, float[384], cosine distance, keyed
--            by memories.id and chunks.id; removed with their rows (and on supersede) by triggers.
--            memories.embedded / chunks.embedded mark rows that have a vector (0 = pending)

CREATE TABLE audit_log (
  id            INTEGER PRIMARY KEY,
  ts            TEXT NOT NULL,
  client        TEXT NOT NULL,
  tool          TEXT NOT NULL,
  summary       TEXT,                  -- short, never the full payload
  result_count  INTEGER,
  duration_ms   REAL NOT NULL          -- for the performance budget (Section 4.3)
);

CREATE TABLE ingest_jobs (
  id           INTEGER PRIMARY KEY,
  provider     TEXT NOT NULL,
  file_path    TEXT NOT NULL,
  file_hash    TEXT NOT NULL UNIQUE,   -- re-importing the same zip is a no-op
  status       TEXT NOT NULL CHECK (status IN ('queued','running','done','failed')),
  total        INTEGER,
  processed    INTEGER NOT NULL DEFAULT 0,
  cursor       TEXT,                   -- resume point
  error        TEXT,
  started_at   TEXT,
  finished_at  TEXT
);

-- Transcript files already captured (M1.5): unchanged files are skipped
CREATE TABLE capture_files (
  path             TEXT PRIMARY KEY,
  source           TEXT NOT NULL,        -- 'claude-code' | 'vscode-copilot'
  mtime_ns         INTEGER NOT NULL,
  size             INTEGER NOT NULL,
  resume_offset    INTEGER NOT NULL DEFAULT 0,   -- append-only transcripts: re-read from the last turn's start
  conversation_id  INTEGER,
  ignored          INTEGER NOT NULL DEFAULT 0,   -- set when the user deletes the conversation: never re-captured
  captured_at      TEXT NOT NULL
);

CREATE TABLE schema_version (version INTEGER NOT NULL);
```

Rules:
- "Active" memories are those with `superseded_by IS NULL`.
- `update_memory` never overwrites. It inserts a new row and points the old one at it.
- `forget` is a **hard delete** of the row, its history chain, and its vectors. The user owns their data.
- `metadata` is for fields that vary by provider or are still experimental. Pydantic models expose it as `dict[str, Any]`. Never put fields there that tools filter on routinely; those get real columns.

---

## 7. MCP surface (v1)

Every tool call:
1. Resolves the calling client's scopes from `config.toml` (Section 9).
2. Writes one `audit_log` row.
3. Returns read results wrapped per Section 8.

Tool descriptions matter: models decide when to call tools based on them. Write them carefully and keep them specific.

The signatures below are shorthand. Implement each tool's arguments and result as Pydantic models in `models.py`, so validation errors come back to the calling model as clear messages instead of exceptions.

### 7.1 `save_memories`
Bulk write. Used for both everyday saves and the instant cold-start dump.

```
save_memories(
  items: list[{
    kind: 'fact'|'preference'|'project'|'decision'|'note',
    content: str,             # one self-contained statement
    subject?: str,
    project?: str,
    confidence?: float
  }],                         # max 100 per call
  scope?: str                 # defaults to the client's write scope
) -> { saved: [id], skipped_duplicates: int, possible_duplicates: [{new_index, existing_id, existing_content}], rejected: [{index, reason}] }
```

- Run every item through the secret filter (Section 8). Reject matches; never store them.
- Exact duplicate (same normalized hash, active, same scope) → skip.
- Near-duplicate (cosine ≥ 0.92, same kind, in a scope the client can read, including earlier items in the same call) → **save anyway** but report it in `possible_duplicates` (existing content wrapped per Section 8), so the calling model can follow up with `update_memory` or `forget`. No silent merging in v1.
- Description should tell the model to save when the user states a preference, makes a decision, or finishes meaningful work on a project, and to **never save secrets, credentials, or other people's private details**. One-off questions go to `record_exchange` instead.

### 7.2 `search_context`
```
search_context(
  query: str,
  kinds?: list[str],
  project?: str,
  include_chats?: bool = true,
  limit?: int = 10            # max 25
) -> { results: [{ type: 'memory'|'chat', id, content, kind?, subject?, project?, conversation_id?, title?, role?, source, created_at, score }] }
```
- Hybrid: FTS5 (BM25) + vector similarity, merged with reciprocal rank fusion (k = 60), plus a mild recency boost. Up to four ranked lists: keyword and vector, for memories and for chat chunks. Without the embedding model (not downloaded yet) the vector lists are simply missing.
- Vector search always returns the nearest items, related or not, so vector hits must have cosine similarity ≥ 0.55 and be within 0.08 of the best vector hit (calibrated for bge-small; Section 14).
- Recency boost: at most 0.0005 added to a fused score (a top rank is worth ≈ 0.016), halving every 30 days, counted in whole hours so items written close together still tie.
- **Recall turns** ("what did I ask Copilot?") are stored but marked (`messages.recall`) and ranked below the conversations they recap: their score is multiplied by 0.25, they don't set the bar for the vector similarity window, and they keep their own positions when later turns are moved ahead. Unmarked, each recap matched the next such question better than the original and pushed it further down (found in the M2 manual test). A turn is a recall when:
  - `record_exchange` follows, within 10 minutes and in the same server process, a `search_context` or `recent_conversations` call that returned something, and carries no `code` (`get_profile` doesn't count);
  - a captured transcript turn called `recent_conversations` or `search_context` (Claude Code `tool_use` names, VS Code `toolCallRounds`).
- The instructions also tell models not to record pure lookups, and to search for the topic itself ("cat names", not "the pet question I asked Copilot"), narrowing by assistant with `recent_conversations`.
- Chat keyword search leaves out the most common query words once the rest already match 8,000 chunks between them: ranking costs ~1.6ms per thousand matches, and BM25 gives such words almost no weight.
- Only searches scopes the client may read.
- Chat results return the matching chunk plus `conversation_id`, conversation title, and source, not whole conversations; `recent_conversations(conversation_id=…)` reads the rest.
- Chats have no kind or project, so filtering on either excludes them.
- Within one conversation, later messages rank ahead of earlier ones (a conversation often discusses options before settling on one). When chats are in the results, a `note` says that a later message supersedes an earlier one it contradicts.

### 7.3 `get_profile`
```
get_profile(project?: str) -> { preferences: [...], facts: [...], active_projects: [...], recent_decisions: [...] }
```
- **Deterministic in v1**: no LLM. Group active memories by kind; "active projects" means project memories updated in the last 30 days; cap each group (e.g., 15 items).
- This is the cheap call a model makes at the start of a session.

### 7.4 `update_memory`
```
update_memory(id: int, content: str, subject?: str) -> { new_id, previous_id }
```
Supersede, don't overwrite. Secret filter applies. Client must have read+write on the memory's scope.

### 7.5 `forget`
```
forget(id: int) -> { deleted: int }
```
Requires an explicit id (no "forget by query," which is too easy to trigger through injected text). Hard delete, logged.

### 7.6 MCP prompt: `import_my_memory`
The instant cold start. A prompt template the user triggers in any connected client. It tells the model:

> List everything you know or remember about me (preferences, facts, ongoing projects, past decisions) as separate self-contained statements, then save them with `save_memories`. Skip anything that is a secret, credential, or another person's private detail. Report what you saved.

This gives each assistant's existing memory a path into the store within minutes, while full exports index in the background.

### 7.7 MCP resource: `openmcp://profile`
Same payload as `get_profile()`, exposed as a resource for clients that prefer resources.

### 7.8 `record_exchange`
Live capture: how a question asked in one assistant becomes visible to the others.

```
record_exchange(
  question: str,              # what the user asked, verbatim if short, else condensed (max 8000)
  answer_summary: str,        # short summary of the answer: conclusions, recommendations, next steps (max 4000)
  conversation_id?: int,      # omit on the first call in a chat; then pass back the id it returned
  title?: str,                # for a new conversation; defaults to the question's first line
  code?: list[{ content: str, path?: str, language?: str }]   # code written in this answer, in full (max 10, 20000 chars each)
) -> { conversation_id, message_ids: [int], redacted: [reason] }
```
- Stores a `user` message and an `assistant` message in `conversations`/`messages`, chunked and indexed for search. `provider` is the calling client; `scope` is its write scope; `metadata.capture = 'record_exchange'`.
- A client may only append to conversations it started (same `provider`, writable scope).
- Not called for turns that only looked up earlier conversations; the server marks the ones that are recorded anyway as recall (Section 7.2).
- Code blocks are appended to the assistant message as fenced Markdown, so another assistant can continue the work (summaries alone lost the code in the M1 manual test).
- Secrets are **redacted** (replaced with `[REDACTED]`), not rejected, in the question, answer, code, and title; the reasons come back in `redacted` (Section 8).
- The description and the server instructions tell the model to call it after every answer, skipping only greetings and thanks.
- Capture depends on the model choosing to call the tool, which it doesn't do reliably without a standing instruction (Section 14). Clients whose transcripts are captured (M1.5) aren't offered this tool at all.

### 7.9 `recent_conversations`
```
recent_conversations(
  source?: str,               # assistant/provider name, partial, case-insensitive ('claude' matches 'claude-desktop')
  conversation_id?: int,      # just this conversation, with up to 50 messages in full (code included)
  limit?: int = 5             # max 20
) -> { conversations: [{ id, title, source, started_at, updated_at, message_count, messages: [{ role, content, created_at }] }] }
```
- Newest first by last activity; listings carry the latest 6 messages of each conversation, each cut to the usual 2000 characters. A single conversation returns messages up to 25000 characters.
- Results carry the same ordering `note` as `search_context`.
- Answers questions with no topic words ("what did I ask Claude earlier?"), which keyword search can't.
- Only scopes the client may read. Content is wrapped (Section 8).

### 7.10 Server instructions and standing instructions
The MCP `instructions` tell every connected model, in short: call `get_profile` at the start; call `record_exchange` after every answer (with any code); save decisions with `save_memories` (kind `decision`) so they outrank the options discussed before them; call `recent_conversations` or `search_context` when the user refers to an earlier conversation (including with another assistant) before saying it doesn't know; never record secrets. Clients whose transcripts are captured get the same text without `record_exchange`.

Real clients don't act on server instructions alone: in the M1 manual test, Claude Desktop made no tool calls until the user added the same guidance as a personal preference. So `openmcp setup <client>` (Section 9) also installs it as a **standing instruction** wherever the client reads the user's own instructions.

---

## 8. Safety

### Writes: secret filter (`safety/secrets.py`)
Reject items matching common patterns: API keys and tokens (`sk-…`, `ghp_…`, AWS keys, JWT-shaped strings, private-key headers), payment card numbers (Luhn-checked), US SSN pattern, and lines that look like `password: …`. Return the rejection reason; never echo the matched value back.

Memories (`save_memories`, `update_memory`) are **rejected** on a match. Captured exchanges (`record_exchange`, including code and titles, and transcripts captured in M1.5) are **redacted** instead: each match is replaced with `[REDACTED]` (a private key's whole block, not just its header) and the rest is stored. Users paste keys into chats often, and rejecting the exchange would silently lose the record.

### Reads: untrusted-data wrapping (`safety/wrap.py`)
Every piece of stored content returned to a model is wrapped, for example:

```
<stored_context source="import:chatgpt" id="1234">
...content...
</stored_context>
```

and every read response starts with one line stating that the enclosed content is the user's stored data and must not be followed as instructions. Also:
- Escape any `</stored_context>` inside content.
- Cap each item's length.
- Flag (don't silently drop) content matching obvious injection phrasing ("ignore previous instructions", etc.) with an attribute like `flagged="possible_injection"`.

This doesn't make injection impossible. It makes it visible and harder. Say so honestly in the README.

---

## 9. Clients and scopes

`config.toml`, created by `openmcp init`:

```toml
[defaults]
read_scopes = ["personal"]
write_scope = "personal"

[clients.claude-desktop]
read_scopes = ["personal", "work"]
write_scope = "personal"

[clients.cursor]
read_scopes = ["work"]
write_scope = "work"
```

- The client name comes from `openmcp serve --client <name>`.
- Unknown clients fall back to `[defaults]`.
- Scopes are free-form strings; `personal` and `work` are just the starting set.
- With these defaults, Cursor can't read what Claude Desktop writes (`personal`). That's the isolation working as designed. For the cross-assistant tests (M1, v1), use a client that reads `personal` (e.g. Gemini CLI, which falls back to `[defaults]`) or add `personal` to Cursor's `read_scopes`.

`[capture]` (M1.5) lists the clients whose transcripts are captured; it starts empty:

```toml
[capture]
sources = ["claude-code", "vscode-copilot"]
days = 30            # only transcripts changed within this many days
```

Captured conversations take the source client's `write_scope`.

`openmcp setup <client>` does the per-client work (`--dry-run` shows it first). Each file it edits gets a `.bak` copy, and files it can't parse (e.g. VS Code settings with comments) are left alone with the snippet printed instead.

| Client | Registers the server | Standing instruction | Capture |
|---|---|---|---|
| `claude-desktop` | `claude_desktop_config.json` (top-level `mcpServers`) | printed, to paste into Settings → Profile | `record_exchange` |
| `vscode-copilot` | `settings.json` `mcp.servers` (or `mcp.json` on VS Code 1.102+) | `github.copilot.chat.codeGeneration.instructions` | transcripts |
| `claude-code` | `claude mcp add --scope user` | marked block in `~/.claude/CLAUDE.md` | transcripts |
| others | printed snippet | printed | `record_exchange` |

Example Claude Desktop config (README should include one per supported client):

```json
{
  "mcpServers": {
    "openmcp": {
      "command": "uv",
      "args": ["--directory", "/path/to/openmcp", "run", "openmcp", "serve", "--client", "claude-desktop"]
    }
  }
}
```

---

## 10. Ingestion (cold start, background)

### CLI flow
```
openmcp import ~/Downloads/chatgpt-export.zip      # provider auto-detected
openmcp import ~/Downloads/claude-export.zip --scope personal
openmcp status                                     # job progress, counts
```

### Pipeline
1. **Detect** provider from the zip's contents (`detect.py`).
2. **Hash** the file. If a `done` job has this hash, exit early.
3. **Parse** with the provider adapter into the normalized schema (`normalize.py`).
4. **Order newest-first** by conversation `updated_at`. Recent history becomes searchable within minutes; older history fills in later.
5. **Batch** (e.g., 50 conversations per transaction). Insert conversations, messages, and chunks; embed chunks; update FTS. Commit, then advance `cursor` and `processed`.
6. **Resume**: if interrupted, re-running `openmcp import` on the same file continues from `cursor`.

Imports store **conversations and messages only**. They do not auto-create memories in v1; distilling memories from history is consolidation ("Later").

### Provider adapters
Write parsers defensively and test them against synthetic fixtures modeled on real exports. Formats change without notice, so verify against a real export (kept out of the repo) before relying on these notes:

- **ChatGPT export**: `conversations.json` holds a list of conversations. Each has a `mapping` of message nodes with parent/child links and a `current_node`. Walk from `current_node` up to the root to get the main branch (ignore abandoned branches). Message text lives in `content.parts`; roles in `author.role`; timestamps are epoch seconds.
- **Claude export**: `conversations.json` holds a list with `uuid`, `name`, timestamps, and `chat_messages` (each with `sender` of `human` or `assistant`, text, and `created_at`). Map `human` to `user`.

Skip empty messages, system scaffolding, and tool noise. Unknown fields are ignored, not fatal.

---

## 11. Milestones

### M0 — Skeleton
- `pyproject.toml`, `uv` setup, console script.
- `openmcp init` creates data dir, DB (migration 001), and default config.
- `openmcp serve --client X` starts a FastMCP stdio server with a `ping` tool.
- `models.py` and `repo.py` exist with the base pattern (one model, one repository function) so later milestones follow it.
- **Done when:** MCP Inspector connects and `ping` works; startup is under 500ms; tests cover init and migrations.

### M1 — Memory store
- `save_memories`, `update_memory`, `forget`, `get_profile`, `search_context` (**FTS5 only** for now, over memories and captured chats).
- Live capture: `record_exchange` and `recent_conversations`, chat chunks indexed with FTS5, server instructions (Section 7.10).
- Secret filter (reject for memories, redact for exchanges), read wrapping, audit log.
- Scope checks on every tool (read scopes for reads; write scope plus read for writes).
- `import_my_memory` prompt and `openmcp://profile` resource.
- CLI: `openmcp memories list|show|rm`, `openmcp conversations list|show|rm`, `openmcp audit` (with `--stats` for p50/p95 per tool).
- `openmcp setup <client>` with standing instructions (Sections 7.10, 9); `code` on `record_exchange`; later turns ranked first and the ordering note. (Added after the first manual test: models didn't call the tools unprompted, summaries lost code, and an assistant mixed an early suggestion with the later decision.)
- **Done when:** you ask one assistant (e.g. Claude Desktop) a substantive question, then ask a **different** assistant (e.g. Gemini CLI, or Cursor with `personal` readable) about that previous question, both with and without naming its topic ("what did I ask Claude earlier?"), and it answers from the stored exchange without being told. An automated test scripts the same flow across two real server processes.

### M1.5 — Transcript capture
Capture that doesn't depend on the model calling a tool, for clients that save transcripts on the user's disk.
- Parsers for Claude Code (session JSONL under `~/.claude/projects/`) and VS Code Copilot Chat (`chatSessions/*.json` under the VS Code user directory). User prompts and answer text only: tool calls, tool output, thinking, subagent turns, and command scaffolding are skipped. Both formats are undocumented, so parsers ignore what they don't recognize and are tested against synthetic fixtures modeled on real files.
- `capture_files` tracking: unchanged files are skipped; append-only transcripts are re-read from the start of their last turn; messages upsert by the transcript's own ids, so a growing answer updates in place. Deleting a captured conversation stops its transcript from being captured again.
- Opt-in per client through `[capture]`, which `openmcp setup` turns on. `serve` refreshes before `search_context` and `recent_conversations` within a 250ms budget; `openmcp capture` backfills.
- Captured clients get instructions without `record_exchange` and aren't offered the tool.
- **Done when:** you ask Copilot (VS Code) something without any tool call recording it, then ask Claude Desktop "what did I ask Copilot?" and about the topic by name, and it answers from the captured transcript. An automated test scripts the same flow with a real server process and a synthetic transcript.

### M2 — Semantic search
- `embed.py` with lazy model loading (don't slow server startup). Servers never download the model (too slow for a tool call): without it they use keyword search and leave new rows pending. `openmcp embed` downloads it (to `<data dir>/models`) and embeds everything pending; `setup` and `capture` run it too. `ping` reports whether semantic search is ready.
- `sqlite-vec` tables; embed on write: memories before their write transaction (the vector is needed for near-duplicate checks), chunks from `record_exchange` and transcript capture right after it, within 200ms, newest first.
- Hybrid search with reciprocal rank fusion and recency boost; near-duplicate detection in `save_memories`.
- **Done when:** a paraphrased query finds a memory that keyword search alone misses (test it), and a benchmark script on a synthetic 10k-memory / 100k-chunk store meets the Section 4.3 budget.

### M3 — Import
- Job table, ChatGPT and Claude adapters, auto-detect, newest-first batching, resume.
- `openmcp import`, `openmcp status` with a `rich` progress bar.
- Imported conversations are searchable and listable alongside captured ones (same tables, same tools).
- **Done when:** a synthetic export of 500 conversations imports, the newest 50 are searchable before the job finishes, and killing and re-running resumes without duplicates.

### M4 — Scopes and ownership
- Scope checks exist from M1; M4 audits that every tool and CLI path (including imports) honors them, and documents scope setup.
- `openmcp export` dumps all memories and conversations to JSON.
- README: install, per-client config snippets, the safety model, honest limitations.
- **Done when:** a `cursor` client scoped to `work` cannot see `personal` memories (tested), and the cross-client success test below passes.

### v1 success test
Start a project in Claude Desktop and work on it over a few exchanges. The next day, open a different client (Cursor or Gemini CLI) and continue the project **without re-explaining anything**, including picking up the questions you asked and the answers you got.

---

## 12. Later (do not build in v1)

- **Consolidation ("dream mode")**: scheduled job that distills imported chats into memories and merges near-duplicates, with the user reviewing changes.
- **Drift detection**: notice when recent behavior contradicts stored preferences, lower confidence, and ask instead of assuming.
- **Remote access**: opt-in tunnel plus strong auth so web and mobile clients (claude.ai connectors, ChatGPT) can reach the local server. Works only while the device is awake.
- **Review UI**: small local web page to browse, edit, and approve memories.
- **More importers**: Gemini (Google Takeout), others.
- **Vector compression**, when stores grow well past 100k chunks (the exact scan is ~60ms there). sqlite-vec stores only float32, int8, and bit vectors, so TurboQuant or PolarQuant as published would need a custom scan outside SQLite; their core idea (a fixed random rotation, so one calibrated quantizer fits every coordinate) can be applied to sqlite-vec's int8 storage. Its built-in `vec_quantize_int8(v, 'unit')` wastes most of the range on bge vectors (coordinates ≈ ±0.05) and kept only 82% of the true top 10 in a test.
- **A stronger embedding model**, if bge-small's misses matter in practice (Section 14).
- **More capture sources**: Cursor and others that keep local transcripts (verify each format on a real install first); client hooks where they exist, for capture without waiting for the next read.
- **Per-user evals**: auto-generated tests checking whether stored context actually improves answers.

---

## 13. Pre-flight before going public

- [ ] Get employer approval for publishing open source work in this area. Keep the repo private until then.
- [ ] Pick a final name and check package-registry and GitHub availability.
- [ ] Confirm no real personal data exists anywhere in the git history.
- [ ] Choose a license (MIT or Apache-2.0 are the usual picks).

## 14. Open questions

- Should `get_profile` be called automatically at session start, or left to the model? (Depends on client support for resources versus tools.)
- ~~What's the right near-duplicate threshold?~~ 0.92 with bge-small: rewordings score 0.97–0.995, different facts 0.73–0.84 ("allergic to peanuts" vs "shellfish": 0.84), but true paraphrases sit right on the line ("Allergic to peanuts." vs "Has a peanut allergy.": 0.920). Revisit on real data.
- Search similarity cutoffs (0.55 floor, 0.08 window) come from a 20-question synthetic set: 18/20 paraphrases found, ~1 unrelated item per query. bge-small misses some paraphrases entirely: "how was I planning to deal with the fitness app throttling me" ranks the Strava 429 memory 11th (similarity 0.48). A test records this as an expected failure, so a better model shows up as a test to update.
- Should imported chats ever be readable by default, or opt-in per scope?
- ~~How reliably do models call `record_exchange` unprompted?~~ Measured in the M1 manual test (2026-10-17): Claude Desktop and Copilot made **no** recording calls from tool descriptions and server instructions alone. With a standing instruction, Claude Desktop recorded every technical turn but skipped a casual one. Hence standing instructions via `openmcp setup` (M1) and transcript capture (M1.5). Open: how well standing instructions hold up over long chats, and in clients that show no transcript on disk (Claude Desktop, claude.ai).
- Recall detection is heuristic. A Claude Desktop server serves all its chats, so a lookup in one chat followed by a recorded answer in another within 10 minutes marks that answer as a recall too (ranked lower, not hidden). The verbatim question "what was the pet thing I asked Copilot about?" scores 0.543 against the original chat, just under the similarity floor, so only a topic search ("cat names": 0.81) finds it by meaning.
- Claude Code's `--resume` and VS Code's chat export may copy history into a new session file, which is captured as a separate conversation. Dedupe across sessions if it turns out to matter.
