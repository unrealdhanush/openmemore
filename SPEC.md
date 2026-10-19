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
   openmcp serve  openmcp serve  openmcp serve   (each names its client from clientInfo)
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
  imports/           # reserved; exports are read where they are, not copied
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
    clients.py          # naming clients from what they report (M3.5); no table of clients
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
      __init__.py       # store_conversation: shared by imports and transcript capture
      jobs.py           # job table, resumable runner (store, then embed newest first)
      chatgpt.py        # ChatGPT export parser
      claude.py         # Claude export parser
      detect.py         # open a zip/json/folder, recognize the format from its contents
      registry.py       # the importer registry (M3.5): built-in importers plus register()
      openmcp_format.py # OpenMCP's own format (openmcp.export): what export writes, and a way in for any provider
    audit.py
    export.py           # `openmcp export`: everything, in OpenMCP's own format
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
  duration_ms   REAL NOT NULL,         -- for the performance budget (Section 4.3)
  app           TEXT                   -- the application on the other end (MCP clientInfo.name), when it says
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

-- Conversations the user deleted, by source id, so a later export or capture doesn't bring them back
CREATE TABLE deleted_conversations (
  provider     TEXT NOT NULL,
  external_id  TEXT NOT NULL,
  deleted_at   TEXT NOT NULL,
  PRIMARY KEY (provider, external_id)
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
- **Recall turns** ("what did I ask Copilot?") are stored but marked (`messages.recall`) and ranked below the conversations they recap: their score is multiplied by 0.25, they don't set the bar for the vector similarity window, and they keep their own positions when later turns are moved ahead. Unmarked, each recap matched the next such question better than the original and pushed it further down (found in the M2 manual test). A captured transcript turn is a recall when it called `recent_conversations` or `search_context` (Claude Code `tool_use` names, VS Code `toolCallRounds`): the transcript shows what the turn did.
- `record_exchange` calls aren't marked. A timing rule (marking an exchange recorded soon after a lookup) failed in the follow-up manual test: Claude Desktop runs two server processes and spreads calls across them, and when it did fire, it marked new content (the assistant's own opinion, given right after reading another chat) as a recap. Instead, the instructions tell models not to record pure lookups (Claude Desktop then recorded neither of two lookups), and to search for the topic itself ("cat names", not "the pet question I asked Copilot"), narrowing by assistant with `recent_conversations`.
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
- Not called for turns that only looked up earlier conversations (Section 7.2).
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

Where setup can write that place (Copilot's settings, `~/.claude/CLAUDE.md`), it installs the full guidance and refreshes it on every run. Where the user has to paste it (Claude Desktop keeps personal preferences in the user's account, not on disk; Cursor's User Rules), it is a short **pointer** instead: use the openmcp tools as the server's instructions describe, naming only the calls that matter most (`get_profile` at the start, `record_exchange` after every answer, `save_memories` for decisions and preferences, recall tools before saying "I don't know"). Everything that may change lives in the server instructions, which ship with each upgrade, so the pasted text doesn't go stale. Setup copies it to the clipboard (`pbcopy`, `clip`, `wl-copy`/`xclip`/`xsel`); without a clipboard tool it prints it. Manual test (2026-10-18, Claude Desktop): with only the pointer, it reused conversation ids and searched by topic, details found only in the server instructions, but didn't save a stated decision until `save_memories` was named in the pointer.

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

**A client is named by what its application reports when it connects** (MCP `clientInfo`), made config-friendly: `claude-ai` (Claude Desktop), `visual-studio-code`, `cursor-vscode`, `Some Editor 2` → `some-editor-2`. There's no client table, no `--client` option, and no renaming: `openmcp clients` lists the names of the applications that have connected, with their scopes. (Earlier versions had `--client` and config `app` pins; M3.5 made them unnecessary and they were removed. A config that still has an `app` key fails to load, naming the key.)

`config.toml`, created by `openmcp init`, starts with only `[defaults]` and a commented example:

```toml
ignore_scopes = false

[defaults]
read_scopes = ["personal"]
write_scope = "personal"

# [clients.cursor-vscode]
# read_scopes = ["work", "personal"]
# write_scope = "work"
```

- Clients without a `[clients.<name>]` table use `[defaults]`.
- Scopes are labels on every memory and conversation; free-form strings, `personal` and `work` are just the starting set. They let an assistant be kept away from part of the user's data.
- `ignore_scopes = true` (top level; off by default) turns that off: every client reads every scope and can change items in any scope. New items still get the client's `write_scope` as their label, so separation can be turned back on later. `ping` reports `read_scopes: null`, and `openmcp clients` shows "all".
- A client configured to read only `work` can't read what others write to `personal`. That's the isolation working as designed; the cross-assistant tests (M1, v1) need clients that share a scope.

`[capture]` (M1.5) lists the transcript readers in use; it starts empty:

```toml
[capture]
sources = ["claude-code", "vscode-copilot"]
days = 30            # only transcripts changed within this many days
```

Captured conversations are stored under the reader's name (`vscode-copilot`, so "what did I ask Copilot?" finds them), in the write scope of the application the reader belongs to (`visual-studio-code`), the same client its live calls come from. Each reader declares that application's `clientInfo` name, which is also how a captured client's `record_exchange` calls are recognized and skipped instead of stored twice: the server instructions go out before a client says who it is, so every client gets the recording variant.

`openmcp setup [client]` connects a client (M3.5, `--dry-run` shows it first). It never edits another app's files: it uses the vendor's own mechanism where there is one, and otherwise prints the snippet that works in every MCP client (`--config <path>` merges it into a JSON file you name, keeping a `.bak`). The standing instruction is the pointer, put on the clipboard.

| Client | How setup adds the server | Capture |
|---|---|---|
| `vscode-copilot` | `code --add-mcp <json>` (VS Code's CLI; checked on 1.99.3) | transcripts |
| `cursor` | Cursor's install link, then its own review prompt | `record_exchange` |
| `claude-code` | `claude mcp add --scope user` | transcripts |
| anything else, Claude Desktop included | the generic snippet, printed (or merged with `--config`) | `record_exchange` |

The generic snippet (the same for every client):

```json
{
  "mcpServers": {
    "openmcp": {"command": "/path/to/openmcp", "args": ["serve"]}
  }
}
```

- **Run from a development checkout** (an editable install, e.g. `uv run openmcp setup` inside the repo), setup hands the whole job to an installed copy (a `uv tool install` first, then `openmcp` on PATH outside the checkout): registering the checkout would make the app run whatever branch is checked out, and the checkout's code would migrate the real database. Without an installed copy it stops and says how to install one; `--dev` registers the checkout deliberately.
- **VS Code** can also start MCP servers configured for other apps (`chat.mcp.discovery.enabled`). That once ran Claude Desktop's entry, with `--client claude-desktop`, as Copilot. With clients named by what they report, a server started that way is still named after VS Code, so discovery is harmless.

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
3. **Parse** with the provider adapter into the normalized schema (`ParsedConversation`, shared with transcript capture), one conversation at a time.
4. **Order newest-first** by conversation `updated_at`. Recent history becomes searchable first.
5. **Store**, 50 conversations per transaction: conversations, messages, and chunks (indexed for keyword search), through the same store as capture (secrets redacted, messages upserted by the provider's ids). The job's `cursor` and `processed` advance in the same transaction. Storing runs at ~10k messages a second, so a whole export is keyword-searchable within seconds.
6. **Embed**, newest conversations first, after everything is stored. With bge-small this runs at ~70 messages a second on a laptop (a 1,000-conversation, 10,000-message export: ~1s to store, ~2 min to embed). Embedding inside each batch instead held the oldest conversations back from keyword search for as long as embedding everything took. If this phase is interrupted, `openmcp embed` finishes it.
7. **Resume**: if interrupted while storing, re-running `openmcp import` on the same file continues from `cursor`. A newer export of the same account adds only what's new. Conversations the user deleted are not imported again (`deleted_conversations`).

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
- `openmcp serve` starts a FastMCP stdio server with a `ping` tool (M0 had `--client X`; M3.5 removed it).
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
- Parsers: ChatGPT (follows `current_node` to the root, so abandoned branches are left out; skips system and tool messages, hidden scaffolding, tool calls, custom instructions, reasoning, and images) and Claude (text blocks; tool use and results skipped). Unknown shapes are skipped, and a conversation that fails to parse is reported, not fatal.
- **Done when:** a synthetic export of 500 conversations imports, the newest 50 are searchable before the job finishes, and killing and re-running resumes without duplicates. Both are tested: after the first batch, a second connection finds the newest conversation and not the oldest; an import killed after 150 conversations resumes at 150 and ends with exactly 500 conversations and 2,000 messages. Still to do by hand: check the parsers against a real export of each provider (kept out of the repo), since the formats are undocumented.

### M3.5 — Any MCP client
**Goal: OpenMCP works with every MCP client and every assistant provider, not a fixed list.**

**Principle: the core needs no per-client knowledge.** A first version kept a registry of clients (config file locations, entry shapes, instruction locations, `clientInfo` names). Every fact in it belonged to another vendor and could change without notice; Cursor showed it, reporting `cursor-vscode` where its code suggested `Cursor`. So anything client-specific is optional, isolated so that when it breaks nothing else does, and delegated to interfaces the vendor maintains:
- **Naming:** a client is named by what its application reports when the session starts (MCP `clientInfo`): `Some Editor 2` becomes `some-editor-2`, and `[clients.some-editor-2]` gives it scopes. No table of client names. (`--client` and config `app` pins first stayed as overrides, then were removed: nothing needed them.)
- **Connecting:** one snippet, identical for every client: `openmcp serve`. Setup never edits another app's files. Where a vendor provides a way to add an MCP server, setup uses it, detected when it runs rather than assumed from a version table: VS Code's `code --add-mcp <json>`, Cursor's install link (`cursor://anysphere.cursor-deeplink/mcp/install?name=…&config=<base64>`, which opens Cursor's own review prompt), and `claude mcp add`. Everywhere else, Claude Desktop included, it prints the snippet; `--config <path>` merges it into a JSON file the user names. (Cursor's CLI also accepts `--add-mcp`, a leftover from VS Code that writes a settings file Cursor's agent doesn't read; checked, and not used.)
- **Instructions:** the pointer instruction, on the clipboard, pasted wherever the client reads the user's own instructions. A variant for captured clients says not to call `record_exchange`.
- **Capture:** transcript readers are the only client-specific code that's hard to avoid: there's no standard for transcripts. They're optional (`record_exchange` covers every client) and isolated (a reader that fails is reported and skipped). Each declares the application it reads, as that application reports itself, so a captured client isn't recorded twice: its `record_exchange` calls are skipped.
- **`openmcp clients`** lists the applications that have actually connected (from the audit log), with the scopes each gets.
- **History from any provider:** importers are a registry (`openmcp.ingest.registry`) with `register()`; OpenMCP's own conversation format is the third importer, a provider-neutral way in through a converter.
- Web-only assistants (ChatGPT, claude.ai, Gemini on the web) can't start a local server: they're reachable through exports (M3) or remote access (Section 12). Documented, not hidden.
- **Done when:** a client with no special support (a test client with an arbitrary `clientInfo` name) connects with the generic snippet, gets the scopes configured for its name, and records and recalls; the vendor install paths are each checked against a real install; a provider's importer can be added through the registry alone; existing setups keep working.
- **Status (2026-10-18):** all built and tested. Real installs: Cursor 3.25.9 (connected from a config setup wrote, reported `cursor-vscode`, followed the pointer: `get_profile`, `record_exchange` reusing the conversation, topic search, lookups not recorded); VS Code 1.99.3's `--add-mcp` (writes its own settings and replaces an earlier entry, checked against a throwaway profile); Cursor's install link (used for real: Cursor's prompt wrote `openmcp serve` with no `--client` into its own `~/.cursor/mcp.json`; its server then reported `cursor-vscode`, was named `cursor` through the config pin, and answered from a `personal` memory); `claude mcp add` (its help output; the old registry-era run used the same command). Open: one-click bundles (e.g. for Claude Desktop); plug-in transcript readers in `[capture] sources`.

### M4 — Scopes and ownership
- Scope checks exist from M1; M4 audits that every tool and CLI path (including imports) honors them, and documents scope setup.
- `openmcp export` dumps all memories and conversations to JSON.
- README: install, client setup (the generic snippet, and the vendor installers setup uses), the safety model, honest limitations.
- **Done when:** a client configured to read only `work` cannot see `personal` memories (tested), and the cross-client success test below passes.
- **Status (2026-10-19):**
  - **Scope audit:** `tests/test_scopes.py` has one app write `personal` data through every write path (memories, recorded chats, vectors). A work-only client then tries every read and write path. It sees and changes none of it: search by keyword and by meaning, profile and its resource, recent conversations, update, forget, appending to a chat, writing into `personal`, near-duplicate reports. Errors for out-of-scope ids are identical to errors for ids that don't exist. Imports take an explicit scope (or an export's own), and capture uses the scope of the app it belongs to. The CLI acts as the owner and sees every scope.
  - **`openmcp export`:** it writes the `openmcp.export` format (memories with their history, conversations, scopes, recall flags; the file is readable only by the user). `openmcp import` reads it back with scopes kept, unless `--scope` overrides. A round trip into an empty store restores everything current; importing into the same store adds nothing; imported memories go through the secret filter.
  - **README:** rewritten.
  - **v1 success test: passed (2026-10-19)**, below. **M4 and v1 are done.**

### v1 success test
Start a project in Claude Desktop and work on it over a few exchanges. The next day, open a different client (Cursor or Gemini CLI) and continue the project **without re-explaining anything**, including picking up the questions you asked and the answers you got.

**Result (2026-10-19), passed.** Two exchanges in Claude Desktop planned a project: three ideas, then architecture diagrams for two of them. In Cursor, a new chat referred to the project by one of its details rather than its name. Without being told where to look, Cursor called `get_profile`, `recent_conversations` and `search_context`, then opened Claude's conversation in full. It answered with both ideas, the trade-off between them and the agreed next step, and recorded its own exchange for Claude to see. Both chats happened the same day rather than a day apart; the cross-client path is the same. One gap was found and fixed: `get_profile` had matched project names exactly, and the model's guess returned an empty profile (it now matches loosely, and lists the projects there are when nothing matches).


### M5: Consolidation and review (after v1)
Captured and imported chats hold most of what the assistants have learned, but only memories make up the profile. Consolidation turns chats into proposed memories and finds near-duplicates. A local review page lets the user approve, edit or reject each proposal before it becomes a memory, and browse and edit memories in general.

**Where the model comes from:** the user's own assistants, through MCP. OpenMCP has no model of its own, no API key and no account, and that stays true: the client the user is talking to reads the chats and writes the proposals. Consequences:
- Consolidation runs when the user asks for it, not on a schedule.
- The proposals are only as good as that client's model.
- Nothing becomes a memory without the user's approval.

The proposal queue doesn't depend on what wrote a proposal, so a local-model or API-key generator can be added later without changing the review side (Section 12).

**Data (migration 010):**
- `proposals`: a proposed memory (`kind`, `content`, `subject`, `project`) with its `scope`, `reason`, `source_client`, `status` (`pending`, `approved`, `rejected`) and `content_hash`. Approving one records the memory it became.
  - **Sources:** the conversations a proposal came from (`proposal_conversations`), and, for a merge, the memories it replaces (`proposal_memories`).
  - **Deleting really deletes:** deleting a conversation deletes the proposals that came from it, unless they were already approved. Deleting a memory deletes the merge proposals that would replace it.
  - **Rejected stays rejected:** a proposal the same as one already rejected in that scope isn't proposed again.
- `conversations.consolidated_through`: the last message id read for consolidation. A conversation is due for consolidation while it has messages after that id, so a chat that grows comes back with only its new messages.

**MCP surface:**
- **`consolidation_batch(limit)`:** the conversations due for consolidation that this client can read, newest first, with their unread messages. Recap turns (`recall`) are left out, and each batch is capped in size; a long conversation continues in the next batch. Each conversation comes with `through_message_id`. Content is wrapped like every read (Section 8).
- **`propose_memories(items, done)`:** each item is a memory plus the `conversation_ids` it came from. `done` lists `{conversation_id, through_message_id}` pairs copied from the batch, which marks those conversations consolidated up to that message, whether or not they gave any proposals.
  - **Scope:** a proposal takes its source conversations' scope, which the caller must be able to read; sources in different scopes are an error. With no sources, it takes the caller's write scope.
  - **Updates:** an item can name the memory it updates (`replaces`), for example a project that has moved on. It must be a current memory the caller can read, in the same scope as the item's sources. On the review page, approving such a proposal makes it a new version of that memory, unless the user chooses to keep both.
  - **Checks:** the secret filter applies. A proposal is skipped when it duplicates an active memory, or a pending or rejected proposal, in its scope. A near-duplicate of an active memory is kept, and noted with that memory's id, so the review page can offer "replace it".
- **Prompt `consolidate`:** asks the model to work through `consolidation_batch` until it's empty, or for a set number of batches, and then tells the user to run `openmcp review`. Saying "consolidate my chats" works too, because the tool descriptions say when to use the tools.

**Near-duplicates, without a model:** on `openmcp review`, and from a button on the page, active memories of the same kind and scope with similarity at or above the near-duplicate threshold (0.92, Section 7.1) are grouped. Each group becomes a `merge` proposal whose text is the newest memory's, which the user can edit. Approving it adds the merged memory and marks every memory in the group as superseded by it, so history and `forget` cover the whole group.

**`openmcp review`:**
- **Server:** a local web page on `127.0.0.1` at a random port, opened in the browser. The URL carries a random token made for that run, which the page exchanges for an `HttpOnly`, `SameSite=Strict` cookie. Requests with another `Host` header are refused (DNS rebinding). Requests that change something must carry an `X-OpenMCP` header, which a page on another site can't send without a CORS preflight that this server never grants. The page is plain HTML and JavaScript inside the package: no build step and no CDN, so it works offline. Stored text is always shown as text, never as HTML.
- **Proposals:** each one shows its source conversations, and the similar memory or the memories it would merge. Actions: approve, edit and approve, approve as a replacement for the similar memory, reject, and approve or reject all.
- **Memories:** search and filter (scope, kind, project), edit (a new version, as `update_memory`), delete (`forget`), and history.
- **Actions** are taken as the owner, like the CLI, and logged in the audit log as client `review`. An approved memory keeps the proposing client as its source, and records the proposal and source conversations in its metadata.

**Done when:**
- a client's `consolidate` run over captured chats leaves proposals with their sources;
- approving one makes a memory that other clients see in `get_profile`;
- rejecting one keeps it from coming back;
- a near-duplicate pair can be merged into one memory with history;
- the page refuses requests without the token or with a foreign `Host`;
- tested, including scopes: a client can't consolidate or propose from conversations it can't read.
- **Status (2026-10-19):** built and tested (`tests/test_consolidation.py`, `tests/test_review.py`). The page's script was run against real API responses with a stand-in DOM, and the server was probed with curl: no token, a foreign `Host`, and a change without the header are all refused.
  - **Verified with real clients (2026-10-19):**
    - **First run:** Cursor read two chats, from Claude Desktop and from Cursor itself, and proposed two memories citing both. One was approved and one rejected on the page; both decisions were audited as `review`.
    - **Gaps it found, now fixed:**
      - a proposal couldn't name the memory it updates (the `replaces` field);
      - the page's counts went stale while it was open.
    - **Second run:** a chat grew in Claude Desktop. Consolidation returned only its new messages, and Cursor proposed a project update with `replaces` naming the older project memory. Approving it made the older memory history (`superseded_by`), with provenance in the new memory's metadata.
    - **Not yet seen on real data:** merging near-duplicates, which is tested.

### Milestone map (after M5)

M0 to M5 are done: v1 is M0 to M4, and M5 (consolidation and review) is verified with real clients. The order below is the build order agreed on 2026-10-19. Features come first (M6 and M7), then a period of real use (M8) and the public release (M9). Employer approval for M9 has the longest lead time, so it can start now, alongside M6.

| | Milestone | Depends on |
|---|---|---|
| M6 | Background service | — |
| M7 | Automatic consolidation | M6 |
| M7.5 | OpenMCP as a library | — (experimental until M9's rename) |
| M8 | Use it for real | M6, M7 |
| M8.5 | Documentation | everything before it; pages can be written as features land |
| M9 | Public release | approval (start now) |
| M10 | Web and mobile clients | M9 |
| M11 | Transcript standard | M9, and evidence from M6 and M8 |

### M6: Background service
- `openmcp daemon`, installed as a per-user service (launchd on macOS, systemd user units on Linux, a scheduled task or service on Windows). It shares the store with the `serve` processes, like any other connection (WAL).
- **Live capture:** transcripts are captured as they change (file watching), not only when a client next reads. Client hooks are used where a client has them.
- **An OpenTelemetry endpoint** (OTLP/HTTP on 127.0.0.1) for any app or library that records conversations following OpenTelemetry's GenAI semantic conventions. It has no app-specific code; the app is its `service.name`, scoped like any client.
- **Permission first:** nothing runs in the background until the user allows it. No OS asks for a per-user background task (macOS only reports it afterwards, under Login Items), so OpenMCP asks itself: a native Allow / Don't Allow dialog when there's a desktop session (osascript, a PowerShell message box, zenity or kdialog), else the terminal. The default is no, `--yes` is for scripts, and the decision is audited.
- `openmcp setup` offers to install the service when there's something to capture, and `openmcp status` reports whether it is running.
- **Capture by folder** (added after M6, low-key: config only, with one dimmed line in `setup`):
  - `[capture] include_folders` and `exclude_folders` limit transcript capture to some projects, for apps whose transcripts record a folder: Claude Code's working directory, and VS Code's workspace. Subfolders count, exclusions win, and with `include_folders` set, chats with no folder aren't captured.
  - Each transcript's folder is remembered (migration 014), so a skipped one isn't read again while it's excluded, and is read from the start once it's captured.
  - Excluding a folder later stops capturing its new turns; what was already stored stays until the user deletes it.
- **Done when:** an app instrumented with OpenTelemetry's standard GenAI instrumentation is captured live, with no app-specific code and no client reading first, and the service survives a reboot on macOS, Linux and Windows (CI covers what it can). (First written as "a Claude Code chat captured through OpenTelemetry". Claude Code uses its own event names, not the standard's, so that would have needed an app-specific mapping, which OpenMCP doesn't keep. Its chats come in through the transcript reader instead.)
- **Decisions while building:**
  - **Polling:** transcripts are checked every few seconds (`[daemon] interval`) rather than watched with a file-system library. Unchanged files cost one stat each, and there's no new dependency.
  - **OpenTelemetry follows the standard only** (open-telemetry/semantic-conventions-genai, status Development):
    - **What it reads:** a model call is a span or a `gen_ai.client.inference.operation.details` event, carrying `gen_ai.input.messages` (the history) and `gen_ai.output.messages` (the answer), as `{role, parts}` messages. It also reads the earlier version of the conventions, which released instrumentation still emits (`opentelemetry-instrumentation-openai-v2` 2.3): one `gen_ai.*.message` or `gen_ai.choice` log event per message, tied to its call by span id.
    - **Conversations:** a conversation is `gen_ai.conversation.id`, or, when the library has none, its opening message. Messages are identified by role and text, so the history re-sent with every call changes nothing.
    - **Recaps:** a standard `tool_call` part naming a recall tool marks the turn as a recap.
    - **What's kept:** only the text of user and assistant messages; system instructions, tool output, reasoning and every other attribute are dropped.
    - **History:** the first version of M6 mapped Claude Code's own event names instead (migration 012). The user chose not to keep app-specific code, so migration 013 removes that, and `openmcp daemon telemetry` prints only the standard OTLP settings.
  - **Encodings:** OTLP/HTTP logs and traces, in JSON or protobuf, optionally gzipped. Python's official exporter only sends protobuf, so it's read directly from the wire format for the few OTLP messages used, with no added package. Tests check this against the official `opentelemetry-proto` encoder, a test-only dependency. Metrics are accepted and dropped, so a shared endpoint doesn't make exporters complain.
  - **Receiver checks:** it needs the config's token in `x-openmcp-token`, its own `Host`, and `application/json` or `application/x-protobuf`. A web page can send neither without a CORS preflight, which the receiver never grants.
  - **Never edits another app's files:** `openmcp daemon telemetry` prints the settings rather than writing them.
- **Status (2026-10-19):** built and tested (`tests/test_otel.py`, `tests/test_daemon.py`, including the daemon run as a real process). Verified on macOS:
  - **OpenTelemetry's official Python SDK and OTLP exporter**, configured only through the standard environment variables the command prints (protobuf, gzip, token header), sent a two-call conversation. It was stored as one conversation, filed under the app's `service.name`.
  - **OpenTelemetry's released OpenAI instrumentation** (`opentelemetry-instrumentation-openai-v2` 2.3, earlier conventions) wrapped an ordinary OpenAI app, against a local stand-in API. Its message events became one conversation, identified by its opening message, with the system prompt left out. The newest release, 2.4b0, implements today's conventions but didn't import with its dependencies at the time, so it's covered by tests only.
  - (Earlier, before the mapping was made standard-only, a real Claude Code 2.1 chat had come through Claude Code's own events. It confirmed the transport and the token check.)
  - **The launchd agent** started at install, came back 10 seconds after being killed, and was fully removed by `uninstall`.
  - **Not yet tested:** an actual reboot, and the Linux and Windows services on real machines (their service definitions are tested).

### M7: Automatic consolidation
- **Off by default, and nothing to install.** Without setup, consolidation happens only when the user asks an assistant to (M5). OpenMCP ships no model.
- **A model OpenMCP calls itself, brought by the user,** through the open standard for LLM APIs rather than any provider's SDK:
  - **Open Responses** (openresponses.org, spec 2026-04-24): `POST {endpoint}/responses` with `model`, `instructions` and `input`; the text is in `output` items of type `message`, as `output_text` parts. OpenAI, Ollama, LM Studio, vLLM, OpenRouter, Hugging Face and others implement it.
  - **Chat Completions** (`api = "chat-completions"`) for servers that don't speak it yet, such as some providers' compatibility endpoints.
  - The client is `httpx`, already a dependency of the MCP SDK and now declared. Structured output isn't part of Open Responses compliance, so the model is asked for JSON and the text is parsed leniently.
- **Local or remote:**
  - A local endpoint (loopback) is called with proxy settings ignored and without following redirects, so nothing leaves the machine.
  - Any other host needs the user's permission first, from the same dialog as M6, naming the host. Its `allowed_host` is kept in config, and changing the endpoint asks again.
  - API keys go in `~/.openmcp/consolidation.key` (0600), because the OS-run daemon doesn't get the user's environment; `OPENMCP_MODEL_API_KEY` overrides it for manual runs.
- **The work,** the same pipeline as M5:
  - One model call per conversation: its new messages, wrapped as data, plus up to 30 related memories in its scope (nearest by meaning, or newest).
  - Proposals go through `propose_memories`, so scopes, the secret filter, duplicate and rejection checks, and review are unchanged; they're credited to `model:<name>`.
  - **Drift detection:** a memory the conversation contradicts is proposed as its correction with `replaces`. A `replaces` naming a memory the model wasn't shown is cleared.
  - A model error stops the run, and the conversation stays due.
- **Scheduled by M6:** daily at `[consolidation] schedule` (03:00), in its own thread. A run missed while the machine slept happens when the daemon next looks. `openmcp consolidate run` runs it now.
  - Every run is recorded in `consolidation_runs` (migration 015) and shown by `openmcp status` and `openmcp consolidate status`.
- **CLI:** `openmcp consolidate setup --endpoint URL --model NAME [--api chat-completions] [--key]` sends a one-line test request before turning it on. `openmcp consolidate off [--forget]` turns it off, and `--forget` also deletes the key and the allowed host.
- **Done when:** proposals appear overnight without asking an assistant, they're as reviewable as M5's, and a local endpoint sends nothing off the machine (tested). This was first written as "the local model" with Ollama in mind; the user didn't want a model install, so any local endpoint counts.
- **Status (2026-10-19):** built and tested (`tests/test_consolidate_auto.py`).
  - Both protocols are tested against recorded requests; the run, drift, errors, the schedule and the CLI are covered.
  - The local-only check runs a real local stand-in server, with every outgoing connection recorded and a proxy configured in the environment. Only 127.0.0.1 is contacted, and with the guard removed the test fails.
  - **Not yet run:** against a real model, which waits for the user to pick one.

### M7.5: OpenMCP as a library
Agents and programs the user writes get the same shared memory as their assistants, in-process, with no server: `from openmcp import Memory`. It's the M5/M6 store and rules (scopes, secret filter, wrapping, embeddings, audit), not a second implementation: the library calls the same tool functions the MCP server does.

```python
from openmcp import Memory

memory = Memory("trip-planner")      # a client name, like an app's MCP clientInfo: [clients.trip-planner] scopes it

@memory.recall(query="request")      # before each call: related memories and chats, passed in as `context`
@memory.remember(question="request") # after each call: the request and the result, stored as an exchange
def plan(request: str, context: str = "") -> str: ...

memory.save("Prefers aisle seats.", kind="preference")
hits = memory.search("seat preference")        # Hit(text, wrapped, type, kind, source, conversation_id, score, ...)
memory.context("seat preference")              # a prompt-ready block: the stored-data notice and wrapped items
memory.profile(); memory.recent(); memory.record(question, answer); memory.update(id, text); memory.forget(id)
```

- **Decorators:** `@memory.remember` and `@memory.recall` work on sync and `async` functions; the store work runs in a thread for async ones.
  - **Errors:** a store failure is logged (`openmcp` logger) and never breaks the decorated function. Direct calls raise `OpenMCPError`.
- **What `remember` records:** the question is the named argument (`question=`), or the first string argument, or all arguments. The answer is the return value as text.
  - **Conversations:** calls are grouped into one conversation per decorated function, per `Memory`, per process. `conversation=` names a group, and `with memory.conversation("trip-42"):` groups across functions.
- **What `recall` passes in:** `memory.context(...)` goes into the argument named by `into=` (default `context`). The function must accept it; that's checked when it's decorated.
  - `profile=True` adds the profile, and `limit=` and `kinds=` narrow the search.
- **Text and wrapping:** results carry `wrapped` text (Section 8), for putting in a prompt, and plain `text`, for program logic.
- **Home:** `Memory(..., home=path)` overrides the data directory, which is otherwise the same as the CLI's, so the agent shares the store with the user's apps.
- **Import cost:** `import openmcp` stays cheap. `Memory` loads lazily, and the embedding model only when something needs a vector, as in the server.
- **Experimental until M9.** The import name changes with the final name. A lighter install without the embedding runtime is an M9 packaging question.
- **Done when:**
  - a decorated function's exchanges are found by an assistant through MCP;
  - the function receives what an assistant saved;
  - scopes apply by client name;
  - sync and async both work;
  - a store failure doesn't break the function (tested).
- **Status (2026-10-19):** built and tested (`tests/test_library.py`): every "done when" item, plus importing `openmcp` loading nothing else. Documented in [docs/library.md](docs/library.md).

### M8: Use it for real
- **A week of daily use** across the user's apps, with consolidation running.
- **`openmcp report`:** a usage report from the audit log:
  - what was recalled, and by which client;
  - searches that found nothing;
  - which clients recorded, and which were captured;
  - proposals approved and rejected.
- **Fix what hurts,** including the slow-runner timeout in the stdio scope test.
- **Done when:** a week of real use needs no manual database fixes, and the report gives an evidence-based list of what to build next.

### M8.5: Documentation
Every public surface documented in `docs/`, written for someone who has never seen the project, before the public release (M9). Pages can be written as features land; the first, [the model endpoint for consolidation](docs/consolidation-endpoint.md), came with M7.
- **Getting started:** install, connect each kind of app, the standing instruction, what to expect.
- **Guides:** scopes, capture (transcripts, folders, the daemon, OpenTelemetry), consolidation and review, import and export, deleting things.
- **Reference:**
  - every CLI command and option;
  - every `config.toml` key;
  - every MCP tool, prompt and resource, with arguments and results;
  - the OpenMCP export format, as JSON Schema;
  - what the OpenTelemetry endpoint accepts (GenAI conventions, encodings, the token);
  - the consolidation endpoint contract;
  - the library API (M7.5).
- **Security and privacy:** what's stored where, what leaves the machine and when, permissions, the threat model, and its limits.
- **Kept current by tests:** a test fails when a CLI command, config key or MCP tool has no reference entry, so the docs can't fall behind.
- **Done when:** a reader can install, connect, configure and troubleshoot from `docs/` alone, and the coverage tests pass.

### M9: Public release
- **Pre-flight (Section 13):** employer approval, a final name, a license.
- **New repository:** publish only the clean `main`, because GitHub keeps old PR refs.
- **A package on PyPI** (`uv tool install <name>`).
- **A README** for people who've never seen the project, linking the documentation (M8.5).
- **The OpenMCP format** published as JSON Schema.
- **Done when:** someone else, on a clean machine, installs it and connects two apps in under five minutes from the README alone.

### M10: Web and mobile clients
- An opt-in tunnel, with MCP's authorization spec (OAuth), so web and mobile clients (claude.ai connectors, ChatGPT) can reach the local server. It works only while the machine is awake.
- Scopes apply as for any client, and a remote client gets nothing until the user grants it scopes.
- **Done when:** claude.ai on a phone reads a memory saved from Cursor, and a security review of the tunnel and authorization finds nothing blocking.

### M11: Transcript standard
A standard for conversation transcripts, so no client needs a transcript reader: what MCP did for tools, for conversations. The schema is the easy part (OpenMCP's conversation format, published as JSON Schema, plus rules for chats that grow: events appended as they happen, stable message ids for in-place updates, tool use marked); adoption is the hard part. Routes, most to least promising:
  1. **An MCP extension**: a server declares it wants conversation turns, and a client that supports it sends each user and assistant turn to the servers the user approved. It replaces both `record_exchange` (model-dependent) and transcript readers (undocumented formats), and the client owns consent. Needs proposing through MCP's own process.
  2. **A published file schema** clients write to disk: simpler, but needs every vendor to adopt it without a protocol pulling them.
  3. **OpenTelemetry's GenAI semantic conventions**, a standard that already exists: some clients export telemetry, and OpenMCP could receive it like any backend (an OTLP receiver). Depends on what each client actually sends (see Section 14).

  Meanwhile transcript readers stay optional adapters into the schema, deleted as clients produce it natively.
- **Done when:** the MCP extension is written up and proposed through MCP's process, with OpenMCP as the reference server and one open-source client implementing it.

---

## 12. Later (unscheduled)

Built when there's a reason to, not on the milestone map:

- **More importers and capture sources** beyond the M3.5 registries' first entries, each checked against a real export or install.
- **Vector compression**, when stores grow well past 100k chunks (the exact scan is ~60ms there). sqlite-vec stores only float32, int8, and bit vectors, so TurboQuant or PolarQuant as published would need a custom scan outside SQLite; their core idea (a fixed random rotation, so one calibrated quantizer fits every coordinate) can be applied to sqlite-vec's int8 storage. Its built-in `vec_quantize_int8(v, 'unit')` wastes most of the range on bge vectors (coordinates ≈ ±0.05) and kept only 82% of the true top 10 in a test.
- **A stronger embedding model**, if bge-small's misses matter in practice (Section 14).
- **Per-user evals**: auto-generated tests checking whether stored context actually improves answers.

---

## 13. Pre-flight before going public

- [ ] Get employer approval for publishing open source work in this area. Keep the repo private until then.
- [ ] Pick a final name and check package-registry and GitHub availability.
- [ ] Confirm no real personal data exists anywhere in the git history.
  - **Done (2026-10-19), for `main`:**
    - **Scan:** all commits, file versions, commit messages and PR texts. One real project name from a live test was found in a test, a docstring, a commit message and a PR description.
    - **Rewrite:** `git filter-repo` replaced that name with synthetic text, and moved every commit's author email to the GitHub noreply address. The rewritten `main` was force-pushed; the final tree was unchanged and the tests passed.
    - **Cleanup:** the merged feature branches were deleted, the PR description was edited, and a rescan found nothing.
  - **Still on GitHub:** each PR's original commits under `refs/pull/N/head`. Git can't delete these, so PR pages still show the old commits and authors. Before going public, either publish as a new repository with only the clean `main`, or ask GitHub Support to purge them. A new repository is simpler, and fits the rename. Rescan whatever gets published.
- [ ] Choose a license (MIT or Apache-2.0 are the usual picks).

## 14. Open questions

- Should `get_profile` be called automatically at session start, or left to the model? (Depends on client support for resources versus tools.)
- ~~What's the right near-duplicate threshold?~~ 0.92 with bge-small: rewordings score 0.97–0.995, different facts 0.73–0.84 ("allergic to peanuts" vs "shellfish": 0.84), but true paraphrases sit right on the line ("Allergic to peanuts." vs "Has a peanut allergy.": 0.920). Revisit on real data.
- Search similarity cutoffs (0.55 floor, 0.08 window) come from a 20-question synthetic set: 18/20 paraphrases found, ~1 unrelated item per query. bge-small misses some paraphrases entirely: "how was I planning to deal with the fitness app throttling me" ranks the Strava 429 memory 11th (similarity 0.48). A test records this as an expected failure, so a better model shows up as a test to update.
- ~~Should imported chats ever be readable by default, or opt-in per scope?~~ They go to the default write scope, or `--scope`; which clients can read them follows that scope's config like everything else.
- ~~How reliably do models call `record_exchange` unprompted?~~ Measured in the M1 manual test (2026-10-17): Claude Desktop and Copilot made **no** recording calls from tool descriptions and server instructions alone. With a standing instruction, Claude Desktop recorded every technical turn but skipped a casual one. Hence standing instructions via `openmcp setup` (M1) and transcript capture (M1.5). Open: how well standing instructions hold up over long chats, and in clients that show no transcript on disk (Claude Desktop, claude.ai).
- Recaps recorded through `record_exchange` despite the instructions aren't marked (Section 7.2). Telling a recap from new content by what it says (does the answer mostly restate what the lookup returned?) may work where timing didn't. The verbatim question "what was the pet thing I asked Copilot about?" scores 0.543 against the original chat, just under the similarity floor, so only a topic search ("cat names": 0.81) finds it by meaning.
- Do clients' OpenTelemetry exports carry whole conversations, and can OpenMCP use them instead of transcript readers? **Experiment (2026-10-18)**, a local OTLP/HTTP receiver:
  - **Claude Code 2.1** (`CLAUDE_CODE_ENABLE_TELEMETRY=1`, `OTEL_LOG_USER_PROMPTS=1`) sends `user_prompt` (the prompt text) and `assistant_response` (the answer text) log events, linked by `prompt.id`, with `message.uuid` and `session.id`. That's enough to rebuild conversations live. The transport is standard (OTLP), but the event names are Claude Code's own, not OpenTelemetry's GenAI conventions. M6 briefly mapped them, then dropped the mapping so that OpenTelemetry capture has no app-specific code; Claude Code's chats come in through its transcript reader.
  - **Cursor 3.25** uses GenAI attributes only for request details (model, tokens), with no message text and no user-configurable export.
  - **Copilot Chat 0.26.7**: inconclusive (it bundles the OpenTelemetry libraries, apparently for internal telemetry).
  - Trade-off for Claude Code: a receiver must be running whenever Claude Code runs, and can't backfill, where the transcript reader needs no setup and backfills. **Resolved in M6:** the daemon's OpenTelemetry endpoint reads only the standard GenAI conventions, so Claude Code's chats stay with its transcript reader until it emits those.
- Should OpenMCP propose an MCP extension for clients to share conversation turns with servers the user approves (Section 12)? Its usage would make the case.
- Claude Code's `--resume` and VS Code's chat export may copy history into a new session file, which is captured as a separate conversation. Dedupe across sessions if it turns out to matter.
