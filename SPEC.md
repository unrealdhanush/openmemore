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

Any connected model can **write** what it learns (preferences, projects, decisions) and **read** what other models learned. Start a project in Claude, continue it in another assistant tomorrow without re-explaining anything.

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
5. **Legitimate access only.** No scraping, no session cookies. Context comes in through MCP writes and official data exports.

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
- **Ingestion never runs inside `serve`.** Bulk import is a CLI command with resumable jobs, so a large import can't stall a chat.

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
| Storage | SQLite (stdlib `sqlite3`), WAL mode |
| Keyword search | SQLite FTS5 |
| Vector search | `sqlite-vec` |
| Embeddings | `fastembed` with `BAAI/bge-small-en-v1.5` (384-dim, local, ONNX) |
| CLI | `typer` + `rich` (progress bars) |
| Config | `tomllib` (read), `tomli-w` (write) |
| Tests | `pytest` |
| Manual testing | MCP Inspector (`npx @modelcontextprotocol/inspector`) |

Verify current SDK APIs against the installed versions rather than relying on memory.

---

## 5. Repo layout

```
openmcp/
  pyproject.toml
  README.md
  SPEC.md
  src/openmcp/
    __init__.py
    cli.py              # typer app: init, serve, import, status, memories, audit, export
    config.py           # load/write config.toml, client scope resolution
    db.py               # connection, pragmas, migrations
    migrations/         # 001_init.sql, ...
    server.py           # FastMCP server, tool + prompt registration
    tools/
      memories.py       # save, update, forget
      search.py         # hybrid search
      profile.py        # get_profile
    safety/
      secrets.py        # secret/PII filter on writes
      wrap.py           # untrusted-data wrapping on reads
    embed.py            # fastembed wrapper, lazy-loaded
    ingest/
      jobs.py           # job table, resumable runner
      normalize.py      # common conversation/message schema
      chatgpt.py        # ChatGPT export parser
      claude.py         # Claude export parser
      detect.py         # auto-detect provider from zip contents
    audit.py
  tests/
    fixtures/           # SYNTHETIC export samples only
    test_*.py
```

---

## 6. Data model

Logical schema. Implementation details (FTS5 external-content wiring, triggers, vec0 syntax) are up to the implementer.

```sql
-- Imported or client-written conversation history
CREATE TABLE conversations (
  id            INTEGER PRIMARY KEY,
  provider      TEXT NOT NULL,          -- 'chatgpt' | 'claude' | 'gemini' | ...
  external_id   TEXT NOT NULL,          -- provider's conversation id
  title         TEXT,
  scope         TEXT NOT NULL DEFAULT 'personal',
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
  content_hash     TEXT NOT NULL,
  UNIQUE (conversation_id, content_hash)
);

-- Messages split for embedding (long messages -> several chunks)
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
  created_at     TEXT NOT NULL,
  updated_at     TEXT NOT NULL,
  superseded_by  INTEGER REFERENCES memories(id)   -- history kept on update
);

-- Full-text: FTS5 over memories(subject, content) and chunks(text)
-- Vectors:   sqlite-vec vec0 tables, float[384], keyed by memories.id and chunks.id

CREATE TABLE audit_log (
  id            INTEGER PRIMARY KEY,
  ts            TEXT NOT NULL,
  client        TEXT NOT NULL,
  tool          TEXT NOT NULL,
  summary       TEXT,                  -- short, never the full payload
  result_count  INTEGER
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

CREATE TABLE schema_version (version INTEGER NOT NULL);
```

Rules:
- "Active" memories are those with `superseded_by IS NULL`.
- `update_memory` never overwrites. It inserts a new row and points the old one at it.
- `forget` is a **hard delete** of the row, its history chain, and its vectors. The user owns their data.

---

## 7. MCP surface (v1)

Every tool call:
1. Resolves the calling client's scopes from `config.toml` (Section 9).
2. Writes one `audit_log` row.
3. Returns read results wrapped per Section 8.

Tool descriptions matter: models decide when to call tools based on them. Write them carefully and keep them specific.

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
- Exact duplicate (same normalized hash, active) → skip.
- Near-duplicate (cosine ≥ 0.92, same kind) → **save anyway** but report it in `possible_duplicates`, so the calling model can follow up with `update_memory` or `forget`. No silent merging in v1.
- Description should tell the model to save when the user states a preference, makes a decision, or finishes meaningful work on a project, and to **never save secrets, credentials, or other people's private details**.

### 7.2 `search_context`
```
search_context(
  query: str,
  kinds?: list[str],
  project?: str,
  include_chats?: bool = true,
  limit?: int = 10            # max 25
) -> { results: [{ type: 'memory'|'chat', id, content, kind?, project?, source, created_at, score }] }
```
- Hybrid: FTS5 (BM25) + vector similarity, merged with reciprocal rank fusion, plus a mild recency boost.
- Only searches scopes the client may read.
- Chat results return the matching chunk plus conversation title and provider, not whole conversations.

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

---

## 8. Safety

### Writes: secret filter (`safety/secrets.py`)
Reject items matching common patterns: API keys and tokens (`sk-…`, `ghp_…`, AWS keys, JWT-shaped strings, private-key headers), payment card numbers (Luhn-checked), US SSN pattern, and lines that look like `password: …`. Return the rejection reason; never echo the matched value back.

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
- **Done when:** MCP Inspector connects and `ping` works; tests cover init and migrations.

### M1 — Memory store
- `save_memories`, `update_memory`, `forget`, `get_profile`, `search_context` (**FTS5 only** for now).
- Secret filter, read wrapping, audit log.
- `import_my_memory` prompt and `openmcp://profile` resource.
- CLI: `openmcp memories list|show|rm`, `openmcp audit`.
- **Done when:** in Claude Desktop, "save that I prefer Python and terse answers," then a new chat with "what do you know about my preferences?" retrieves it via the tools.

### M2 — Semantic search
- `embed.py` with lazy model loading (don't slow server startup).
- `sqlite-vec` tables; embed on write.
- Hybrid search with reciprocal rank fusion and recency boost; near-duplicate detection in `save_memories`.
- **Done when:** a paraphrased query finds a memory that keyword search alone misses (test it).

### M3 — Import
- Job table, ChatGPT and Claude adapters, auto-detect, newest-first batching, resume.
- `openmcp import`, `openmcp status` with a `rich` progress bar.
- `search_context` returns chat chunks.
- **Done when:** a synthetic export of 500 conversations imports, the newest 50 are searchable before the job finishes, and killing and re-running resumes without duplicates.

### M4 — Scopes and ownership
- Enforce read and write scopes on every tool.
- `openmcp export` dumps all memories and conversations to JSON.
- README: install, per-client config snippets, the safety model, honest limitations.
- **Done when:** a `cursor` client scoped to `work` cannot see `personal` memories (tested), and the cross-client success test below passes.

### v1 success test
Start a project in Claude Desktop and save context. The next day, open a different client (Cursor or Gemini CLI) and continue the project **without re-explaining anything**.

---

## 12. Later (do not build in v1)

- **Consolidation ("dream mode")**: scheduled job that distills imported chats into memories and merges near-duplicates, with the user reviewing changes.
- **Drift detection**: notice when recent behavior contradicts stored preferences, lower confidence, and ask instead of assuming.
- **Remote access**: opt-in tunnel plus strong auth so web and mobile clients (claude.ai connectors, ChatGPT) can reach the local server. Works only while the device is awake.
- **Review UI**: small local web page to browse, edit, and approve memories.
- **More importers**: Gemini (Google Takeout), others.
- **Per-user evals**: auto-generated tests checking whether stored context actually improves answers.

---

## 13. Pre-flight before going public

- [ ] Get employer approval for publishing open source work in this area. Keep the repo private until then.
- [ ] Pick a final name and check package-registry and GitHub availability.
- [ ] Confirm no real personal data exists anywhere in the git history.
- [ ] Choose a license (MIT or Apache-2.0 are the usual picks).

## 14. Open questions

- Should `get_profile` be called automatically at session start, or left to the model? (Depends on client support for resources versus tools.)
- What's the right near-duplicate threshold? Tune it on real data in M2.
- Should imported chats ever be readable by default, or opt-in per scope?
