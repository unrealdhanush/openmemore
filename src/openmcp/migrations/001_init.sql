-- Core tables (SPEC.md Section 6). FTS5 and sqlite-vec tables arrive in later migrations.

CREATE TABLE schema_version (version INTEGER NOT NULL);

CREATE TABLE conversations (
  id            INTEGER PRIMARY KEY,
  provider      TEXT NOT NULL,
  external_id   TEXT NOT NULL,
  title         TEXT,
  scope         TEXT NOT NULL DEFAULT 'personal',
  created_at    TEXT,
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

CREATE TABLE chunks (
  id          INTEGER PRIMARY KEY,
  message_id  INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
  ord         INTEGER NOT NULL,
  text        TEXT NOT NULL
);
CREATE INDEX chunks_message_id ON chunks (message_id);

CREATE TABLE memories (
  id             INTEGER PRIMARY KEY,
  kind           TEXT NOT NULL CHECK (kind IN ('fact','preference','project','decision','note')),
  subject        TEXT,
  content        TEXT NOT NULL,
  project        TEXT,
  scope          TEXT NOT NULL DEFAULT 'personal',
  source_client  TEXT NOT NULL,
  confidence     REAL CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
  content_hash   TEXT NOT NULL,
  created_at     TEXT NOT NULL,
  updated_at     TEXT NOT NULL,
  superseded_by  INTEGER REFERENCES memories(id)
);
CREATE INDEX memories_active_hash ON memories (content_hash) WHERE superseded_by IS NULL;
CREATE INDEX memories_superseded_by ON memories (superseded_by);
CREATE INDEX memories_scope_kind ON memories (scope, kind);

CREATE TABLE audit_log (
  id            INTEGER PRIMARY KEY,
  ts            TEXT NOT NULL,
  client        TEXT NOT NULL,
  tool          TEXT NOT NULL,
  summary       TEXT,
  result_count  INTEGER
);
CREATE INDEX audit_log_ts ON audit_log (ts);

CREATE TABLE ingest_jobs (
  id           INTEGER PRIMARY KEY,
  provider     TEXT NOT NULL,
  file_path    TEXT NOT NULL,
  file_hash    TEXT NOT NULL UNIQUE,
  status       TEXT NOT NULL CHECK (status IN ('queued','running','done','failed')),
  total        INTEGER,
  processed    INTEGER NOT NULL DEFAULT 0,
  cursor       TEXT,
  error        TEXT,
  started_at   TEXT,
  finished_at  TEXT
);
