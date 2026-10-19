-- Capture through OpenTelemetry (SPEC.md M6). A Claude Code turn arrives as a user_prompt
-- event and, seconds later, one assistant_response event per model call, all linked by
-- prompt.id. These rows link the later events to the conversation and message they
-- belong to, and keep the answer's parts so it can be rebuilt as they arrive. Deleting
-- the conversation deletes them.

CREATE TABLE otel_prompts (
  prompt_id        TEXT PRIMARY KEY,
  conversation_id  INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  message_uuid     TEXT NOT NULL,
  recall           INTEGER NOT NULL DEFAULT 0,
  created_at       TEXT NOT NULL
);
CREATE INDEX otel_prompts_conversation ON otel_prompts (conversation_id);

CREATE TABLE otel_responses (
  prompt_id   TEXT NOT NULL REFERENCES otel_prompts(prompt_id) ON DELETE CASCADE,
  sequence    INTEGER NOT NULL,
  text        TEXT NOT NULL,
  created_at  TEXT,
  PRIMARY KEY (prompt_id, sequence)
);
