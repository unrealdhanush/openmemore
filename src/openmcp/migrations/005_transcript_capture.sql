-- Transcript capture (SPEC.md M1.5): conversations read from the transcript files that
-- clients keep on the user's disk.

-- A captured message keeps the transcript's id for it, so re-reading a transcript that
-- has grown (an answer still being written) updates the message instead of adding a copy.
ALTER TABLE messages ADD COLUMN external_id TEXT;
CREATE UNIQUE INDEX messages_external_id ON messages (conversation_id, external_id)
  WHERE external_id IS NOT NULL;

-- One row per transcript file already read: unchanged files are skipped, and growing
-- ones are re-read from resume_offset (the start of their last turn).
CREATE TABLE capture_files (
  path             TEXT PRIMARY KEY,
  source           TEXT NOT NULL,
  mtime_ns         INTEGER NOT NULL,
  size             INTEGER NOT NULL,
  resume_offset    INTEGER NOT NULL DEFAULT 0,
  conversation_id  INTEGER,
  ignored          INTEGER NOT NULL DEFAULT 0,
  captured_at      TEXT NOT NULL
);

-- Deleting a captured conversation stops its transcript from being captured again;
-- otherwise the next turn in that chat would quietly bring it back.
CREATE TRIGGER conversations_delete_ignores_capture AFTER DELETE ON conversations BEGIN
  UPDATE capture_files SET ignored = 1 WHERE conversation_id = old.id;
END;
