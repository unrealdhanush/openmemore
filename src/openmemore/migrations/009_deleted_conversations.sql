-- Deleted conversations are remembered by their source's id, so a later import or
-- a capture of the same conversation doesn't quietly bring it back: the user deleted it.
CREATE TABLE deleted_conversations (
  provider     TEXT NOT NULL,
  external_id  TEXT NOT NULL,
  deleted_at   TEXT NOT NULL,
  PRIMARY KEY (provider, external_id)
);

CREATE TRIGGER conversations_delete_remembered AFTER DELETE ON conversations BEGIN
  INSERT OR IGNORE INTO deleted_conversations (provider, external_id, deleted_at)
  VALUES (old.provider, old.external_id, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));
END;
