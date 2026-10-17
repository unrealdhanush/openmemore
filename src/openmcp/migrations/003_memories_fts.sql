-- Full-text index over memories(subject, content), kept in sync by triggers.
-- External-content table: the text lives only in `memories`. Superseded rows stay
-- indexed; queries join back to `memories` and filter on superseded_by IS NULL.
-- porter stemming so "preferences" matches "prefer".

CREATE VIRTUAL TABLE memories_fts USING fts5(
  subject,
  content,
  content = 'memories',
  content_rowid = 'id',
  tokenize = 'porter unicode61 remove_diacritics 2'
);

CREATE TRIGGER memories_fts_insert AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts (rowid, subject, content) VALUES (new.id, new.subject, new.content);
END;

CREATE TRIGGER memories_fts_delete AFTER DELETE ON memories BEGIN
  INSERT INTO memories_fts (memories_fts, rowid, subject, content)
  VALUES ('delete', old.id, old.subject, old.content);
END;

CREATE TRIGGER memories_fts_update AFTER UPDATE OF subject, content ON memories BEGIN
  INSERT INTO memories_fts (memories_fts, rowid, subject, content)
  VALUES ('delete', old.id, old.subject, old.content);
  INSERT INTO memories_fts (rowid, subject, content) VALUES (new.id, new.subject, new.content);
END;

-- Index anything written before this migration.
INSERT INTO memories_fts (memories_fts) VALUES ('rebuild');
