-- Relations between memories (after LongMemory's executable edges): one
-- table, written only through openmemore.relations, which applies each link's effect in the
-- same transaction. superseded_by stays as the fast pointer to a memory's newer version.
CREATE TABLE memory_links (
  id               INTEGER PRIMARY KEY,
  type             TEXT NOT NULL CHECK (type IN ('supersedes', 'contradicts', 'supports', 'derived_from')),
  from_memory      INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  to_memory        INTEGER REFERENCES memories(id) ON DELETE CASCADE,
  to_conversation  INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
  source           TEXT,
  created_at       TEXT NOT NULL,
  resolved_at      TEXT,
  resolution       TEXT,
  CHECK ((to_memory IS NULL) <> (to_conversation IS NULL)),
  CHECK (to_conversation IS NULL OR type IN ('derived_from', 'supports'))
);
CREATE INDEX memory_links_from ON memory_links (from_memory, type);
CREATE INDEX memory_links_to ON memory_links (to_memory, type);
CREATE INDEX memory_links_conversation ON memory_links (to_conversation);
CREATE INDEX memory_links_open_contradictions ON memory_links (type) WHERE type = 'contradicts' AND resolved_at IS NULL;

-- Existing history as links: every replacement, and the conversations approved
-- proposals came from (until now kept in the memory's metadata).
INSERT INTO memory_links (type, from_memory, to_memory, source, created_at)
SELECT 'supersedes', superseded_by, id, 'migration', COALESCE(superseded_at, updated_at)
  FROM memories WHERE superseded_by IS NOT NULL AND superseded_by IN (SELECT id FROM memories);

INSERT INTO memory_links (type, from_memory, to_conversation, source, created_at)
SELECT 'derived_from', m.id, j.value, 'migration', m.created_at
  FROM memories m, json_each(m.metadata, '$.conversations') j
 WHERE json_type(m.metadata, '$.conversations') = 'array' AND j.value IN (SELECT id FROM conversations);

UPDATE memories SET metadata = json_remove(metadata, '$.conversations', '$.replaced')
 WHERE json_type(metadata, '$.conversations') IS NOT NULL OR json_type(metadata, '$.replaced') IS NOT NULL;
