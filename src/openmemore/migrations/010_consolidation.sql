-- Consolidation and review (SPEC.md M5): memories proposed from conversations, or merges
-- of near-duplicates, waiting for the user to approve them.

CREATE TABLE proposals (
  id                INTEGER PRIMARY KEY,
  action            TEXT NOT NULL CHECK (action IN ('add','merge')),
  kind              TEXT NOT NULL CHECK (kind IN ('fact','preference','project','decision','note')),
  content           TEXT NOT NULL,
  subject           TEXT,
  project           TEXT,
  scope             TEXT NOT NULL,
  reason            TEXT,
  source_client     TEXT NOT NULL,
  confidence        REAL CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
  content_hash      TEXT NOT NULL,
  similar_memory_id INTEGER REFERENCES memories(id) ON DELETE SET NULL,
  status            TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','rejected')),
  created_at        TEXT NOT NULL,
  decided_at        TEXT,
  memory_id         INTEGER REFERENCES memories(id) ON DELETE SET NULL
);
CREATE INDEX proposals_status_scope ON proposals (status, scope);
CREATE INDEX proposals_hash ON proposals (scope, content_hash);

-- Where a proposal came from.
CREATE TABLE proposal_conversations (
  proposal_id      INTEGER NOT NULL REFERENCES proposals(id) ON DELETE CASCADE,
  conversation_id  INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  PRIMARY KEY (proposal_id, conversation_id)
);
CREATE INDEX proposal_conversations_conversation ON proposal_conversations (conversation_id);

-- The memories a merge replaces.
CREATE TABLE proposal_memories (
  proposal_id  INTEGER NOT NULL REFERENCES proposals(id) ON DELETE CASCADE,
  memory_id    INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  PRIMARY KEY (proposal_id, memory_id)
);
CREATE INDEX proposal_memories_memory ON proposal_memories (memory_id);

-- Deleting really deletes: a proposal made from a deleted conversation goes too, unless
-- the user already approved it; a merge of a deleted memory can't be approved any more.
CREATE TRIGGER conversations_delete_proposals BEFORE DELETE ON conversations BEGIN
  DELETE FROM proposals WHERE status != 'approved' AND id IN (
    SELECT proposal_id FROM proposal_conversations WHERE conversation_id = old.id
  );
END;

CREATE TRIGGER memories_delete_merges BEFORE DELETE ON memories BEGIN
  DELETE FROM proposals WHERE status = 'pending' AND id IN (
    SELECT proposal_id FROM proposal_memories WHERE memory_id = old.id
  );
END;

-- The last message read for consolidation: a conversation is due while it has later ones.
ALTER TABLE conversations ADD COLUMN consolidated_through INTEGER;
