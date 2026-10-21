-- Semantic search (SPEC.md M2): sqlite-vec tables holding one 384-dimension embedding
-- (BAAI/bge-small-en-v1.5) per active memory and per chunk, keyed by their ids.

CREATE VIRTUAL TABLE memories_vec USING vec0(embedding float[384] distance_metric=cosine);
CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[384] distance_metric=cosine);

-- Rows still waiting for an embedding: written while the model wasn't available, before
-- this migration, or by transcript capture, which embeds after storing. Partial indexes
-- keep "what's pending" cheap however large the store gets.
ALTER TABLE memories ADD COLUMN embedded INTEGER NOT NULL DEFAULT 0;
ALTER TABLE chunks ADD COLUMN embedded INTEGER NOT NULL DEFAULT 0;
CREATE INDEX memories_pending_embedding ON memories (id) WHERE embedded = 0 AND superseded_by IS NULL;
CREATE INDEX chunks_pending_embedding ON chunks (id) WHERE embedded = 0;

-- Vectors go with their rows (forget, conversation deletes via ON DELETE CASCADE, re-chunking).
CREATE TRIGGER memories_vec_delete AFTER DELETE ON memories BEGIN
  DELETE FROM memories_vec WHERE rowid = old.id;
END;

CREATE TRIGGER chunks_vec_delete AFTER DELETE ON chunks BEGIN
  DELETE FROM chunks_vec WHERE rowid = old.id;
END;

-- How many chunks each indexed (stemmed) term appears in, read straight from the index,
-- so search can leave out words so common that ranking them costs more than they add.
CREATE VIRTUAL TABLE chunks_fts_terms USING fts5vocab(chunks_fts, 'row');

-- A superseded memory is never searched, so its vector is dropped.
CREATE TRIGGER memories_vec_superseded AFTER UPDATE OF superseded_by ON memories
WHEN new.superseded_by IS NOT NULL BEGIN
  DELETE FROM memories_vec WHERE rowid = old.id;
END;
