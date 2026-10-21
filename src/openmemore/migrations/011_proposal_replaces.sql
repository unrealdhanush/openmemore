-- A proposal can name the memory it updates (SPEC.md M5): the proposing model says so,
-- where similar_memory_id is only what the vectors suggest.
ALTER TABLE proposals ADD COLUMN replaces_memory_id INTEGER REFERENCES memories(id) ON DELETE SET NULL;
