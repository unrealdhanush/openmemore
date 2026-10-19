-- A proposal can say it conflicts with a memory when it's unclear which is right (SPEC.md
-- M7.7), unlike replaces, which is a confident correction. The user settles it in review.
ALTER TABLE proposals ADD COLUMN contradicts_memory_id INTEGER REFERENCES memories(id) ON DELETE SET NULL;
