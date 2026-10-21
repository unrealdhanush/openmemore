-- History is searchable by meaning (SPEC.md M7.7): a replaced memory keeps its vector, so
-- search_context(mode="history") finds old versions as well as current ones. Versions
-- replaced before this lost theirs, so they're queued to be embedded again.
DROP TRIGGER memories_vec_superseded;
UPDATE memories SET embedded = 0 WHERE superseded_by IS NOT NULL;
