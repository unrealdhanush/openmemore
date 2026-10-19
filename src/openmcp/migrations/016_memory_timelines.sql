-- Two timelines for memories (SPEC.md M7.7, after LongMemory's bitemporal model):
-- valid_from / valid_to say when a memory was true in the user's life; created_at and
-- superseded_at say when OpenMCP learned it and when it learned otherwise. A correction
-- closes the old memory's validity where the new one starts; a memory whose validity has
-- ended is history even if nothing replaced it.
ALTER TABLE memories ADD COLUMN valid_from TEXT;
ALTER TABLE memories ADD COLUMN valid_to TEXT;
ALTER TABLE memories ADD COLUMN superseded_at TEXT;

UPDATE memories SET valid_from = created_at;
UPDATE memories
   SET superseded_at = updated_at,
       valid_to = (SELECT n.created_at FROM memories n WHERE n.id = memories.superseded_by)
 WHERE superseded_by IS NOT NULL;

-- A proposal carries when its statement became true: what the model says, or when the
-- conversation it came from happened, not when the user approves it.
ALTER TABLE proposals ADD COLUMN valid_from TEXT;
ALTER TABLE proposals ADD COLUMN valid_to TEXT;

CREATE INDEX memories_current ON memories (scope, kind) WHERE superseded_by IS NULL AND valid_to IS NULL;
