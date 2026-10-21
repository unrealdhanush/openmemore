-- metadata JSON columns for provider-specific / experimental fields,
-- and per-call timing for the performance budget (Section 4.3).

ALTER TABLE conversations ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(metadata));
ALTER TABLE memories ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(metadata));

-- ADD COLUMN ... NOT NULL needs a default; rows written before this migration read as 0.
ALTER TABLE audit_log ADD COLUMN duration_ms REAL NOT NULL DEFAULT 0;
