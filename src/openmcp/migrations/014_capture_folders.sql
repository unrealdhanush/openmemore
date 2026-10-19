-- Capture by folder ([capture] include_folders / exclude_folders): each transcript's
-- project folder is remembered, so one from a folder that isn't captured is checked once
-- and not read again, and is read from the start if its folder is captured later.
ALTER TABLE capture_files ADD COLUMN folder TEXT;
ALTER TABLE capture_files ADD COLUMN filtered INTEGER NOT NULL DEFAULT 0;
