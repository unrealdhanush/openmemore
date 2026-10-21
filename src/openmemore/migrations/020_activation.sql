-- Activation (after ACT-R's base-level learning, as LongMemory uses it):
-- how often and how recently a memory or a conversation chunk was presented, counting
-- its creation, each time a client was shown it, and (memories) each confirmation. A
-- count and the last time are enough for Petrov's approximation; no log is kept.
ALTER TABLE memories ADD COLUMN presented INTEGER NOT NULL DEFAULT 0;
ALTER TABLE memories ADD COLUMN last_presented_at TEXT;
ALTER TABLE chunks ADD COLUMN presented INTEGER NOT NULL DEFAULT 0;
ALTER TABLE chunks ADD COLUMN last_presented_at TEXT;
