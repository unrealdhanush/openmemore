-- Automatic consolidation runs (SPEC.md M7): when, with which model, what came of it.
CREATE TABLE consolidation_runs (
  id             INTEGER PRIMARY KEY,
  trigger        TEXT NOT NULL CHECK (trigger IN ('schedule', 'manual')),
  model          TEXT NOT NULL,
  host           TEXT NOT NULL,
  started_at     TEXT NOT NULL,
  finished_at    TEXT,
  conversations  INTEGER NOT NULL DEFAULT 0,
  proposals      INTEGER NOT NULL DEFAULT 0,
  error          TEXT
);
CREATE INDEX consolidation_runs_started ON consolidation_runs (started_at);
