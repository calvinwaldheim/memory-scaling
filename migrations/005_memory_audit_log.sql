-- Pass A · multi-user safety: append-only audit log for every memory mutation.
--
-- The `memories` row carries the *latest* state of each fact (rule, content,
-- superseded_at, forgotten_at, etc.). What it doesn't carry is the action
-- stream: who did what, when, why, what was the row's state before vs. after.
-- This table is the action-of-record. Audit writes happen in the same
-- transaction as the data change (see storage.py), so the log can never drift
-- from reality.

CREATE TABLE IF NOT EXISTS memory_audit_log (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  memory_id    UUID NOT NULL,
  project_id   TEXT NOT NULL REFERENCES projects(project_id),
  action       TEXT NOT NULL,
  actor        TEXT NOT NULL,
  reason       TEXT,
  before_state JSONB,
  after_state  JSONB,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  CONSTRAINT memory_audit_action_check
    CHECK (action IN ('created','superseded','forgotten','purged','updated'))
);

-- Lookups:
--   - "give me this memory's full history" → by memory_id
--   - "what changed in this project recently" → by project_id, created_at DESC
--   - "what has alice done lately" → by actor, created_at DESC
CREATE INDEX IF NOT EXISTS memory_audit_memory_idx
  ON memory_audit_log(memory_id, created_at DESC);
CREATE INDEX IF NOT EXISTS memory_audit_project_idx
  ON memory_audit_log(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS memory_audit_actor_idx
  ON memory_audit_log(actor, created_at DESC);
