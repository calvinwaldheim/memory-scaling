-- Supersede + soft-forget: lineage and audit trail on memories.
-- Adds first-class supersedence so corrections leave a chain (old → new) rather
-- than silently deleting and re-inserting. Also makes `forget` a soft action
-- by default so retracted memories stay queryable for audit purposes.
--
-- Both states (superseded, forgotten) are excluded from default retrieval via
-- partial indexes on the "active" predicate, which keeps the hot recall path
-- fast and fixes the embedding-space contamination that a pure quality-score
-- downgrade would leave behind.

-- 1. Add supersedence columns.
ALTER TABLE memories
  ADD COLUMN IF NOT EXISTS superseded_by      UUID,
  ADD COLUMN IF NOT EXISTS superseded_at      TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS superseded_reason  TEXT,
  ADD COLUMN IF NOT EXISTS superseded_by_user TEXT;

-- 2. Add soft-forget columns. Mirror shape so audit queries are symmetric.
ALTER TABLE memories
  ADD COLUMN IF NOT EXISTS forgotten_at      TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS forgotten_reason  TEXT,
  ADD COLUMN IF NOT EXISTS forgotten_by_user TEXT;

-- 3. Self-FK for the supersede chain. ON DELETE SET NULL so a future hard-purge
-- of the head doesn't orphan the FK; the ancestor row keeps its history but
-- loses the dangling pointer.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'memories_superseded_by_fkey'
      AND conrelid = 'memories'::regclass
  ) THEN
    ALTER TABLE memories
    ADD CONSTRAINT memories_superseded_by_fkey
    FOREIGN KEY (superseded_by) REFERENCES memories(id) ON DELETE SET NULL;
  END IF;
END
$$;

-- 4. Sanity: a row cannot be both superseded and forgotten in the same op.
-- Either is allowed; both at once is not.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'memories_supersede_xor_forget'
      AND conrelid = 'memories'::regclass
  ) THEN
    ALTER TABLE memories
    ADD CONSTRAINT memories_supersede_xor_forget
    CHECK (NOT (superseded_at IS NOT NULL AND forgotten_at IS NOT NULL));
  END IF;
END
$$;

-- 5. Partial index on the active predicate. Every default `recall` filters with
-- `superseded_at IS NULL AND forgotten_at IS NULL`, so this is the hot path.
CREATE INDEX IF NOT EXISTS memories_active_idx
  ON memories (project_id)
  WHERE superseded_at IS NULL AND forgotten_at IS NULL;

-- 6. Reverse-lookup index for chain walks (`SELECT ... WHERE superseded_by = $1`).
CREATE INDEX IF NOT EXISTS memories_superseded_by_idx
  ON memories (superseded_by)
  WHERE superseded_by IS NOT NULL;
