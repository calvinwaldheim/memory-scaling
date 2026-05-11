-- Schema hardening for memories.
-- This migration aligns the live Lakebase table with application expectations:
-- 1) report duplicate (project_id, content_hash) groups before enforcing uniqueness,
-- 2) require created_at for deterministic retention ordering,
-- 3) enforce database-level deduplication with a unique constraint, and
-- 4) add a cosine HNSW index to support the <=> retrieval operator efficiently.

-- Surface duplicate keys for manual review before any destructive cleanup.
SELECT project_id, content_hash, COUNT(*) AS dup_count
FROM memories
GROUP BY project_id, content_hash
HAVING COUNT(*) > 1
ORDER BY dup_count DESC;

-- Optional cleanup if the duplicate report above is non-empty.
-- Tradeoff: this preserves the oldest row per (project_id, content_hash)
-- using created_at ASC and id ASC as the tie-breaker, but it permanently
-- deletes newer duplicates and any metadata attached to them.
-- Uncomment only after reviewing the duplicate report.
-- WITH ranked AS (
--   SELECT
--     id,
--     ROW_NUMBER() OVER (
--       PARTITION BY project_id, content_hash
--       ORDER BY created_at ASC, id ASC
--     ) AS rn
--   FROM memories
-- )
-- DELETE FROM memories AS m
-- USING ranked AS r
-- WHERE m.id = r.id
--   AND r.rn > 1;

-- Safety pre-check: fail loudly if any existing row still has NULL created_at.
DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM memories
    WHERE created_at IS NULL
  ) THEN
    RAISE EXCEPTION 'Cannot set memories.created_at NOT NULL: NULL values exist';
  END IF;
END
$$;

-- Ensure created_at has the expected default for new rows.
ALTER TABLE memories
ALTER COLUMN created_at SET DEFAULT NOW();

-- Require created_at so retention and duplicate cleanup ordering are stable.
ALTER TABLE memories
ALTER COLUMN created_at SET NOT NULL;

-- Add the project/content uniqueness rule only if it is not already present.
-- This intentionally fails if duplicates remain after optional cleanup.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'memories_project_content_unique'
      AND conrelid = 'memories'::regclass
  ) THEN
    ALTER TABLE memories
    ADD CONSTRAINT memories_project_content_unique
    UNIQUE (project_id, content_hash);
  END IF;
END
$$;

-- Add a cosine HNSW index for pgvector similarity search if one is not
-- already present on the embedding column.
CREATE INDEX IF NOT EXISTS memories_embedding_hnsw_idx
ON memories USING hnsw (embedding vector_cosine_ops);
