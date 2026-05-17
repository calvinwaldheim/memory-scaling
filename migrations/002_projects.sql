-- Multi-project support: introduce a registry for projects and link memories to it.
-- Before applying, the surface-state checks below should both return 0 rows.

-- Surface 1: any memories with a NULL project_id would orphan after the FK.
SELECT 'memories with NULL project_id' AS check, COUNT(*) AS rows
FROM memories
WHERE project_id IS NULL;

-- Surface 2: any project_id that already exists in memories but would not be
-- captured by the backfill below (none expected — backfill is DISTINCT).
-- This is informational; it's safe to apply even if non-zero.
SELECT 'distinct project_ids in memories' AS check, COUNT(DISTINCT project_id) AS rows
FROM memories;

-- 1. Create the projects registry.
CREATE TABLE IF NOT EXISTS projects (
  project_id   TEXT PRIMARY KEY,
  name         TEXT NOT NULL,
  project_type TEXT NOT NULL,
  description  TEXT,
  tags         TEXT[] NOT NULL DEFAULT '{}',
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  created_by   TEXT,
  archived_at  TIMESTAMPTZ,

  -- Slug rule: lower-case alphanumerics and dashes only.
  CONSTRAINT projects_id_slug_check
    CHECK (project_id ~ '^[a-z0-9][a-z0-9-]*$'),

  -- Enum from concept.txt. Strict — callers picking new categories must
  -- migrate this constraint, by design.
  CONSTRAINT projects_type_check
    CHECK (project_type IN ('data_domain','engineering','compliance','customer','product'))
);

-- 2. Backfill: every distinct project_id that already exists in memories gets a
-- minimal projects row, so the upcoming FK doesn't blow up. We infer the type
-- from the most-common project_type stamped on the memory rows.
INSERT INTO projects (project_id, name, project_type, description, created_by)
SELECT
  m.project_id,
  m.project_id AS name,  -- placeholder; user can rename later
  COALESCE(
    (SELECT project_type FROM memories
     WHERE project_id = m.project_id
       AND project_type IN ('data_domain','engineering','compliance','customer','product')
     GROUP BY project_type
     ORDER BY COUNT(*) DESC
     LIMIT 1),
    'product'
  ) AS project_type,
  'Backfilled from existing memories on ' || NOW()::DATE AS description,
  'migration:002' AS created_by
FROM (SELECT DISTINCT project_id FROM memories WHERE project_id IS NOT NULL) m
ON CONFLICT (project_id) DO NOTHING;

-- 3. Add the FK from memories to projects. This is the lock-in: future writes
-- against an unknown project_id will be rejected at the database layer.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'memories_project_id_fkey'
      AND conrelid = 'memories'::regclass
  ) THEN
    ALTER TABLE memories
    ADD CONSTRAINT memories_project_id_fkey
    FOREIGN KEY (project_id) REFERENCES projects(project_id);
  END IF;
END
$$;

-- 4. Helpful indexes for the new lookup patterns.
CREATE INDEX IF NOT EXISTS projects_archived_idx ON projects (archived_at)
  WHERE archived_at IS NULL;  -- partial index: most queries want active projects
CREATE INDEX IF NOT EXISTS projects_tags_gin_idx ON projects USING gin (tags);
