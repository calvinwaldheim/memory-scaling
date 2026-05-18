-- Pass A · multi-user safety: project ACL + remove the personal-scope stub.
--
-- The previous design carried a `scope` column on memories (enum
-- 'personal'|'organizational') and a `user_id` column intended for personal-scope
-- isolation. None of it was ever enforced past the bootstrap default. We're
-- replacing the half-built concept with a real access boundary:
--
--   - Privacy belongs to the *project*, not to individual rows.
--   - `project_acl` defines who can read or write each project.
--   - "Not shared" = "private by default" — no row-level flag needed.
--
-- `user_id` is renamed to `created_by` to reflect what it actually means now:
-- the verified user (from the MCP server's token verifier) who wrote the row.

-- 1. Pre-check report — surface anything unusual before destructive ops.
--    Expected: every row is 'organizational', user_id is NULL.
SELECT scope AS current_scope_value, COUNT(*) AS row_count
FROM memories
GROUP BY scope
ORDER BY 2 DESC;

SELECT 'rows with non-NULL user_id (will become created_by)' AS note, COUNT(*) AS row_count
FROM memories
WHERE user_id IS NOT NULL;

-- 2. Drop the scope column and its CHECK. Safe per the pre-check.
ALTER TABLE memories DROP CONSTRAINT IF EXISTS memories_scope_check;
ALTER TABLE memories DROP COLUMN IF EXISTS scope;

-- 3. Rename user_id → created_by. Semantic shift: row author, not "this row is for user X".
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'memories' AND column_name = 'user_id'
  ) AND NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'memories' AND column_name = 'created_by'
  ) THEN
    ALTER TABLE memories RENAME COLUMN user_id TO created_by;
  END IF;
END
$$;

-- 4. Project ACL table. One row per (project, user) granting a role.
CREATE TABLE IF NOT EXISTS project_acl (
  project_id  TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
  user_name   TEXT NOT NULL,
  role        TEXT NOT NULL,
  granted_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  granted_by  TEXT NOT NULL,
  PRIMARY KEY (project_id, user_name),
  CONSTRAINT project_acl_role_check CHECK (role IN ('viewer','contributor','owner'))
);

-- Lookup index for the "what projects can this user see?" query.
CREATE INDEX IF NOT EXISTS project_acl_user_idx ON project_acl(user_name);

-- 5. Backfill — whoever runs this migration becomes owner of every live project.
--    They can then grant other users via the grant_access MCP tool.
INSERT INTO project_acl (project_id, user_name, role, granted_by)
SELECT p.project_id, current_user, 'owner', 'migration:004'
FROM projects p
WHERE p.archived_at IS NULL
ON CONFLICT (project_id, user_name) DO NOTHING;
