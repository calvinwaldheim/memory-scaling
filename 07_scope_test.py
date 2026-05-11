# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Scope & Project Containment Smoke Test
# MAGIC
# MAGIC Defensive validation before handoff. The schema in `01_schema_setup`
# MAGIC supports two access-control dimensions:
# MAGIC
# MAGIC 1. **`project_id`** — every memory belongs to one project. Queries must
# MAGIC    filter on this so that one project's memories never leak into another.
# MAGIC 2. **`scope` ∈ {'personal', 'organizational'}** + **`user_id`** —
# MAGIC    personal memories should only be visible to the user who created
# MAGIC    them; organizational memories are visible to anyone in the project.
# MAGIC
# MAGIC The POC exercised only `project_id='memory-kb-poc'` /
# MAGIC `scope='organizational'`. This notebook seeds a small set of test
# MAGIC memories in a second project and with mixed scopes, then runs the
# MAGIC queries an organizational-memory layer would use to verify that
# MAGIC filters behave correctly. Test data is fully cleaned up at the end.

# COMMAND ----------

# MAGIC %pip install psycopg2-binary mlflow --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run /Users/calvin.waldheim@gmail.com/lakebase_config

# COMMAND ----------

import hashlib
import json

import psycopg2
from mlflow.deployments import get_deploy_client
from psycopg2.extras import RealDictCursor

client = get_deploy_client("databricks")

# --- Test fixtures ---------------------------------------------------------
POC_PROJECT = "memory-kb-poc"
TEST_PROJECT = "scope-test-project"
USER_ALICE = "alice@example.com"
USER_BOB = "bob@example.com"
TEST_SOURCE_REF = "scope-test"   # everything we insert here is tagged for cleanup

EMBED_ENDPOINT = "databricks-gte-large-en"

# --- Helpers ---------------------------------------------------------------
def embed(text: str) -> list[float]:
    response = client.predict(endpoint=EMBED_ENDPOINT, inputs={"input": [text]})
    return response["data"][0]["embedding"]


def insert_memory(project_id, scope, user_id, content, domain="scope-test"):
    """Insert a single test memory and return its UUID."""
    e = embed(content)
    content_hash = hashlib.md5(content.encode()).hexdigest()
    with psycopg2.connect(CONN_STRING, password=TOKEN) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO memories
                (project_id, project_type, memory_type, scope, user_id, domain,
                 rule, context, source_ref, content_hash, embedding, quality_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            RETURNING id
            """,
            (
                project_id, "product", "episodic", scope, user_id, domain,
                content[:100], content, TEST_SOURCE_REF, content_hash,
                json.dumps(e), 0.9,
            ),
        )
        row = cur.fetchone()
        conn.commit()
    return row[0] if row else None


def cleanup_test_data():
    """Delete anything tagged source_ref='scope-test'. Safe to run anytime."""
    with psycopg2.connect(CONN_STRING, password=TOKEN) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM memories WHERE source_ref = %s", (TEST_SOURCE_REF,))
        deleted = cur.rowcount
        conn.commit()
    print(f"  Cleaned up {deleted} prior test rows.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 0 — clean any leftovers from prior runs

# COMMAND ----------

print("Pre-run cleanup:")
cleanup_test_data()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test A — multi-project containment
# MAGIC
# MAGIC Seed 3 memories under `scope-test-project`. Verify:
# MAGIC - Filtering by `project_id = 'memory-kb-poc'` returns **zero** test rows
# MAGIC - Filtering by `project_id = 'scope-test-project'` returns **exactly 3** test rows
# MAGIC
# MAGIC Failure of either condition means the project filter isn't actually
# MAGIC containing data, which would be a serious leak risk.

# COMMAND ----------

print("Test A — seeding 3 memories in scope-test-project...")
for i, content in enumerate([
    "TEST_PROJECT_FACT_1: The scope test project's mascot is a kangaroo.",
    "TEST_PROJECT_FACT_2: Builds for this project run on Tuesdays.",
    "TEST_PROJECT_FACT_3: The scope test project uses orange as its theme color.",
], 1):
    new_id = insert_memory(
        project_id=TEST_PROJECT, scope="organizational", user_id=None,
        content=content,
    )
    print(f"  Inserted {i}: id={new_id}")

# Query 1: from the POC project's perspective, should see zero test-project leakage
with psycopg2.connect(CONN_STRING, password=TOKEN) as conn, conn.cursor() as cur:
    cur.execute(
        "SELECT COUNT(*) FROM memories WHERE project_id = %s AND source_ref = %s",
        (POC_PROJECT, TEST_SOURCE_REF),
    )
    leakage_into_poc = cur.fetchone()[0]

# Query 2: from the test project's perspective, should see exactly 3
with psycopg2.connect(CONN_STRING, password=TOKEN) as conn, conn.cursor() as cur:
    cur.execute(
        "SELECT COUNT(*) FROM memories WHERE project_id = %s AND source_ref = %s",
        (TEST_PROJECT, TEST_SOURCE_REF),
    )
    contained_in_test = cur.fetchone()[0]

test_a_pass = (leakage_into_poc == 0) and (contained_in_test == 3)
print(f"\nResult — Test A (multi-project containment):")
print(f"  Leakage into memory-kb-poc:      {leakage_into_poc}    (expected 0)")
print(f"  Rows in scope-test-project:      {contained_in_test}    (expected 3)")
print(f"  → {'PASS' if test_a_pass else 'FAIL'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test B — personal scope filtering
# MAGIC
# MAGIC Seed three memories in `scope-test-project`:
# MAGIC - One **personal** memory for `alice@example.com`
# MAGIC - One **personal** memory for `bob@example.com`
# MAGIC - One **organizational** memory (no user_id)
# MAGIC
# MAGIC Then run three queries representing different caller identities:
# MAGIC
# MAGIC 1. **As Alice** — should see her personal + the org one: 2 rows
# MAGIC 2. **As Bob** — should see his personal + the org one: 2 rows
# MAGIC 3. **Anonymous (no user filter)** — should see only the org one: 1 row
# MAGIC
# MAGIC The query template encodes the recommended access pattern:
# MAGIC
# MAGIC ```sql
# MAGIC WHERE project_id = %s
# MAGIC   AND (
# MAGIC     scope = 'organizational'
# MAGIC     OR (scope = 'personal' AND user_id = %s)
# MAGIC   )
# MAGIC ```
# MAGIC
# MAGIC With `user_id=NULL` for anonymous callers, the `AND` short-circuits
# MAGIC and personal rows are excluded.

# COMMAND ----------

print("Test B — seeding scoped memories...")
insert_memory(
    project_id=TEST_PROJECT, scope="personal", user_id=USER_ALICE,
    content="ALICE_PERSONAL_NOTE: I prefer to start meetings at 10am.",
)
insert_memory(
    project_id=TEST_PROJECT, scope="personal", user_id=USER_BOB,
    content="BOB_PERSONAL_NOTE: I'm tracking a side project on cost monitoring.",
)
insert_memory(
    project_id=TEST_PROJECT, scope="organizational", user_id=None,
    content="ORG_SHARED_NOTE: All sprints kick off on the first Monday of the month.",
)
print("  Seeded.")

SCOPE_QUERY = """
SELECT context FROM memories
WHERE project_id = %s
  AND source_ref = %s
  AND (
        scope = 'organizational'
        OR (scope = 'personal' AND user_id = %s)
      )
"""

def query_as(caller_user_id):
    """Return rows visible to the given caller. None = anonymous."""
    with psycopg2.connect(CONN_STRING, password=TOKEN) as conn, \
         conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(SCOPE_QUERY, (TEST_PROJECT, TEST_SOURCE_REF, caller_user_id))
        return cur.fetchall()


alice_rows = query_as(USER_ALICE)
bob_rows = query_as(USER_BOB)
anon_rows = query_as(None)

# Compose expectations
alice_visible = {r["context"][:30] for r in alice_rows}
bob_visible = {r["context"][:30] for r in bob_rows}
anon_visible = {r["context"][:30] for r in anon_rows}

ORG_PREFIX = "ORG_SHARED_NOTE: All sprints"[:30]
ALICE_PREFIX = "ALICE_PERSONAL_NOTE: I prefer"[:30]
BOB_PREFIX = "BOB_PERSONAL_NOTE: I'm tracking"[:30]

alice_ok = (len(alice_rows) == 2) and (ALICE_PREFIX in alice_visible) and (ORG_PREFIX in alice_visible) and (BOB_PREFIX not in alice_visible)
bob_ok = (len(bob_rows) == 2) and (BOB_PREFIX in bob_visible) and (ORG_PREFIX in bob_visible) and (ALICE_PREFIX not in bob_visible)
anon_ok = (len(anon_rows) == 1) and (ORG_PREFIX in anon_visible) and (ALICE_PREFIX not in anon_visible) and (BOB_PREFIX not in anon_visible)

test_b_pass = alice_ok and bob_ok and anon_ok

print(f"\nResult — Test B (personal scope filtering):")
print(f"  As Alice:       {len(alice_rows)} rows  {'✓' if alice_ok else '✗'}   (expected: her note + org note, not Bob's)")
print(f"  As Bob:         {len(bob_rows)} rows  {'✓' if bob_ok else '✗'}   (expected: his note + org note, not Alice's)")
print(f"  Anonymous:      {len(anon_rows)} rows  {'✓' if anon_ok else '✗'}   (expected: org note only)")
print(f"  → {'PASS' if test_b_pass else 'FAIL'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 99 — clean up all test data

# COMMAND ----------

print("Post-run cleanup:")
cleanup_test_data()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 60)
print(" Scope & Project Containment — Summary")
print("=" * 60)
print(f"  Test A (multi-project containment): {'PASS' if test_a_pass else 'FAIL'}")
print(f"  Test B (personal scope filtering):  {'PASS' if test_b_pass else 'FAIL'}")
print(f"  Overall:                            {'PASS' if (test_a_pass and test_b_pass) else 'FAIL'}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recommended access pattern
# MAGIC
# MAGIC Once an internal expert wires this up to additional agents, the canonical
# MAGIC retrieval template should be:
# MAGIC
# MAGIC ```sql
# MAGIC SELECT ... FROM memories
# MAGIC WHERE project_id = :project
# MAGIC   AND (
# MAGIC         scope = 'organizational'
# MAGIC         OR (scope = 'personal' AND user_id = :caller_user_id)
# MAGIC       )
# MAGIC ORDER BY embedding <=> :query_vec ASC
# MAGIC LIMIT :top_k;
# MAGIC ```
# MAGIC
# MAGIC With `caller_user_id = NULL` for callers without an identified user
# MAGIC (e.g. a scheduled job), personal rows are correctly hidden.
# MAGIC
# MAGIC The existing `03_retrieval` / `04_agent` notebooks don't yet apply the
# MAGIC scope filter (they just filter by `project_id`). For the corporate
# MAGIC workspace, the retrieve() helper should be updated to use this template.
