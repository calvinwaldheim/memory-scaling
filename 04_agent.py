# Databricks notebook source
# MAGIC %pip install -e /Workspace/Users/calvin.waldheim@gmail.com/memory-scaling

# COMMAND ----------

# DBTITLE 1,Load Lakebase credential
# MAGIC %run /Users/calvin.waldheim@gmail.com/lakebase_config

# COMMAND ----------

# DBTITLE 1,Run live verification
import hashlib

import memory_agent.storage as storage
from memory_agent.agent import answer, retrieve
from memory_agent.config import DEFAULT_PROJECT_ID

question = "What did we discuss about end to end implementation?"

cleanup_sql = """
WITH ranked AS (
    SELECT
        id,
        content_hash,
        ROW_NUMBER() OVER (
            PARTITION BY content_hash, project_id, source_ref
            ORDER BY created_at ASC, id ASC
        ) AS rn
    FROM memories
    WHERE project_id = %s
      AND source_ref = 'agent-interaction'
      AND rule = %s
)
DELETE FROM memories AS m
USING ranked AS r
WHERE m.id = r.id
  AND r.rn > 1
RETURNING m.id, m.content_hash
"""

question_rows_sql = """
SELECT id, content_hash, created_at, context
FROM memories
WHERE project_id = %s
  AND source_ref = 'agent-interaction'
  AND rule = %s
ORDER BY created_at ASC, id ASC
"""

with storage._get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(cleanup_sql, (DEFAULT_PROJECT_ID, question))
        deleted_rows = cur.fetchall()
        conn.commit()
        cur.execute(question_rows_sql, (DEFAULT_PROJECT_ID, question))
        preexisting_rows = cur.fetchall()

preexisting_hashes = {row[1] for row in preexisting_rows}
response = answer(question)
retrieved = retrieve(question)
content = f"Q: {question}\nA: {response}"
new_hash = hashlib.md5(content.encode()).hexdigest()
shape = ["fresh" if hashlib.md5(memory.context.encode()).hexdigest() == new_hash else "pre-existing" for memory in retrieved]

print({
    "cleanup_deleted_duplicates": len(deleted_rows),
    "preexisting_question_rows_after_cleanup": len(preexisting_rows),
    "new_content_hash_preexisted": new_hash in preexisting_hashes,
    "question": question,
    "retrieved_count": len(retrieved),
    "retrieval_shape": shape,
    "answer": response,
})
for i, memory in enumerate(retrieved, start=1):
    label = "fresh" if hashlib.md5(memory.context.encode()).hexdigest() == new_hash else "pre-existing"
    print(
        f"RETRIEVED {i} label={label} distance={memory.distance:.3f} "
        f"preview={memory.context[:200]!r}"
    )
