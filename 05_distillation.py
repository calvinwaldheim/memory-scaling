# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Distillation Pipeline
# MAGIC
# MAGIC Converts **episodic → semantic memories** via cluster + summarize.
# MAGIC
# MAGIC 1. Fetch every `memory_type='episodic'` row for the configured project that hasn't been distilled yet
# MAGIC 2. Cluster their embeddings (cosine, agglomerative, distance-threshold — no fixed k)
# MAGIC 3. For each cluster ≥ `MIN_CLUSTER_SIZE`, an LLM synthesizes one semantic statement
# MAGIC 4. Embed the synthesis, write back as `memory_type='semantic'` with `derived_from` provenance
# MAGIC 5. Validate retrieval surfaces semantic memories for general queries
# MAGIC
# MAGIC Idempotent: the fetch SQL excludes any episodic memory already referenced
# MAGIC in some semantic row's `derived_from` array, so re-running only processes
# MAGIC episodic memories created since the last distillation pass.
# MAGIC
# MAGIC Schema reference (from `01_schema_setup`): `memories.id` is `UUID` with
# MAGIC `DEFAULT gen_random_uuid()`; this notebook adds `derived_from UUID[]` on
# MAGIC first run.

# COMMAND ----------

# MAGIC %pip install psycopg2-binary scikit-learn mlflow --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Lakebase credentials
import uuid

import requests
from databricks.sdk import WorkspaceClient

workspace = WorkspaceClient()

# URI (user@host:port/db?sslmode=require) is sourced from the Databricks secret
# scope used by the rest of the repo — the canonical path that
# memory_agent.storage._get_connection() also uses. The runtime credential
# below supplies only the short-lived password (token).
CONN_STRING = workspace.dbutils.secrets.get(
    scope="memory-scaling",
    key="lakebase_uri",
)

credential = requests.post(
    f"{workspace.config.host.rstrip('/')}/api/2.0/postgres/credentials",
    headers={
        **workspace.config.authenticate(),
        "Content-Type": "application/json",
        "Accept": "application/json",
    },
    json={
        "endpoint": "projects/memory-kb-poc/branches/production/endpoints/primary",
        "request_id": str(uuid.uuid4()),
    },
    timeout=30,
)
credential.raise_for_status()
TOKEN = credential.json()["token"]
print("Connecting to Lakebase using URI from memory-scaling/lakebase_uri secret")

# COMMAND ----------

# DBTITLE 1,Config notes
# MAGIC %md
# MAGIC ## Config — all tuning knobs live here
# MAGIC
# MAGIC `CONN_STRING` and `TOKEN` are created in the inline Lakebase credential cell above using a fresh runtime postgres credential.

# COMMAND ----------

import hashlib
import json
from collections import Counter

import numpy as np
import pandas as pd
import psycopg2
from mlflow.deployments import get_deploy_client
from psycopg2.extras import RealDictCursor, execute_values
from sklearn.cluster import AgglomerativeClustering

# --- Project scope ---------------------------------------------------------
# Distillation runs per project; semantic memories only generalize across one
# project's episodic rows.
PROJECT_ID = "memory-kb-poc"
PROJECT_TYPE = "product"            # mirrors what 02_bootstrap and 04_agent write
SEMANTIC_SCOPE = "organizational"   # semantic memories are cross-cutting by definition
SEMANTIC_SOURCE_REF = "distilled-v1"
SEMANTIC_QUALITY_SCORE = 0.85       # slightly higher than episodic (LLM-vetted generalization)

# --- Foundation Model endpoints --------------------------------------------
EMBED_ENDPOINT = "databricks-gte-large-en"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

# --- Distillation knobs ----------------------------------------------------
DISTANCE_THRESHOLD = 0.25   # cosine distance: lower = tighter clusters, less aggregation
MIN_CLUSTER_SIZE = 2        # singletons are skipped (nothing to generalize from)
MAX_CHUNKS_PER_PROMPT = 10  # cap on episodic examples sent to the LLM per cluster
SYNTHESIS_TEMPERATURE = 0.2

client = get_deploy_client("databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. One-time additive migration: add `derived_from UUID[]`
# MAGIC
# MAGIC `ADD COLUMN IF NOT EXISTS` — safe to run on every invocation.

# COMMAND ----------

MIGRATE_SQL = """
ALTER TABLE memories
  ADD COLUMN IF NOT EXISTS derived_from UUID[] DEFAULT NULL;
"""

with psycopg2.connect(CONN_STRING, password=TOKEN) as conn, conn.cursor() as cur:
    cur.execute(MIGRATE_SQL)
    conn.commit()
print("derived_from column ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Fetch undistilled episodic memories
# MAGIC
# MAGIC Pulls `id`, `context` (full chunk text), `domain` (inherited to the
# MAGIC semantic memory later), and the embedding as text. The `NOT EXISTS`
# MAGIC clause excludes anything already referenced by a semantic row's
# MAGIC `derived_from` array — that's what makes re-runs cheap.

# COMMAND ----------

FETCH_SQL = """
SELECT m.id,
       m.context,
       m.domain,
       m.embedding::text AS embedding_text,
       m.created_at
FROM memories m
WHERE m.memory_type = 'episodic'
  AND m.project_id = %s
  AND NOT EXISTS (
        SELECT 1
        FROM memories s
        WHERE s.memory_type = 'semantic'
          AND s.project_id = m.project_id
          AND m.id = ANY(s.derived_from)
      )
ORDER BY m.created_at ASC;
"""

with psycopg2.connect(CONN_STRING, password=TOKEN) as conn, \
     conn.cursor(cursor_factory=RealDictCursor) as cur:
    cur.execute(FETCH_SQL, (PROJECT_ID,))
    rows = cur.fetchall()

df = pd.DataFrame(rows)
# pgvector returns "[0.1,0.2,...]" when cast to text
df["embedding"] = df["embedding_text"].apply(
    lambda s: np.array(json.loads(s), dtype=np.float32)
)
# psycopg2 returns UUIDs as uuid.UUID instances; stringify so we can round-trip
# them through execute_values without surprises.
df["id"] = df["id"].astype(str)

print(f"Pulled {len(df)} undistilled episodic memories for project '{PROJECT_ID}'")
df[["id", "domain", "context"]].head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Cluster embeddings
# MAGIC
# MAGIC `AgglomerativeClustering` with cosine distance and a `distance_threshold` —
# MAGIC no fixed `k`. The threshold is the lever:
# MAGIC
# MAGIC - too low → everything is its own singleton, nothing to distill
# MAGIC - too high → unrelated memories get fused, syntheses become vague
# MAGIC
# MAGIC Start at 0.25 and inspect the largest clusters before synthesizing.

# COMMAND ----------

if len(df) < 2:
    raise SystemExit("Not enough undistilled episodic memories to cluster — bootstrap more first.")

X = np.stack(df["embedding"].values)

clusterer = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=DISTANCE_THRESHOLD,
    metric="cosine",
    linkage="average",
)
df["cluster"] = clusterer.fit_predict(X)

sizes = df["cluster"].value_counts().sort_values(ascending=False)
distillable = (sizes >= MIN_CLUSTER_SIZE).sum()
singletons = (sizes == 1).sum()

print(f"Total clusters:                       {sizes.shape[0]}")
print(f"  Distillable (size ≥ {MIN_CLUSTER_SIZE}):              {distillable}")
print(f"  Singletons (skipped this run):       {singletons}")
print("\nLargest clusters:")
sizes.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sanity check — peek at the largest cluster
# MAGIC
# MAGIC Do the grouped memories actually share a theme? If they look unrelated,
# MAGIC drop `DISTANCE_THRESHOLD` and re-run from section 2.

# COMMAND ----------

def show_cluster(cluster_id: int, max_chars: int = 200) -> None:
    members = df[df["cluster"] == cluster_id]
    print(f"=== Cluster {cluster_id}: {len(members)} members ===")
    for _, row in members.iterrows():
        snippet = row["context"][:max_chars].replace("\n", " ")
        suffix = "…" if len(row["context"]) > max_chars else ""
        print(f"  [{row['id'][:8]}…] ({row['domain']}) {snippet}{suffix}")

show_cluster(sizes.index[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Synthesize one semantic memory per cluster
# MAGIC
# MAGIC The synthesis prompt is the second main quality lever (the first is the
# MAGIC clustering threshold). The "drop names, dates, one-off details" rule is
# MAGIC what pushes the model from "summary of these chunks" toward
# MAGIC "generalizable statement."
# MAGIC
# MAGIC `embed()` mirrors the function in `03_retrieval` / `04_agent` exactly —
# MAGIC same endpoint, same list-wrapped input shape, same response unwrap.

# COMMAND ----------

def embed(text: str) -> list[float]:
    """Mirror of the embed() helper in 03_retrieval / 04_agent."""
    response = client.predict(
        endpoint=EMBED_ENDPOINT,
        inputs={"input": [text]},
    )
    return response["data"][0]["embedding"]


SYNTHESIS_PROMPT = """You are distilling raw episodic memories (interaction logs and source-document chunks) into a single semantic memory — a generalizable statement that captures the shared idea across these examples.

Rules:
- Produce ONE concise paragraph (3–6 sentences).
- State the generalization, not the specifics. Drop names, dates, and one-off details unless they're essential to the concept.
- If the examples don't share a coherent idea, respond with exactly: SKIP
- No preamble, no "Here is the synthesis", no headers. Just the paragraph.

Episodic memories to distill:
---
{chunks}
---"""


def synthesize(cluster_members: pd.DataFrame) -> str | None:
    """Return a semantic-memory paragraph for a cluster, or None if the LLM says SKIP."""
    chunks_text = "\n\n".join(
        f"({i + 1}) {row['context']}"
        for i, (_, row) in enumerate(
            cluster_members.head(MAX_CHUNKS_PER_PROMPT).iterrows()
        )
    )
    prompt = SYNTHESIS_PROMPT.format(chunks=chunks_text)
    response = client.predict(
        endpoint=LLM_ENDPOINT,
        inputs={
            "messages": [{"role": "user", "content": prompt}],
            "temperature": SYNTHESIS_TEMPERATURE,
            "max_tokens": 400,
        },
    )
    text = response["choices"][0]["message"]["content"].strip()
    return None if text.upper() == "SKIP" else text


def majority_domain(cluster_members: pd.DataFrame) -> str:
    """Pick the most common domain among cluster members; semantic memory inherits it."""
    domains = [d for d in cluster_members["domain"].tolist() if d]
    if not domains:
        return "general"
    return Counter(domains).most_common(1)[0][0]


semantic_rows = []
llm_skipped = 0
for cluster_id, size in sizes.items():
    if size < MIN_CLUSTER_SIZE:
        continue
    members = df[df["cluster"] == cluster_id]
    synthesis = synthesize(members)
    if synthesis is None:
        llm_skipped += 1
        print(f"Cluster {cluster_id} ({size} members): SKIP")
        continue
    semantic_rows.append({
        "cluster_id": int(cluster_id),
        "context": synthesis,
        "domain": majority_domain(members),
        "derived_from": members["id"].tolist(),  # list[str] of UUIDs
    })
    print(f"Cluster {cluster_id} ({size} members, domain={semantic_rows[-1]['domain']}): ✓")

semantic_df = pd.DataFrame(semantic_rows)
print(f"\nProduced {len(semantic_df)} semantic memories from {len(df)} episodic inputs")

if semantic_df.empty:
    print(
        "\nNo semantic memories produced — exiting before write/validate.\n"
        f"  Episodic memories pulled:            {len(df)}\n"
        f"  Clusters formed:                     {sizes.shape[0]}\n"
        f"  Distillable clusters (size ≥ {MIN_CLUSTER_SIZE}):      {distillable}\n"
        f"  Singletons (skipped pre-LLM):        {singletons}\n"
        f"  Multi-member clusters SKIP'd by LLM: {llm_skipped}"
    )
    raise SystemExit("No semantic memories produced this run.")

semantic_df[["cluster_id", "domain", "context"]].head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Eyeball before writing
# MAGIC
# MAGIC Stop here and read the syntheses. If they're vague, hallucinated, or
# MAGIC just re-statements of one source chunk, tighten the threshold (lower
# MAGIC `DISTANCE_THRESHOLD`) or sharpen the synthesis prompt and re-run.
# MAGIC Nothing has been written to Lakebase yet.

# COMMAND ----------

for _, row in semantic_df.iterrows():
    print(f"--- Cluster {row['cluster_id']} · domain={row['domain']} · {len(row['derived_from'])} sources ---")
    print(row["context"])
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Write semantic memories back to Lakebase
# MAGIC
# MAGIC Column list mirrors `02_bootstrap` exactly, plus the new `derived_from`
# MAGIC array. `embedding` goes in as `json.dumps(vec)` (pgvector accepts a
# MAGIC JSON-formatted array as text input). `ON CONFLICT DO NOTHING` mirrors
# MAGIC the bootstrap pattern.

# COMMAND ----------

INSERT_SQL = """
INSERT INTO memories
    (project_id, project_type, memory_type, scope, domain,
     rule, context, source_ref, content_hash, embedding,
     quality_score, derived_from)
VALUES %s
ON CONFLICT DO NOTHING
RETURNING id;
"""

values = []
for _, row in semantic_df.iterrows():
    vec = embed(row["context"])
    content_hash = hashlib.md5(row["context"].encode()).hexdigest()
    values.append((
        PROJECT_ID,
        PROJECT_TYPE,
        "semantic",
        SEMANTIC_SCOPE,
        row["domain"],
        row["context"][:100],       # first 100 chars as the "rule" summary
        row["context"],
        SEMANTIC_SOURCE_REF,
        content_hash,
        json.dumps(vec),            # pgvector accepts a JSON-formatted array as text input
        SEMANTIC_QUALITY_SCORE,
        row["derived_from"],        # list[str] of UUIDs → UUID[] via implicit cast
    ))

if values:
    with psycopg2.connect(CONN_STRING, password=TOKEN) as conn, conn.cursor() as cur:
        execute_values(
            cur,
            INSERT_SQL,
            values,
            template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::uuid[])",
        )
        new_ids = [r[0] for r in cur.fetchall()]
        conn.commit()
    print(f"Wrote {len(new_ids)} semantic memories. IDs: {new_ids}")
else:
    print("Nothing to write.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Validate — does retrieval surface semantic memories for general queries?
# MAGIC
# MAGIC Mirrors the retrieval pattern from `03_retrieval`. Expectation: a broad
# MAGIC question should now surface a `semantic` row at or near the top, with
# MAGIC `episodic` chunks providing supporting specifics underneath.

# COMMAND ----------

QUERY = "How does memory scaling work overall?"

q_vec = embed(QUERY)

VALIDATE_SQL = """
SELECT id,
       memory_type,
       domain,
       context,
       embedding <=> %s::vector AS distance
FROM memories
WHERE project_id = %s
ORDER BY distance ASC
LIMIT 5;
"""

with psycopg2.connect(CONN_STRING, password=TOKEN) as conn, \
     conn.cursor(cursor_factory=RealDictCursor) as cur:
    cur.execute(VALIDATE_SQL, (json.dumps(q_vec), PROJECT_ID))
    results = cur.fetchall()

print(f"Query: {QUERY}\n")
for r in results:
    print(f"[{r['memory_type']:8s}] d={r['distance']:.3f}  domain={r['domain']}  id={str(r['id'])[:8]}…")
    print(f"           {r['context'][:160]}...")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tuning notes
# MAGIC
# MAGIC | Knob | What it controls | Start | Adjust when |
# MAGIC |---|---|---|---|
# MAGIC | `DISTANCE_THRESHOLD` | Cluster tightness | 0.25 | Many singletons → raise; clusters mix topics → lower |
# MAGIC | `MIN_CLUSTER_SIZE` | Minimum members to synthesize | 2 | Lots of noisy 2-member clusters → bump to 3 |
# MAGIC | `MAX_CHUNKS_PER_PROMPT` | Examples per LLM call | 10 | Long content + hitting token limits → lower |
# MAGIC | `SYNTHESIS_TEMPERATURE` | Synthesis variability | 0.2 | Outputs feel too rigid → raise to 0.4 |
# MAGIC | `SYNTHESIS_PROMPT` | What "generalization" means | see cell | Outputs are too specific or too vague |
# MAGIC
# MAGIC ## Promoting to a scheduled Databricks Job
# MAGIC
# MAGIC Once outputs look clean for a couple of manual runs:
# MAGIC
# MAGIC 1. Workspace → **Workflows** → **Create Job**
# MAGIC 2. Task type: **Notebook**, point at this file in the `memory-scaling` repo
# MAGIC 3. Schedule: nightly, e.g. 02:00 UTC
# MAGIC 4. Cluster: a small single-node serverless cluster is enough — this notebook is I/O-bound on the API calls, not compute
# MAGIC 5. Notifications: send failures to your inbox
# MAGIC
# MAGIC The `NOT EXISTS` filter in section 1 is what keeps the Job idempotent —
# MAGIC only episodic memories created since the last run are eligible, so
# MAGIC re-running costs nothing if there's no new material.
