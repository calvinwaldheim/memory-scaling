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
# MAGIC ### Supersede integration (migration 003)
# MAGIC
# MAGIC When `REDISTILL_ALL=True`, the fetch step drops the `NOT EXISTS` filter and
# MAGIC pulls every episodic in the project. The write step then detects when a
# MAGIC newly synthesized semantic strictly generalizes an existing live semantic
# MAGIC (its `derived_from` is a superset of an existing semantic's) and calls
# MAGIC `storage.supersede_memory` instead of inserting fresh. The old semantic is
# MAGIC kept in the table for audit with `superseded_by` pointing at the new one,
# MAGIC and default `recall` automatically filters it out.
# MAGIC
# MAGIC Schema reference (from `01_schema_setup`): `memories.id` is `UUID` with
# MAGIC `DEFAULT gen_random_uuid()`; this notebook adds `derived_from UUID[]` on
# MAGIC first run. Migration 003 adds the `superseded_by` / `superseded_at` /
# MAGIC `superseded_reason` / `superseded_by_user` columns plus the forget triad.

# COMMAND ----------

# MAGIC %pip install psycopg2-binary scikit-learn mlflow --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Lakebase credentials
import uuid
from urllib.parse import quote, urlparse, urlunparse

import requests
from databricks.sdk import WorkspaceClient

workspace = WorkspaceClient()

# The Lakebase OAuth token issued below is scoped to the Job's runtime identity
# (the "Run as" principal). The Postgres user in the connection string must
# match that identity — otherwise Lakebase rejects the connection with
# "OAuth: User is not authorized". We take host/port/db from the secret-stored
# URI (same source memory_agent.storage._get_connection uses) and substitute
# the live runtime user.
secret_uri = workspace.dbutils.secrets.get(scope="memory-scaling", key="lakebase_uri")
runtime_user = workspace.current_user.me().user_name
_parsed = urlparse(secret_uri)
_host_port = _parsed.hostname + (f":{_parsed.port}" if _parsed.port else "")
CONN_STRING = urlunparse(
    _parsed._replace(netloc=f"{quote(runtime_user, safe='')}@{_host_port}")
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
print(f"Connecting to Lakebase as {runtime_user}")

# COMMAND ----------

# DBTITLE 1,Config notes
# MAGIC %md
# MAGIC ## Config — all tuning knobs live here
# MAGIC
# MAGIC `CONN_STRING` and `TOKEN` are created in the inline Lakebase credential cell above using a fresh runtime postgres credential.

# COMMAND ----------

import hashlib
import json
import sys
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
SEMANTIC_SOURCE_REF = "distilled-v1"
SEMANTIC_QUALITY_SCORE = 0.85       # slightly higher than episodic (LLM-vetted generalization)

# --- Foundation Model endpoints --------------------------------------------
EMBED_ENDPOINT = "databricks-gte-large-en"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

# --- Distillation knobs ----------------------------------------------------
DISTANCE_THRESHOLD = 0.4    # was 0.25 — observed pairwise distances among related chunks land at 0.33–0.44
MIN_CLUSTER_SIZE = 2        # singletons are skipped (nothing to generalize from)
MAX_CHUNKS_PER_PROMPT = 10  # cap on episodic examples sent to the LLM per cluster
SYNTHESIS_TEMPERATURE = 0.2

# --- Supersede integration -------------------------------------------------
# When True, FETCH_SQL drops the NOT EXISTS filter and pulls every episodic in
# the project (not just the undistilled ones). Re-clustering then re-evaluates
# existing semantics; any newly produced semantic that strictly generalizes a
# live one (its derived_from is a superset of the live one's) supersedes it.
# Default False keeps nightly runs cheap and append-only.
REDISTILL_ALL = False
SUPERSEDE_REASON = "distillation re-cluster produced a strict generalization"

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

FETCH_UNDISTILLED_SQL = """
SELECT m.id,
       m.context,
       m.domain,
       m.embedding::text AS embedding_text,
       m.created_at
FROM memories m
WHERE m.memory_type = 'episodic'
  AND m.project_id = %s
  AND m.forgotten_at IS NULL
  AND NOT EXISTS (
        SELECT 1
        FROM memories s
        WHERE s.memory_type = 'semantic'
          AND s.project_id = m.project_id
          AND s.superseded_at IS NULL
          AND s.forgotten_at IS NULL
          AND m.id = ANY(s.derived_from)
      )
ORDER BY m.created_at ASC;
"""

# REDISTILL_ALL mode: pull every live episodic regardless of prior distillation
# status. The write path will supersede stale semantics whose derived_from is a
# subset of any newly-formed cluster.
FETCH_ALL_SQL = """
SELECT m.id,
       m.context,
       m.domain,
       m.embedding::text AS embedding_text,
       m.created_at
FROM memories m
WHERE m.memory_type = 'episodic'
  AND m.project_id = %s
  AND m.forgotten_at IS NULL
ORDER BY m.created_at ASC;
"""

fetch_sql = FETCH_ALL_SQL if REDISTILL_ALL else FETCH_UNDISTILLED_SQL
with psycopg2.connect(CONN_STRING, password=TOKEN) as conn, \
     conn.cursor(cursor_factory=RealDictCursor) as cur:
    cur.execute(fetch_sql, (PROJECT_ID,))
    rows = cur.fetchall()

df = pd.DataFrame(rows)
if df.empty:
    # Nightly schedule will hit this whenever no new episodics have been
    # written since the last distillation pass — that's the expected
    # steady state, not a failure. Exit cleanly so the Job shows SUCCESS.
    print(f"No undistilled episodic memories for project '{PROJECT_ID}'. Nothing to distill.")
    dbutils.notebook.exit("no-op: no undistilled episodic memories")
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
    print(f"Only {len(df)} undistilled episodic memory for project '{PROJECT_ID}' — need ≥2 to cluster. Exiting cleanly.")
    dbutils.notebook.exit("no-op: fewer than 2 undistilled episodic memories")

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
        "\nNo semantic memories produced this run — write/validate cells will no-op.\n"
        f"  Episodic memories pulled:            {len(df)}\n"
        f"  Clusters formed:                     {sizes.shape[0]}\n"
        f"  Distillable clusters (size ≥ {MIN_CLUSTER_SIZE}):      {distillable}\n"
        f"  Singletons (skipped pre-LLM):        {singletons}\n"
        f"  Multi-member clusters SKIP'd by LLM: {llm_skipped}"
    )
else:
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
# MAGIC Two write paths:
# MAGIC - **Fresh INSERT** when the new semantic's `derived_from` doesn't fully cover
# MAGIC   any existing live semantic in this project (the default case for nightly
# MAGIC   appends and the only path when `REDISTILL_ALL=False`).
# MAGIC - **Supersede** when the new semantic's `derived_from` is a strict superset
# MAGIC   of one existing live semantic's `derived_from`. The old semantic stays in
# MAGIC   the table with `superseded_by` pointing at the new one. Default `recall`
# MAGIC   excludes it; `get_lineage` can walk the chain.
# MAGIC
# MAGIC When a new semantic's `derived_from` is a strict superset of **multiple**
# MAGIC live semantics, we fall back to fresh INSERT and log a warning — the v1
# MAGIC supersede primitive is 1-to-1, and multi-target consolidation is something
# MAGIC an operator should review (or a follow-up `supersede_many` primitive).

# COMMAND ----------

INSERT_SQL = """
INSERT INTO memories
    (project_id, project_type, memory_type, domain,
     rule, context, source_ref, content_hash, embedding,
     quality_score, derived_from, created_by)
VALUES %s
ON CONFLICT DO NOTHING
RETURNING id;
"""

# Pull every live semantic in the project with its derived_from set so we can
# detect strict-superset overlap before deciding INSERT vs. supersede.
LIVE_SEMANTICS_SQL = """
SELECT id, derived_from
FROM memories
WHERE project_id = %s
  AND memory_type = 'semantic'
  AND superseded_at IS NULL
  AND forgotten_at IS NULL
  AND derived_from IS NOT NULL
"""

sys.path.insert(0, "/Workspace/Users/calvin.waldheim@gmail.com/memory-scaling")  # noqa: E402
from memory_agent import storage as _storage  # noqa: E402

with psycopg2.connect(CONN_STRING, password=TOKEN) as conn, conn.cursor() as cur:
    cur.execute(LIVE_SEMANTICS_SQL, (PROJECT_ID,))
    live_semantics = [(str(row[0]), {str(x) for x in (row[1] or [])}) for row in cur.fetchall()]


def find_supersede_targets(new_derived_from: list[str], live: list[tuple[str, set]]) -> list[str]:
    """Return ids of live semantics whose derived_from is a STRICT subset of the new one."""
    new_set = set(new_derived_from)
    return [old_id for old_id, old_set in live if old_set and old_set < new_set]


supersede_results: list[dict] = []
fresh_inserts: list[tuple] = []
fresh_inserts_meta: list[dict] = []

for _, row in semantic_df.iterrows():
    vec = embed(row["context"])
    content_hash = hashlib.md5(row["context"].encode()).hexdigest()
    new_derived_from = [str(x) for x in row["derived_from"]]

    targets = find_supersede_targets(new_derived_from, live_semantics)

    if len(targets) == 1:
        old_id = targets[0]
        # supersede_memory uses its own _get_connection — relies on env vars
        # LAKEBASE_URI and LAKEBASE_PROJECT_NAME (Notebook config or job env).
        result = _storage.supersede_memory(
            old_id=old_id,
            new_content=row["context"],
            embedding=vec,
            reason=SUPERSEDE_REASON,
            user_id="distillation",
            rule=row["context"][:100],
            quality_score=SEMANTIC_QUALITY_SCORE,
            source_ref=SEMANTIC_SOURCE_REF,
            memory_type="semantic",
            domain=row["domain"],
            derived_from=new_derived_from,
        )
        supersede_results.append({"cluster_id": int(row["cluster_id"]), "old_id": old_id, **result})
        if result["status"] == "superseded":
            print(f"Cluster {row['cluster_id']} → SUPERSEDE  old={old_id[:8]}…  new={result['new_id'][:8]}…")
        else:
            print(f"Cluster {row['cluster_id']} → supersede {result['status']!r}, will not retry as INSERT")
    elif len(targets) >= 2:
        print(
            f"Cluster {row['cluster_id']} matches {len(targets)} live semantics "
            f"(strict-superset overlap with {[t[:8] for t in targets]}). "
            "Falling back to fresh INSERT — operator should review."
        )
        fresh_inserts.append((
            PROJECT_ID, PROJECT_TYPE, "semantic", row["domain"],
            row["context"][:100], row["context"], SEMANTIC_SOURCE_REF, content_hash,
            json.dumps(vec), SEMANTIC_QUALITY_SCORE, new_derived_from, "distillation",
        ))
        fresh_inserts_meta.append({"cluster_id": int(row["cluster_id"]), "multi_target": targets})
    else:
        fresh_inserts.append((
            PROJECT_ID, PROJECT_TYPE, "semantic", row["domain"],
            row["context"][:100], row["context"], SEMANTIC_SOURCE_REF, content_hash,
            json.dumps(vec), SEMANTIC_QUALITY_SCORE, new_derived_from, "distillation",
        ))
        fresh_inserts_meta.append({"cluster_id": int(row["cluster_id"])})

new_ids: list = []
if fresh_inserts:
    with psycopg2.connect(CONN_STRING, password=TOKEN) as conn, conn.cursor() as cur:
        execute_values(
            cur,
            INSERT_SQL,
            fresh_inserts,
            template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::uuid[], %s)",
        )
        new_ids = [r[0] for r in cur.fetchall()]
        conn.commit()

print(
    f"\nDistillation write summary:\n"
    f"  Fresh inserts:        {len(new_ids)} (cluster ids: {[m['cluster_id'] for m in fresh_inserts_meta]})\n"
    f"  Superseded:           {sum(1 for r in supersede_results if r['status'] == 'superseded')}\n"
    f"  Supersede no-ops:     {sum(1 for r in supersede_results if r['status'] != 'superseded')}"
)

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
