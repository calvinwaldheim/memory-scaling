# memory-scaling

A proof-of-concept implementation of a memory-scaling AI agent built entirely on Databricks-native primitives. The agent's accuracy compounds over time as accumulated experience is stored and retrieved — no retraining, no fine-tuning. Built end-to-end on a Databricks free-trial workspace in a single session.

Inspired by Databricks' [memory scaling research](https://www.databricks.com/blog/memory-scaling-ai-agents) (MemAlign, ALHF, Instructed Retriever).

## What it does

Seven notebooks form a closed memory loop on Lakebase (serverless PostgreSQL + pgvector) and the Databricks Foundation Model API.

| Notebook | Layer |
|---|---|
| `01_schema_setup.ipynb` | Lakebase schema: `memories`, `retrieval_log`, `source_registry` with `VECTOR(1024)` columns and pgvector HNSW indexes |
| `02_bootstrap.ipynb` | Chunks source documents into 150-word segments, embeds via `databricks-gte-large-en`, writes to `memories` as episodic |
| `03_retrieval.ipynb` | Cosine-distance similarity search over pgvector embeddings |
| `04_agent.ipynb` | Agent loop: retrieve → generate with Llama 3.3 70B → write the Q+A back as a new episodic memory |
| `05_distillation.py` | Distills episodic memories into semantic generalizations via `AgglomerativeClustering` (cosine, `distance_threshold=0.25`) + LLM synthesis. Idempotent. |
| `06_eval.py` | Memory-retention eval: teach novel facts through the agent, probe with paraphrased recall |
| `07_scope_check.py` | Validates project containment and personal-scope filtering against `project_id` |

## Key findings

- **Memory compounding works.** On a follow-up question, the agent retrieved its own previous answer at rank 1 (cosine distance 0.323), ahead of both the original Q+A pair and the source-document chunk it was first derived from.
- **Distillation produces the intended layered retrieval.** For broad conceptual queries, synthesized semantic memories now outrank raw document chunks; for specific lookups, the underlying episodic chunks still surface underneath.
- **Memory-retention eval: mean score 0.90** across five paraphrased recall probes against facts taught only at runtime. The single sub-1.0 score was a generation failure (LLM didn't synthesize numeric values across retrieved chunks), not a retrieval failure — every taught memory appeared in the top-3 for its probe.
- **Chunk size matters.** Cosine distances dropped roughly 0.07–0.10 across the same retrieval queries when chunks were reduced from 500 → 150 words.

Full architectural rationale, the distillation design, the retrieval numbers, and the eval methodology are in [`memory-scaling-poc-report.md`](memory-scaling-poc-report.md). The concept-level architecture this implements is in [`concept.txt`](concept.txt).

## Running it

**Prerequisites**

- A Databricks workspace (free trial is sufficient — this POC was built on one)
- A Lakebase project (autoscaling, serverless PostgreSQL on Neon) with the `vector` extension enabled
- Foundation Model API access:
  - `databricks-gte-large-en` for embeddings (1024-dim)
  - `databricks-meta-llama-3-3-70b-instruct` for generation
- Cluster libs: `psycopg2-binary`, `mlflow`. `scikit-learn` ships with the Databricks ML runtime.

**Setup**

1. Clone this repo into your Databricks workspace as a Git folder.
2. Create a private notebook called `lakebase_config` in your workspace home that defines `CONN_STRING` (the Lakebase JDBC-style host string) and `TOKEN` (the OAuth token from the Lakebase UI). The other notebooks `%run` it. Keep this notebook out of git — it's in `.gitignore`.
3. Run `01_schema_setup` once to create tables and indexes.
4. Drop source documents into your workspace and run `02_bootstrap` to seed episodic memory.
5. Run `03_retrieval` and `04_agent` to exercise the read and write paths.
6. Run `05_distillation` whenever you want to compress accumulated episodic memories into semantic ones. It's idempotent — re-running with no new material is a no-op.
7. Run `06_eval` to score memory retention end-to-end. Run `07_scope_check` to validate project-scope isolation.

**A note on tokens.** The Lakebase OAuth token expires every ~1 hour and must be refreshed in `lakebase_config`. Replacing this with a Databricks SDK call that fetches a fresh token on demand is on the next-steps list.

## Status

Proof of concept, not production. Distillation and the retention eval are validated end-to-end. Consolidation, staleness detection via Delta CDF, Unity Catalog governance, and cross-project scope work are open — see §6 of the report.

## License

MIT.
