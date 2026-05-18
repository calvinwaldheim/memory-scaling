# memory-scaling

A durable, project-scoped memory layer for AI agents, built on Databricks-native primitives. Agents write to it, read from it, correct it, and audit it — across sessions, across projects, across teams. The accuracy of any single agent compounds over time without retraining or fine-tuning.

Started as a single-session proof of concept on a Databricks free trial. Now a working multi-component system: notebooks for the original layered architecture, a reusable Python package (`memory_agent`), and an MCP server (`app/`) that exposes the whole memory store to any MCP-aware client (Claude Desktop, Claude Code, Cursor, custom agents).

Concept inspired by Databricks' [memory scaling research](https://www.databricks.com/blog/memory-scaling-ai-agents) (MemAlign, ALHF, Instructed Retriever).

---

## What's in this repo

Three layers, each usable on its own:

### 1. POC notebooks (`01_…` → `07_…`)

The original proof of concept. Seven notebooks that build the closed memory loop from scratch on Lakebase (serverless PostgreSQL + pgvector) and the Databricks Foundation Model API:

| Notebook | Purpose |
|---|---|
| [`01_schema_setup.ipynb`](01_schema_setup.ipynb) | Lakebase schema: `memories`, `retrieval_log`, `source_registry`; pgvector HNSW index |
| [`02_bootstrap.py`](02_bootstrap.py) | Chunks source documents (150 words, 20-word overlap), embeds via `databricks-gte-large-en`, writes as `episodic` |
| [`03_retrieval.py`](03_retrieval.py) | Cosine-distance similarity search over pgvector embeddings |
| [`04_agent.py`](04_agent.py) | Retrieve → generate with Llama 3.3 70B → write the Q+A back. The compounding loop. |
| [`05_distillation.py`](05_distillation.py) | Distills `episodic` memories into `semantic` generalizations via `AgglomerativeClustering` + LLM synthesis. Idempotent. Now integrated with `supersede` so re-clustering keeps lineage. |
| [`06_eval.py`](06_eval.py) | Memory-retention eval: teach novel facts, probe with paraphrased recall |
| [`07_scope_check.py`](07_scope_check.py) | Validates project containment and personal-scope filtering |

Findings, retrieval numbers, and the original architectural rationale are in [`memory-scaling-poc-report.md`](memory-scaling-poc-report.md).

### 2. `memory_agent` Python package

Reusable library that distills the notebook behavior into typed Python. Import it directly from any Databricks notebook, job, or Python service that talks to the same Lakebase instance.

| Module | What it does |
|---|---|
| [`memory_agent/storage.py`](memory_agent/storage.py) | All Lakebase I/O: parameterized SQL, connection management, the supersede/forget/lineage primitives |
| [`memory_agent/agent.py`](memory_agent/agent.py) | `retrieve()` and `answer()` — the read-then-generate-then-write loop |
| [`memory_agent/embeddings.py`](memory_agent/embeddings.py) | `embed_text` / `embed_texts` wrappers around the Foundation Model API |
| [`memory_agent/llm.py`](memory_agent/llm.py) | LLM call helper (Llama 3.3 70B) |
| [`memory_agent/chunking.py`](memory_agent/chunking.py) | 150-word chunking with overlap |
| [`memory_agent/config.py`](memory_agent/config.py) | Secret-backed config (env vars override) |

The notebook-to-module mapping and the original refactor rationale are in [`MIGRATION.md`](MIGRATION.md).

### 3. MCP app (`app/`)

A Databricks App that wraps `memory_agent` in an MCP server. Connect Claude Desktop, Claude Code, Cursor, or any MCP client and you get the entire memory store as tools. Full tool surface, deployment notes, and auth details in [`app/README.md`](app/README.md).

For agent operators: read [`AGENT.md`](AGENT.md) — that's the operating manual we want Claude to read when it connects to this MCP server. When-to-use heuristics, anti-patterns, the mental model.

---

## Core concepts

### Memory types

- **Episodic** — raw observations and interactions (Q+A pairs, single events, source-doc chunks). Write these by default.
- **Semantic** — distilled generalizations. Produced by the distillation pipeline from clusters of episodics. Higher quality, broader recall.

### Projects

Every memory belongs to exactly one project, identified by a slug (`memory-kb-poc`, `trackunit-customer`). Projects are registered in a `projects` table with a `project_type` enum (`data_domain` / `engineering` / `compliance` / `customer` / `product`). The MCP app supports cross-project recall and per-conversation active-project state.

### Lineage (the supersede model)

When a memory turns out to be wrong, you don't `DELETE` it. You `supersede` it:

1. The new (correct) memory gets inserted.
2. The old memory's `superseded_by` is set to the new id, `superseded_at = NOW()`, plus `superseded_reason` and `superseded_by_user`.
3. Both rows happen atomically in one transaction.
4. Default `recall` filters out the old one — wrong content never resurfaces in normal use.
5. The chain is queryable via `get_lineage(id)` for audit ("what did we believe before, when did we change our minds, who changed them").

Soft-forget is the parallel for retractions without a replacement. The row stays in the table with `forgotten_at` set; default recall ignores it.

Both behaviors land in migration 003 ([`migrations/003_supersede_lineage.sql`](migrations/003_supersede_lineage.sql)).

### Security model

Three layers, each enforced server-side:

**1. Authentication.** The MCP server runs a `DatabricksTokenVerifier` that calls `/api/2.0/preview/scim/v2/Me` on every request and extracts the verified user identity. There is no anonymous mode in production transports — a request without a valid Databricks workspace token is rejected.

**2. Per-project authorization.** The `project_acl` table maps `(project_id, user_name) → role`, with three roles: `viewer`, `contributor`, `owner`. Every tool checks the caller's role at the top of the call. Privacy is project-level: **a project is private by default**, only the creator has access until they explicitly `grant_access` to others. There's no row-level "personal" flag.

| Action | Required role |
|---|---|
| `recall`, `stats`, `list_hot`, `get_lineage`, `get_audit_log`, `list_access` | viewer |
| `remember`, `supersede`, `update_memory`, soft `forget` | contributor |
| `forget(hard=True)`, `archive_project`, `grant_access`, `revoke_access` | owner |

**3. Identity-bound audit.** Every memory carries `created_by`, populated from the verified token at write time — callers can't forge attribution. Every mutation (create / supersede / forget / update / purge) also writes a row to `memory_audit_log` inside the same transaction, recording the actor, reason, and before/after JSON state. The log is the action stream of record and survives even hard-deletes.

The ACL and audit-log shipped in migrations 004 ([`migrations/004_project_acl.sql`](migrations/004_project_acl.sql)) and 005 ([`migrations/005_memory_audit_log.sql`](migrations/005_memory_audit_log.sql)). See [`AGENT.md`](AGENT.md) for the agent-facing access model and [`app/README.md`](app/README.md) for the per-tool role gates.

---

## Running it

### Prerequisites

- A Databricks workspace (free trial works)
- A Lakebase project (autoscaling serverless Postgres) with `pgvector` enabled
- Foundation Model API access: `databricks-gte-large-en` (embeddings, 1024-dim) and `databricks-meta-llama-3-3-70b-instruct` (generation)
- Cluster libs: `psycopg2-binary`, `mlflow`. `scikit-learn` ships with the ML runtime.

### First-time setup

1. Clone this repo into your Databricks workspace as a Git folder.
2. Run [`01_schema_setup.ipynb`](01_schema_setup.ipynb) to create tables and the HNSW index.
3. Apply each migration in [`migrations/`](migrations/) in numerical order:
   - `001_schema_hardening.sql` — required for deduplication
   - `002_projects.sql` — required for multi-project support
   - `003_supersede_lineage.sql` — required for supersede/forget
   - `004_project_acl.sql` — required for per-project access control; drops the legacy `scope` column and renames `user_id` → `created_by`
   - `005_memory_audit_log.sql` — required for the action-stream audit log
4. Drop source documents in your workspace and run [`02_bootstrap.py`](02_bootstrap.py) to seed episodic memory.
5. Run [`03_retrieval.py`](03_retrieval.py) and [`04_agent.py`](04_agent.py) to exercise read and write paths.

### Day-to-day

Three ways to interact with the store, depending on the use case:

**a) Notebooks** — exploratory, single-shot, when you want to see what's happening at the SQL level. Run the relevant `0X_*.py` notebook.

**b) Python package** — for jobs, services, or notebooks that need the same I/O without re-implementing it.

```python
from memory_agent import storage, agent

memories = agent.retrieve("what's the retrieval threshold", project_id="memory-kb-poc")
storage.supersede_memory(
    old_id="<id>",
    new_content="threshold is 0.50",
    embedding=embeddings.embed_text("threshold is 0.50"),
    reason="re-tuned 2026-05",
)
```

**c) MCP app** — for any agent (Claude Desktop, Claude Code, Cursor, etc.). Connect to the deployed MCP server and you get `recall`, `remember`, `supersede`, `forget`, `update_memory`, `get_lineage`, `stats`, `list_hot`, plus the project-management tools. See [`app/README.md`](app/README.md) for deploy + connect instructions and [`AGENT.md`](AGENT.md) for the operating manual.

### Distillation

Run [`05_distillation.py`](05_distillation.py) manually or as a scheduled Databricks Job to compress accumulated episodics into semantics. Two modes:

- **Append** (`REDISTILL_ALL=False`, the default): only undistilled episodics enter clustering. Cheap, idempotent, nightly-friendly.
- **Refresh** (`REDISTILL_ALL=True`): every live episodic is re-clustered. Newly synthesized semantics whose `derived_from` is a **strict superset** of an existing semantic's automatically `supersede` the old one, leaving a versioned chain. Run periodically when the corpus has grown enough that prior generalizations are stale.

### Evaluation

[`06_eval.py`](06_eval.py) teaches novel facts via the agent loop and probes with paraphrased recall. [`07_scope_check.py`](07_scope_check.py) validates project isolation. Both are notebooks — run them when you want a confidence check.

---

## Status

Working past the POC stage. Multi-project support, supersede/lineage, soft-forget, and the MCP app are all in production-shape (tests pass, migrations applied, validated end-to-end against live Lakebase). Distillation is integrated with supersede.

What's still open:
- A scheduled pruning job that hard-deletes rows past `superseded_at` / `forgotten_at` retention windows.
- Many-to-one supersede primitive (distillation falls back to fresh INSERT today when a new semantic generalizes multiple old ones).
- Staleness detection via Delta Change Data Feed against `source_registry`.
- Unity Catalog governance (row-level security per project).
- Cross-project synthesis (Layer 2 in the original concept).

Original concept-level architecture: [`concept.txt`](concept.txt). POC technical report: [`memory-scaling-poc-report.md`](memory-scaling-poc-report.md).

## License

MIT.
