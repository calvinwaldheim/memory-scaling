# Memory-Scaling Knowledge Base on Databricks
## Proof of Concept — Technical Report
*May 2026 · Calvin Waldheim*

---

## 1. Overview

This report documents a working proof of concept for a memory-scaling AI agent architecture built entirely on Databricks-native primitives. The POC validates the core thesis from the Databricks memory scaling research: that agent performance can compound over time as accumulated experience is stored and retrieved, reducing reasoning steps and improving answer quality without retraining the underlying model.

The POC was built on a Databricks free trial workspace in a single session, demonstrating that the foundational architecture is reachable without requiring corporate infrastructure access.

---

## 2. What Was Built

Six notebooks in a Git-backed Databricks workspace (`memory-scaling` repo), each responsible for one layer of the architecture:

| Notebook | Purpose |
|---|---|
| `01_schema_setup` | Creates the Lakebase (PostgreSQL) schema: `memories`, `retrieval_log`, and `source_registry` tables with pgvector extension. |
| `02_bootstrap` | Ingests source documents, chunks into 150-word segments, generates 1024-dim embeddings via Foundation Model API, stores as episodic memories. |
| `03_retrieval` | Tests semantic retrieval using cosine distance on pgvector embeddings. Validates natural language queries return relevant chunks. |
| `04_agent` | Wires retrieval into an LLM call (Llama 3.3 70B). Retrieves memories, injects as context, generates grounded answer, writes interaction back as new episodic memory. |
| `05_distillation` | Converts episodic memories into semantic memories by clustering embeddings and synthesizing one generalization per cluster. Idempotent — only undistilled memories enter each run. |
| `06_eval` | Memory retention eval. Teaches novel facts via the agent loop, then asks paraphrased recall questions and scores whether the agent surfaces the taught fact. Separates retrieval failures from generation failures. |

---

## 3. Architecture

### 3.1 Storage: Lakebase

A Lakebase autoscaling project (serverless PostgreSQL on Neon) serves as the memory store. Three tables:

- **`memories`** — core store for episodic and semantic memories, with `VECTOR(1024)` column for pgvector similarity search and GIN index for full-text search.
- **`retrieval_log`** — tracks which memories were retrieved per query (foundation for future quality analysis).
- **`source_registry`** — tracks source documents and versions (foundation for staleness detection).

### 3.2 Embedding

Embeddings generated via Databricks Foundation Model API endpoint `databricks-gte-large-en`, producing 1024-dimensional vectors. Chunks are embedded one at a time to avoid API batch limits discovered during the POC.

### 3.3 Retrieval

Retrieval uses pgvector cosine distance (`<=>` operator) to rank memories by semantic similarity. The query is embedded at retrieval time and compared against all stored memory embeddings. Top-k results (default: 3) are returned with distance scores and raw context.

### 3.4 Agent Loop

The agent follows a three-step loop on each call:

1. **Retrieve** — embed the question, fetch top 3 most similar memories.
2. **Generate** — pass retrieved memories as system context to Llama 3.3 70B, generate grounded answer.
3. **Store** — embed the Q+A pair, write back to Lakebase as a new episodic memory, creating the compounding feedback loop.

### 3.5 Distillation Pipeline

The distillation pipeline converts low-precision episodic memories into higher-precision semantic memories, so that the memory store improves in quality over time rather than only growing in size.

The approach is **cluster + summarize**, not LLM-judge-per-memory:

1. **Fetch** all undistilled episodic memories for the project. Idempotency is enforced via a `NOT EXISTS` clause that excludes any episodic memory already referenced in some semantic row's `derived_from` array.
2. **Cluster** their embeddings using scikit-learn's `AgglomerativeClustering` with cosine distance and a `distance_threshold` of 0.25 (no fixed `k`). The threshold is the primary tuning knob — too low produces only singletons; too high fuses unrelated topics.
3. **Synthesize** one semantic statement per cluster of two or more members via a Llama 3.3 70B call. The synthesis prompt explicitly tells the model to drop names, dates, and one-off details, or return `SKIP` if the cluster lacks a coherent idea. Singletons are skipped by design.
4. **Embed and write back** each synthesis as a new row with `memory_type='semantic'`, the inherited majority domain from cluster members, and a `derived_from UUID[]` array storing the provenance link to the source episodic memories. Schema migration is additive (`ADD COLUMN IF NOT EXISTS`).

The notebook is run manually during the POC stage. Once outputs stabilize, it promotes cleanly to a nightly Databricks Job — the idempotent fetch means re-running costs nothing if there's no new material.

---

## 4. Results

### 4.1 Retrieval Quality

Retrieval quality improved significantly when chunk size was reduced from 500 to 150 words. Smaller chunks have tighter semantic focus, producing lower cosine distances on relevant queries:

| Query | 500-word chunks | 150-word chunks |
|---|---|---|
| Episodic vs semantic memory difference? | 0.376 | **0.295** |
| Governance across projects? | 0.413 | **0.317** |
| Distillation pipeline? | 0.472 | 0.453 |

### 4.2 Memory Compounding

The write-back loop was validated by asking a follow-up question after an initial interaction. The retrieval log confirmed the agent retrieved its own previous answer as context — grounding responses in interaction history, not just the original source document.

Retrieved memories for a follow-up query (by distance):

| Distance | Source |
|---|---|
| 0.323 | Agent's own answer to the previous question (stored interaction) |
| 0.407 | Original Q+A pair from prior session |
| 0.424 | Chunk from source document |

### 4.3 Answer Quality

The agent produced accurate, specific answers grounded in retrieved context. The answer to *"How does memory scaling reduce reasoning steps?"* correctly cited the 20→5 reasoning steps figure from the source document — a detail that came from retrieved memory, not model training data.

### 4.4 Distillation: Validated

After running `05_distillation` against the bootstrapped episodic memories, semantic memories were verified to outrank episodic ones for general-flavor queries — the intended layered retrieval behavior.

For the query *"How does memory scaling work overall?"*, retrieval returned:

| Rank | Type | Distance | Source |
|---|---|---|---|
| 1 | **semantic** | **0.330** | Synthesized: "AI agents can improve performance through memory scaling, where accumulated experience and distilled lessons enhance accuracy and efficiency..." |
| 2 | episodic | 0.380 | Concept-doc chunk: "Memory-Scaling Knowledge Base on Databricks — A Practitioner's Architecture Guide..." |
| 3 | episodic | 0.399 | Concept-doc chunk: "deciding during reasoning that a memory query would help..." |
| 4 | episodic | 0.448 | Concept-doc chunk: "Every component maps to a Databricks primitive already available..." |
| 5 | **semantic** | **0.450** | Synthesized: "A unified memory system integrates structured queries, full-text search, and vector similarity search..." |

For a broad conceptual question, the synthesized generalization is now the top result, with raw source chunks providing supporting detail underneath. This is the layered behavior the architecture was designed to produce.

### 4.5 Memory Retention Eval

`06_eval` tests the end-to-end memory loop by teaching novel facts through the agent (`ask()`-style write-back) and then asking paraphrased questions whose answers can only come from memory — not from the bootstrapped source documents or LLM training data.

Five invented facts about the POC itself were taught and then probed with paraphrased recall questions. Scoring is keyword presence in the answer (case-insensitive).

| Result | Count | Detail |
|---|---|---|
| Full pass (score = 1.0) | 4 / 5 | Right memory retrieved, right terms in answer |
| Partial (score = 0.5) | 1 / 5 | Right memory retrieved, but answer didn't surface all expected terms |
| Fail (score = 0) | 0 / 5 | — |
| **Mean score** | **0.90** | Above the 70% pass target |

Retrieval succeeded on every fact — the correct taught memory appeared in the top 3 for every recall query, often at rank 1. The one partial failure (a numeric question about the production distillation threshold) was a generation problem, not a retrieval problem: the LLM had the right memory in its context but got tangled by multiple numeric values across chunks and pleaded ignorance rather than synthesizing. This separates a fixable generation issue (try a different model, loosen the system prompt, or rephrase the query) from a fundamental architectural one.

---

## 5. Infrastructure Used

| Component | Databricks Primitive |
|---|---|
| Memory store | Lakebase autoscaling project (serverless PostgreSQL / Neon) |
| Vector search | pgvector extension on Lakebase (HNSW index) |
| Embeddings | Foundation Model API — `databricks-gte-large-en` (1024 dims) |
| LLM | Foundation Model API — `databricks-meta-llama-3-3-70b-instruct` |
| Notebooks | Databricks Workspace, Git-backed (`memory-scaling` repo) |
| Client libraries | `psycopg2-binary`, `mlflow.deployments` |

---

## 6. What Is Not Yet Built

| Missing Component | Description |
|---|---|
| Consolidation job | Deduplication, pruning of unused memories, navigation artifact generation. |
| Staleness detection | Delta Lake Change Data Feed integration to flag memories linked to changed sources. |
| Multi-project scoping | Project-scoped access control, cross-project read grants, personal vs. organizational separation. |
| Unity Catalog governance | Row-level security, memory lineage tracking, GDPR-compliant purge. |
| Layer 2 systems | Cross-project pattern detection, compliance tracking, schema intelligence. |
| Auto-refresh of Lakebase OAuth | Tokens currently expire every ~1 hour and must be re-pasted into a `lakebase_config` notebook. Replace with a Databricks SDK call that fetches a fresh token on demand. |

---

## 7. Recommended Next Steps

In priority order:

1. **Corporate workspace access** — the free-trial workspace has done its job; the architecture is now proven end-to-end. Moving to a corporate Databricks workspace unlocks real organizational data sources and the governance work below.
2. **More source diversity** — bootstrap from Delta table schemas, internal wikis, dashboard queries. The current memory store is dominated by one concept document; real value comes from heterogeneous organizational sources.
3. **Staleness detection** — wire up Delta Lake Change Data Feed to flag memories when source tables change. Foundation already exists in the `source_registry` table.
4. **Unity Catalog governance** — implement project scoping as catalog schemas with row-level security.
5. **Promote distillation to a scheduled Job** — once the synthesis prompt and clustering threshold are stable, wrap `05_distillation` in a nightly Databricks Job. The notebook is already idempotent.
6. **Harden generation** — the eval surfaced one case where retrieval succeeded but the LLM didn't synthesize across multiple retrieved chunks. Worth testing with a stronger model on the same endpoint, or loosening the "context-only" system prompt.

---

*Architecture based on Databricks Memory Scaling research (MemAlign, ALHF, Instructed Retriever)*
