# Memory-Scaling Knowledge Base on Databricks
## Proof of Concept — Technical Report
*May 2026 · Calvin Waldheim*

---

## 1. Overview

This report documents a working proof of concept for a memory-scaling AI agent architecture built entirely on Databricks-native primitives. The POC validates the core thesis from the Databricks memory scaling research: that agent performance can compound over time as accumulated experience is stored and retrieved, reducing reasoning steps and improving answer quality without retraining the underlying model.

The POC was built on a Databricks free trial workspace in a single session, demonstrating that the foundational architecture is reachable without requiring corporate infrastructure access.

---

## 2. What Was Built

Four notebooks in a Git-backed Databricks workspace (`memory-scaling` repo), each responsible for one layer of the architecture:

| Notebook | Purpose |
|---|---|
| `01_schema_setup` | Creates the Lakebase (PostgreSQL) schema: `memories`, `retrieval_log`, and `source_registry` tables with pgvector extension. |
| `02_bootstrap` | Ingests source documents, chunks into 150-word segments, generates 1024-dim embeddings via Foundation Model API, stores as episodic memories. |
| `03_retrieval` | Tests semantic retrieval using cosine distance on pgvector embeddings. Validates natural language queries return relevant chunks. |
| `04_agent` | Wires retrieval into an LLM call (Llama 3.3 70B). Retrieves memories, injects as context, generates grounded answer, writes interaction back as new episodic memory. |

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
| Distillation pipeline | Episodic → semantic memory conversion via LLM judge + clustering. |
| Consolidation job | Deduplication, pruning of unused memories, navigation artifact generation. |
| Staleness detection | Delta Lake Change Data Feed integration to flag memories linked to changed sources. |
| Multi-project scoping | Project-scoped access control, cross-project read grants, personal vs. organizational separation. |
| Unity Catalog governance | Row-level security, memory lineage tracking, GDPR-compliant purge. |
| Layer 2 systems | Cross-project pattern detection, compliance tracking, schema intelligence. |

---

## 7. Recommended Next Steps

In priority order:

1. **Distillation pipeline** — nightly Databricks Job that converts episodic → semantic memories. Highest-value addition; makes the memory store improve in precision over time rather than just growing.
2. **More source diversity** — bootstrap from Delta table schemas, internal wikis, dashboard queries. Real value comes from heterogeneous organizational sources.
3. **Staleness detection** — wire up Delta Lake Change Data Feed to flag memories when source tables change.
4. **Corporate workspace access** — move from free trial to corporate Databricks workspace to test with real organizational data and validate governance requirements.
5. **Unity Catalog governance** — implement project scoping as catalog schemas with row-level security.

---

*Architecture based on Databricks Memory Scaling research (MemAlign, ALHF, Instructed Retriever)*
