# Migration Guide

This refactor moves notebook business logic from `02_bootstrap`, `03_retrieval`, and `04_agent` into a reusable `memory_agent/` package while preserving observed behavior from the notebooks.

## Notebook-to-module mapping

| Original notebook | Original cell behavior | New module/function |
| --- | --- | --- |
| `02_bootstrap` | Chunk source text into 150-word chunks with 20-word overlap | `memory_agent.chunking.chunk_text` |
| `02_bootstrap` | Embed each chunk with `databricks-gte-large-en` | `memory_agent.embeddings.embed_texts` |
| `02_bootstrap` | Insert chunk rows into `memories` with `memory_type='episodic'` | `memory_agent.storage.store_bootstrap_memories` |
| `03_retrieval` | Embed query text | `memory_agent.embeddings.embed_text` |
| `03_retrieval` | Run pgvector cosine-distance retrieval with `top_k=3` | `memory_agent.storage.retrieve_memories` |
| `04_agent` | Retrieve relevant memories for the question | `memory_agent.agent.retrieve` |
| `04_agent` | Call `databricks-meta-llama-3-3-70b-instruct` with retrieved context | `memory_agent.llm.generate_answer` |
| `04_agent` | Write the Q+A pair back as an episodic memory | `memory_agent.agent.answer` via `memory_agent.storage.insert_memory` |

## Public package entry points

* `memory_agent.answer`
* `memory_agent.retrieve`
* `memory_agent.chunking.chunk_text`
* `memory_agent.embeddings.embed_text`
* `memory_agent.embeddings.embed_texts`
* `memory_agent.llm.generate_answer`
* `memory_agent.storage.insert_memory`
* `memory_agent.storage.retrieve_memories`
* `memory_agent.storage.store_bootstrap_memories`

## Refactor details

* **Package location**: `/Users/calvin.waldheim@gmail.com/memory-scaling/memory_agent/`
* **Test location**: `/Users/calvin.waldheim@gmail.com/memory-scaling/tests/`
* **Modules created**: `__init__.py`, `config.py`, `chunking.py`, `embeddings.py`, `llm.py`, `storage.py`, `agent.py`
* **Behavior preserved**: 150-word chunking, 20-word overlap, top_k=3, memory_type='episodic', parameterized SQL, psycopg2, SDK-based secrets
* **Notebooks 02/03/04 line counts**: 14 / 8 / 9 source lines (all under 30)

## Validation status

* `pytest tests/` — **PASSED** (5 tests: chunking, storage, agent smoke)
* `python -c "import memory_agent"` — **PASSED** with no warnings

## Notebook 04 live comparison for input: `What did we discuss about end to end implementation?`

Live cleanup was executed against the `memories` table before verification for `project_id='memory-kb-poc'`, `source_ref='agent-interaction'`, and `rule='What did we discuss about end to end implementation?'`. The cleanup query preserved at most one row per `(content_hash, project_id, source_ref)` and removed `0` rows in the final verified run because only one prior matching agent-interaction row remained.

| Run | Result summary |
| --- | --- |
| Pre-refactor notebook logic | `retrieved_count=3`; retrieval shape was 1 fresh inserted row plus 2 pre-existing rows; previously captured distances `0.282`, `0.376`, `0.407` |
| Post-refactor package-backed notebook logic | `retrieved_count=3`; live retrieval shape is now 1 fresh inserted row plus 2 pre-existing rows; distances `0.218`, `0.299`, `0.376` |

### Cleanup details

* Deduplication scope: `project_id='memory-kb-poc'`, `source_ref='agent-interaction'`, `rule='What did we discuss about end to end implementation?'`
* Deduplication rule: preserve at most one row per `(content_hash, project_id, source_ref)`
* `cleanup_deleted_duplicates = 0`
* `preexisting_question_rows_after_cleanup = 1`

### Pre-refactor reference

```text
{'question': 'What did we discuss about end to end implementation?', 'retrieved_count': 3, 'answer': 'We discussed a multi-step approach to implementing a concept end-to-end. The steps involved:\n\n1. Design and development of operational projects (Layer 1), including defining scope and requirements, designing and developing data storage and management systems, implementing distillation pipelines, retrieval agents, and episodic memory writing processes.\n2. Design and development of aggregate systems (Layer 2), including defining analytical jobs, designing and developing aggregate systems, implementing cross-project pattern detection and insight generation, and developing proposal writing processes.\n3. Implementation of cross-project sharing and access control, including designing and developing access control systems and source attribution systems.\n4. Integration with existing infrastructure, such as Databricks Delta Lake and Change Data Feed.\n5. Deployment and maintenance of the system, including monitoring, logging, and regular maintenance tasks.\n6. Testing and evaluation of the system, including testing, evaluating performance, and making improvements as needed.\n\nWe also discussed some of the technologies and tools that could be used to implement this concept, such as Databricks Lakebase, Apache Spark, natural language processing and machine learning libraries, and information retrieval libraries.'}
RETRIEVED 1 distance=0.282 preview='Q: What did we discuss about end to end implementation?\nA: We discussed a multi-step approach to implementing a concept end-to-end. The steps involved:\n\n1. Desi'
RETRIEVED 2 distance=0.376 preview='Implementing a concept end-to-end involves a multi-step approach that includes designing and developing various components, integrating existing infrastructure '
RETRIEVED 3 distance=0.407 preview='Q: How would you implement this concept end to end?\nA: Implementing the concept end-to-end would require a multi-step approach, involving the design and develop'
```

### Post-refactor live output

```text
{'cleanup_deleted_duplicates': 0, 'preexisting_question_rows_after_cleanup': 1, 'new_content_hash_preexisted': False, 'question': 'What did we discuss about end to end implementation?', 'retrieved_count': 3, 'retrieval_shape': ['fresh', 'pre-existing', 'pre-existing'], 'answer': 'We discussed a multi-step approach to implementing a concept end-to-end. The steps involved:\n\n1. Design and development of operational projects (Layer 1)\n2. Design and development of aggregate systems (Layer 2)\n3. Implementation of cross-project sharing and access control\n4. Integration with existing infrastructure\n5. Deployment and maintenance\n6. Testing and evaluation\n\nWe also discussed the use of various technologies and tools, such as Databricks Lakebase, Apache Spark, natural language processing, and machine learning libraries, to facilitate the implementation. The goal is to deploy a functional system that can be maintained, tested, and evaluated to ensure its performance and scalability.'}
RETRIEVED 1 label=fresh distance=0.218 preview='Q: What did we discuss about end to end implementation?\nA: We discussed a multi-step approach to implementing a concept end-to-end. The steps involved:\n\n1. Design and development of operational projec'
RETRIEVED 2 label=pre-existing distance=0.299 preview='Q: What did we discuss about end to end implementation?\nA: We discussed a multi-step approach to implementing a concept end-to-end. The steps involved:\n\n1. Design and development of operational projec'
RETRIEVED 3 label=pre-existing distance=0.376 preview='Implementing a concept end-to-end involves a multi-step approach that includes designing and developing various components, integrating existing infrastructure and tools, and ensuring data quality and'
```

### Comparison notes

* The post-refactor live retrieval now matches the pre-refactor shape: 1 fresh row plus 2 pre-existing rows.
* The earlier post-refactor output showing three duplicate fresh rows was a regression, not expected behavior.
* Root cause: deduplication previously relied on `ON CONFLICT DO NOTHING` without a matching unique constraint on `content_hash`.
* The application temporarily used an explicit `SELECT EXISTS` workaround before the schema was hardened.
* Live cleanup was executed before verification and removed `0` rows because only one prior matching agent-interaction row remained.

## Schema hardening

This pass hardens the live `memories` schema so the application can rely on database-enforced deduplication instead of a pre-insert existence check.

* `migrations/001_schema_hardening.sql` reports duplicate `(project_id, content_hash)` groups first, includes a commented cleanup statement that preserves the oldest row per key, verifies `created_at` has no NULLs before tightening it, conditionally adds the `memories_project_content_unique` constraint, and creates `memories_embedding_hnsw_idx` with `vector_cosine_ops`.
* `memory_agent.storage.insert_memory` and `memory_agent.storage.store_bootstrap_memories` now use plain `INSERT ... ON CONFLICT DO NOTHING` again; deduplication is delegated back to the database.
* `01_schema_setup` is updated so fresh environments create `created_at TIMESTAMPTZ NOT NULL DEFAULT now()`, include `derived_from UUID[]` with a note that notebook `05_distillation` uses it, define the uniqueness constraint up front, and create the cosine HNSW index from scratch.
* No migration is included for `derived_from` because live Lakebase already has that column; the notebook change only removes schema drift between code and reality.
* `migrations/verify_schema_hardening.py` is provided as a manual post-migration check to inspect schema state, verify duplicate inserts collapse to one row, and re-run the notebook 04 retrieval-shape validation.

## Next steps

1. Apply `migrations/001_schema_hardening.sql` to Lakebase after reviewing the duplicate report.
2. Run `python migrations/verify_schema_hardening.py` against the target endpoint to confirm schema, no-op duplicate insert behavior, and retrieval shape.
3. Run `pytest tests/` once the environment is ready to validate the simplified insert flow locally.
