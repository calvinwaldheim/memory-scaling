# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Memory Retention Eval
# MAGIC
# MAGIC Tests the core memory-scaling hypothesis end-to-end:
# MAGIC
# MAGIC 1. **Teach** a novel fact through the agent (`ask()` stores it as episodic)
# MAGIC 2. **Recall** it via a paraphrased question that shares few or no keywords
# MAGIC 3. **Score** whether the answer surfaces the taught fact
# MAGIC
# MAGIC The point isn't BLEU or ROUGE — it's whether the agent gets *strictly
# MAGIC better* after a conversation than it was before. If the recall answer
# MAGIC mentions terms that only appear in the taught fact (and nowhere in the
# MAGIC bootstrapped source documents), then memory worked.
# MAGIC
# MAGIC Design notes:
# MAGIC - Recall uses a separate `recall()` function that does NOT write the
# MAGIC   interaction back as an episodic memory (we don't want eval queries
# MAGIC   polluting the store between re-runs).
# MAGIC - LLM calls during recall use `temperature=0` for reproducibility.
# MAGIC - Scoring is simple keyword presence — easy to upgrade to LLM-judge later.

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

PROJECT_ID = "memory-kb-poc"
PROJECT_TYPE = "product"
EMBED_ENDPOINT = "databricks-gte-large-en"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

client = get_deploy_client("databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Helpers — embed, retrieve, teach, recall

# COMMAND ----------

def embed(text: str) -> list[float]:
    response = client.predict(
        endpoint=EMBED_ENDPOINT,
        inputs={"input": [text]},
    )
    return response["data"][0]["embedding"]


def retrieve(query: str, top_k: int = 3) -> list[tuple[str, str, float]]:
    """Return list of (memory_type, context, distance) for the top-k matches."""
    q_vec = embed(query)
    with psycopg2.connect(CONN_STRING, password=TOKEN) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT memory_type, context, embedding <=> %s::vector AS distance
            FROM memories
            WHERE project_id = %s
            ORDER BY distance ASC
            LIMIT %s
            """,
            (json.dumps(q_vec), PROJECT_ID, top_k),
        )
        return cur.fetchall()


def teach(fact: str) -> str:
    """
    Same shape as 04_agent's ask(): retrieve context, generate answer, write
    the interaction back as an episodic memory. We use this to seed novel
    facts into memory.
    """
    memories = retrieve(fact)
    context = "\n\n".join(m[1] for m in memories)

    response = client.predict(
        endpoint=LLM_ENDPOINT,
        inputs={
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Acknowledge the user's "
                        "statement and integrate it with the context below. "
                        "If the context contradicts the statement, treat the "
                        "user's statement as authoritative.\n\nCONTEXT:\n"
                        f"{context}"
                    ),
                },
                {"role": "user", "content": fact},
            ],
        },
    )
    answer = response["choices"][0]["message"]["content"]

    # Write back as episodic — this is the seeding step
    content = f"Q: {fact}\nA: {answer}"
    content_hash = hashlib.md5(content.encode()).hexdigest()
    e = embed(content)

    with psycopg2.connect(CONN_STRING, password=TOKEN) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO memories
                (project_id, project_type, memory_type, scope, domain,
                 rule, context, source_ref, content_hash, embedding, quality_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (
                PROJECT_ID, PROJECT_TYPE, "episodic", "organizational",
                "eval-seed", fact[:100], content, "eval-teach",
                content_hash, json.dumps(e), 0.9,
            ),
        )
        conn.commit()

    return answer


def recall(question: str, top_k: int = 3) -> tuple[str, list[tuple[str, str, float]]]:
    """
    Like ask(), but:
    - temperature=0 for reproducibility
    - does NOT write the interaction back (don't pollute the store)
    Returns (answer, retrieved_memories).
    """
    memories = retrieve(question, top_k=top_k)
    context = "\n\n".join(m[1] for m in memories)

    response = client.predict(
        endpoint=LLM_ENDPOINT,
        inputs={
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the question "
                        "using ONLY the context below. If the context doesn't "
                        "contain the answer, say so explicitly.\n\nCONTEXT:\n"
                        f"{context}"
                    ),
                },
                {"role": "user", "content": question},
            ],
            "temperature": 0,
        },
    )
    return response["choices"][0]["message"]["content"], memories

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test facts
# MAGIC
# MAGIC Each entry:
# MAGIC - `teach`: the novel statement we feed into the agent
# MAGIC - `recall`: a paraphrased question that shares minimal keywords with `teach`
# MAGIC - `expected_terms`: terms whose presence in the recall answer indicates the
# MAGIC   taught fact was successfully retrieved (case-insensitive substring match)
# MAGIC
# MAGIC These are deliberately invented facts about the POC itself — not in
# MAGIC `concept.txt`, not in LLM training data, so a correct answer can only
# MAGIC come from memory.

# COMMAND ----------

TEST_FACTS = [
    {
        "id": "f01",
        "teach": "For this POC, we've decided the production distillation threshold should be 0.30, not the 0.25 we used during development.",
        "recall": "What clustering threshold are we targeting for production?",
        "expected_terms": ["0.30", "production"],
    },
    {
        "id": "f02",
        "teach": "We've named the internal review workflow for distilled memories 'Tinker'. Every semantic memory must pass Tinker before going live.",
        "recall": "Is there a review step before distilled memories are released?",
        "expected_terms": ["Tinker", "review"],
    },
    {
        "id": "f03",
        "teach": "The next major milestone for the memory-scaling POC is integrating Delta Lake Change Data Feed for staleness detection, targeted for end of sprint.",
        "recall": "What's the upcoming priority for the POC?",
        "expected_terms": ["Change Data Feed", "staleness"],
    },
    {
        "id": "f04",
        "teach": "We've capped synthesis at 8 episodic examples per cluster instead of 10, because larger prompts were occasionally truncated by the Foundation Model API.",
        "recall": "Why did we reduce the per-cluster example count?",
        "expected_terms": ["8", "truncated"],
    },
    {
        "id": "f05",
        "teach": "The success criterion for the eval is a memory hit rate of at least 70 percent across paraphrased recall queries.",
        "recall": "What's our pass bar for the retention test?",
        "expected_terms": ["70", "hit rate"],
    },
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 1 — teach
# MAGIC
# MAGIC Feed each fact through `teach()`. The agent writes one new episodic
# MAGIC memory per fact. Inspect the answers to make sure the LLM didn't just
# MAGIC reject the statement (it shouldn't; the system prompt tells it to
# MAGIC accept the user as authoritative).

# COMMAND ----------

print(f"Seeding {len(TEST_FACTS)} novel facts into memory...\n")
for fact in TEST_FACTS:
    answer = teach(fact["teach"])
    print(f"[{fact['id']}] taught")
    print(f"  fact:   {fact['teach']}")
    print(f"  reply:  {answer[:200]}...")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 2 — recall
# MAGIC
# MAGIC Paraphrased queries. We score on whether `expected_terms` appear in the
# MAGIC answer (case-insensitive). Also print which memories were retrieved so
# MAGIC failures are debuggable — if the right memory wasn't retrieved, it's a
# MAGIC retrieval problem; if it was retrieved but the answer didn't mention
# MAGIC the terms, it's a generation problem.

# COMMAND ----------

results = []
for fact in TEST_FACTS:
    answer, mems = recall(fact["recall"])
    answer_lc = answer.lower()
    hits = [t for t in fact["expected_terms"] if t.lower() in answer_lc]
    score = len(hits) / len(fact["expected_terms"])
    results.append({
        "id": fact["id"],
        "query": fact["recall"],
        "expected": fact["expected_terms"],
        "hits": hits,
        "score": score,
        "answer": answer,
        "retrieved": [(m[0], m[2], m[1][:120]) for m in mems],
    })

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results

# COMMAND ----------

passed = sum(1 for r in results if r["score"] == 1.0)
partial = sum(1 for r in results if 0 < r["score"] < 1.0)
failed = sum(1 for r in results if r["score"] == 0)

print(f"=== Summary ===")
print(f"  Full pass:  {passed}/{len(results)}")
print(f"  Partial:    {partial}/{len(results)}")
print(f"  Fail:       {failed}/{len(results)}")
print(f"  Mean score: {sum(r['score'] for r in results) / len(results):.2f}")

for r in results:
    print(f"\n--- [{r['id']}] score={r['score']:.2f} ---")
    print(f"  query:    {r['query']}")
    print(f"  expected: {r['expected']}")
    print(f"  hits:     {r['hits']}")
    print(f"  answer:   {r['answer'][:300]}...")
    print(f"  retrieved (top-3):")
    for kind, dist, snippet in r["retrieved"]:
        print(f"    [{kind:8s}] d={dist:.3f}  {snippet}…")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interpreting the output
# MAGIC
# MAGIC - **All facts score 1.0** → the basic memory loop works end-to-end.
# MAGIC - **A fact has the right memory in `retrieved` but score < 1.0** → the
# MAGIC   answer didn't surface the expected terms. Try sharpening the recall
# MAGIC   query or loosening `expected_terms` (sometimes the LLM paraphrases).
# MAGIC - **A fact has the wrong memories retrieved** → it's a retrieval
# MAGIC   problem, not a generation problem. The taught fact's embedding sits
# MAGIC   too far from the recall query's embedding. Either the query is too
# MAGIC   abstract, or there's a closer source-doc chunk crowding it out.
# MAGIC
# MAGIC ## Re-running after distillation
# MAGIC
# MAGIC Once you have ≥ 2 facts on the same topic, run `05_distillation` to
# MAGIC produce semantic memories from the new episodic rows, then re-run this
# MAGIC notebook. The expected behavior: semantic memories now appear in
# MAGIC `retrieved` for the broadest recall queries, and the answer quality
# MAGIC improves on those without changing the underlying model.
