# AGENT.md — operating manual for Claude using the memory-scaling MCP

You are connected to the **memory-scaling** MCP server. It gives you a durable, project-scoped memory store that compounds across sessions: facts written in one conversation are retrievable in the next.

This file is your operating manual. Read it once at session start and follow it as you decide when to write, when to read, and when to correct what's already there. **It is not background info — the heuristics below are what separate a useful memory layer from noise.**

---

## The shape of the store

One PostgreSQL table (`memories`) holds every fact you have ever stored. Each row has:

- **`content`** — the durable fact, embedded for semantic retrieval. **Immutable after write.** To change content, you `supersede` the row (see below). Never try to "edit" content directly.
- **`memory_type`** — `"episodic"` (raw observations, Q+A pairs, single interactions) or `"semantic"` (distilled generalizations, durable rules). When in doubt, write episodic; the distillation pipeline elevates clusters of episodics into semantics over time.
- **`project_id`** — every memory belongs to exactly one project (a slug like `trackunit-customer`, `memory-kb-poc`). Cross-project recall is supported but explicit.
- **`scope`** — `"personal"` (tied to a `user_id`) or `"organizational"` (visible to all agents in the project). Default to organizational unless you have a reason.
- **`domain`** — optional sub-classifier within a project (`"architecture"`, `"interactions"`, etc.). Useful for filtering.
- **`rule`** — a one-line summary of the content. The retrieval result shows this prominently, so write it like a headline.
- **Lineage fields** — `superseded_by`, `superseded_at`, `forgotten_at` (and friends). These are why this store is different from a vector cache: corrections leave an audit trail rather than vanishing.

**Default retrieval excludes superseded and forgotten rows.** This is the most important invariant in the whole system. Wrong beliefs stay in the database for audit, but they never surface in `recall` unless you ask for them explicitly. You do not need to think about contamination from old beliefs.

---

## The tool surface (memorize this)

### Reading

| Tool | When |
|---|---|
| `recall(query, top_k=3, project_id=…, include_inactive=False)` | The primary read. Embeds the query, returns ranked memories with `distance` (smaller = closer). Use it whenever you need grounded context. |
| `stats(project_id=…)` | Inventory + health snapshot of a project. `total`, `active_count`, `superseded_count`, `forgotten_count`, retrieval percentiles, avg quality. Use when deciding whether a project's memory base is mature enough to trust. |
| `list_hot(top_k=10, project_id=…)` | Most-retrieved memories. Reveals what agents actually keep pulling. Pair with `stats` to gauge concentration. |
| `get_lineage(memory_id)` | Walks the supersede chain forward (target → head) and backward (ancestors). Use for audit questions: "what did we believe before X, when did we change our minds, who changed them". |

### Writing

| Tool | When |
|---|---|
| `remember(content, source_ref, …)` | Store a **new** fact. The right default when the agent has just learned something durable that wasn't in the store before. Dedupes via content hash — calling twice with the same content returns `{"status": "duplicate"}`. |
| `supersede(old_id, new_content, reason, …)` | **Correct** a wrong/outdated memory. Atomically writes the new content, links the old → new via `superseded_by`, fills `superseded_reason` and `superseded_by_user`. Use this — not `forget + remember` — whenever you have a replacement. |
| `forget(memory_id, reason, hard=False)` | **Retract** a memory with no replacement. Soft by default (sets `forgotten_at`, keeps the row for audit). `hard=True` is destructive — only for GDPR-style erasure or true secret leakage. |
| `update_memory(memory_id, rule=…, domain=…, quality_score=…, …)` | Edit **metadata only**. Cannot touch `content` (that's what `supersede` is for). Right call when fixing a `domain` label, lowering `quality_score` after the memory turned out unreliable, or promoting `episodic` → `semantic` after review. |

### Projects

| Tool | When |
|---|---|
| `create_project(project_id, name, project_type, description, tags)` | Register a new project. `project_type` is one of `data_domain`, `engineering`, `compliance`, `customer`, `product`. |
| `list_projects(include_archived=False)` | See what projects exist. Each entry has `memory_count`. |
| `set_active_project(project_id)` | Make this project the default for the rest of the conversation. After this, every subsequent `recall`/`remember`/`supersede`/etc. defaults to this project unless overridden. **Forgotten when the MCP subprocess restarts** — re-set it each session if needed. |
| `get_active_project()` | Reports the currently active project, the default fallback, and the effective project. |
| `archive_project(project_id)` | Soft-delete a project. Memories are not removed — they remain queryable if you pass the archived id explicitly. |

---

## When to do what — concrete heuristics

### Starting a session

If you don't already know which project this work belongs to:
1. Call `list_projects()` to see what exists.
2. Call `get_active_project()` to see what's currently selected.
3. If the user names a project that doesn't exist yet, `create_project(...)` then `set_active_project(...)`.

If you do know the project, just `set_active_project(...)` and proceed.

### Grounding an answer

**Before answering any question whose answer might reasonably exist in memory, call `recall` first.** This is the whole point of the system. Don't reason from training data when the store has the project-specific fact.

- Use `top_k=3` for focused lookups, `top_k=10` when you're surveying ("what do we know about X?").
- Use `domain=…` or `min_quality_score=0.7` to narrow when results feel noisy.
- Use `project_ids=[a, b, c]` for cross-project queries ("how do other customer engagements handle this?").

### Writing a new memory

Write when you've learned something **durable and reusable**:
- Architectural decisions, customer constraints, system invariants, durable preferences.
- The conclusion of a debugging session (the *fix*, not the keystrokes).
- A non-obvious answer the user confirmed.

Don't write:
- Ephemeral conversation state ("currently working on X") — that belongs in a todo or plan.
- Things easily derivable from the current code or git history.
- One-off greetings, social chatter, every interaction by reflex.

Write a sharp `rule` (the one-line headline) — it's what retrieval results lead with, and it's what you'll see first when this memory comes back.

### Correcting a wrong memory

You found a memory in `recall` results that is **wrong** and you know the right answer.

**Always prefer `supersede` over `forget + remember`.** The supersede path is atomic, leaves an audit trail, and excludes the wrong belief from future retrieval automatically. The forget+remember path has a race window where the fact is missing, no link between the new and old versions, and the old content stays in embedding space until purged.

```
supersede(
  old_id="<id from recall>",
  new_content="<corrected content>",
  reason="<why — required>",
  rule="<new one-liner>",
)
```

**Read the response carefully.** Three non-success outcomes you must handle:

- `{"status": "already_superseded", "current_head_id": "<uuid>"}` — someone else corrected this in parallel. Read the head (`recall` with the id, or just trust the head to be more current) and decide whether your correction is still needed. Don't blindly retry against the head — your fix may already be there, or may now conflict.
- `{"status": "forgotten"}` — the memory was retracted with no replacement. Call `remember` with the new content as a fresh memory.
- `{"status": "duplicate_content"}` — your corrected content matches an existing memory's hash. Reword the correction, or `forget` the collider first.

### Retracting a memory with no replacement

Use `forget(memory_id, reason="…")`. Soft is the default and right answer almost always. The memory stops surfacing in `recall` but stays in the database for audit. Always pass a `reason` — anonymous deletes are noise in audit reports.

Only use `hard=True` when:
- A secret leaked into a memory and must actually be erased.
- A GDPR right-to-be-forgotten request applies.
- You are explicitly told to purge.

### Auditing — "what did we believe before?"

`recall(query, include_inactive=True)` surfaces superseded and forgotten rows alongside live ones. Each row in the result has `superseded_at`, `superseded_by`, and `forgotten_at` set (or `None` if it's live). Walk the chain with `get_lineage(memory_id)`.

### Health-checking a project

```
stats(project_id="...")
```

Read it like this:
- `active_count` low relative to `total` ⇒ heavy churn / lots of corrections.
- `cold_count` high relative to `active_count` ⇒ pruning would be productive; many memories never get retrieved.
- `retrieval_count_p90 ≫ p50` ⇒ long-tail usage, a few memories doing most of the work.
- `avg_quality_score < 0.5` ⇒ the store is dominated by low-confidence material.

---

## Anti-patterns (do not do these)

| Anti-pattern | Why it's wrong | Do this instead |
|---|---|---|
| `forget(id)` then `remember(corrected_content)` | Loses the link between old and new; old content sits in embedding space until purged; racy under concurrency. | `supersede(old_id, new_content, reason)`. |
| `forget(id, hard=True)` as the default | Destroys the audit trail. Future "what did we believe?" queries can't see the retraction. | `forget(id, reason="…")` (soft is the default). Only use `hard=True` for GDPR/secret erasure. |
| Writing every interaction with `remember` | Pollutes the store with ephemera; retrieval quality drops for everyone. | Write only durable, reusable facts. Reach for `remember` when *future you, in a future session* would benefit. |
| Skipping `recall` because "I know the answer" | The whole point of the system is that the store carries project-specific facts your training data doesn't. | Call `recall` whenever the answer might plausibly live in the project's memory. |
| Trying to update `content` via `update_memory` | Content is immutable — the call will error or be rejected. | `supersede` is the only way to change content. `update_memory` is for `rule`, `domain`, `quality_score`, `memory_type`, `scope` only. |
| Mixing personal and organizational scope without thinking | Personal-scope rows tied to a `user_id` won't show up for other users. Organizational rows are visible to everyone in the project. | Default to `scope="organizational"`. Use `scope="personal"` only for genuinely user-specific facts. |
| Setting the active project once and never re-checking it | The active project lives in the MCP subprocess; restarts wipe it. | Either set it at the start of every session, or pass `project_id=` explicitly on each call. |

---

## Mental model in three sentences

1. **The store is a project-scoped vector database with a built-in version history.** Every memory has lineage; corrections are first-class.
2. **`recall` is the read primitive, `supersede` is the correction primitive, `remember` is the new-fact primitive.** Everything else is metadata or admin.
3. **Default retrieval excludes inactive rows.** You can write wrong things, correct them later, and trust that the wrong version will never resurface in normal use.

If you remember nothing else: **when something is wrong and you know the fix, `supersede` it. Don't `forget`-then-`remember`.**
