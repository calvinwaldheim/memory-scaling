# AGENT.md — operating manual for Claude using the memory-scaling MCP

You are connected to the **memory-scaling** MCP server. It gives you a durable, project-scoped memory store that compounds across sessions: facts written in one conversation are retrievable in the next.

This file is your operating manual. Read it once at session start and follow it as you decide when to write, when to read, and when to correct what's already there. **It is not background info — the heuristics below are what separate a useful memory layer from noise.**

---

## The shape of the store

One PostgreSQL table (`memories`) holds every fact you have ever stored. Each row has:

- **`content`** — the durable fact, embedded for semantic retrieval. **Immutable after write.** To change content, you `supersede` the row (see below). Never try to "edit" content directly.
- **`memory_type`** — `"episodic"` (raw observations, Q+A pairs, single interactions) or `"semantic"` (distilled generalizations, durable rules). When in doubt, write episodic; the distillation pipeline elevates clusters of episodics into semantics over time.
- **`project_id`** — every memory belongs to exactly one project (a slug like `trackunit-customer`, `memory-kb-poc`). Cross-project recall is supported but explicit. **Sharing is the access boundary** — see Access model below.
- **`domain`** — optional sub-classifier within a project (`"architecture"`, `"interactions"`, etc.). Useful for filtering.
- **`rule`** — a one-line summary of the content. The retrieval result shows this prominently, so write it like a headline.
- **`created_by`** — the verified user (from the MCP token verifier) who wrote the row. You don't set this — the server does, from your authenticated identity. Trying to attribute a write to anyone else is rejected by design.
- **Lineage fields** — `superseded_by`, `superseded_at`, `forgotten_at` (and friends). These are why this store is different from a vector cache: corrections leave an audit trail rather than vanishing.

**Default retrieval excludes superseded and forgotten rows.** This is the most important invariant in the whole system. Wrong beliefs stay in the database for audit, but they never surface in `recall` unless you ask for them explicitly. You do not need to think about contamination from old beliefs.

## Access model

Every project has an ACL (`project_acl` table) defining who can do what. There are three roles:

- **`viewer`** — can read (`recall`, `stats`, `list_hot`, `get_lineage`, `get_audit_log`).
- **`contributor`** — viewer + can write (`remember`, `supersede`, `update_memory`, soft `forget`).
- **`owner`** — contributor + can administer (`forget(hard=True)`, `archive_project`, `grant_access`, `revoke_access`).

**Sharing is the only access boundary.** A project is private by default — only the creator has access until they explicitly grant others. There's no per-row "personal" flag any more; if you want a fact private to you, put it in a project you haven't shared.

When you call `create_project`, you become its owner automatically. When you call `recall` or `list_projects`, you only ever see projects you have at least viewer on — projects you can't see are silently omitted, never errored on. Trying to write to a project where you don't have contributor (or trying to `forget(hard=True)` without owner) raises `PermissionError` — that's a real authorization failure and should surface back to the user.

**Every mutation writes an audit-log entry** (`memory_audit_log`) in the same transaction as the data change. The entry records who, when, what action, the reason you supplied, and a before/after JSON snapshot. The log is the action stream of record — query it via `get_audit_log(memory_id)`.

---

## The tool surface (memorize this)

### Reading (require `viewer`)

| Tool | When |
|---|---|
| `recall(query, top_k=3, project_id=…, include_inactive=False)` | The primary read. Embeds the query, returns ranked memories with `distance` (smaller = closer). Use it whenever you need grounded context. Silently filters out projects you don't have access to. |
| `stats(project_id=…)` | Inventory + health snapshot of a project. `total`, `active_count`, `superseded_count`, `forgotten_count`, retrieval percentiles, avg quality. Use when deciding whether a project's memory base is mature enough to trust. |
| `list_hot(top_k=10, project_id=…)` | Most-retrieved memories. Reveals what agents actually keep pulling. Pair with `stats` to gauge concentration. |
| `get_lineage(memory_id)` | Walks the supersede chain forward (target → head) and backward (ancestors). Use for audit questions: "what did we believe before X". |
| `get_audit_log(memory_id, limit=50)` | The action stream for one memory — every create / supersede / forget / update / purge with actor, reason, before/after JSON. Use for "who did what, when" investigations. |

### Writing (require `contributor`)

| Tool | When |
|---|---|
| `remember(content, source_ref, …)` | Store a **new** fact. The right default when the agent has just learned something durable that wasn't in the store before. Dedupes via content hash — calling twice with the same content returns `{"status": "duplicate"}`. Your verified identity is stored as `created_by` — you don't pass it. |
| `supersede(old_id, new_content, reason, …)` | **Correct** a wrong/outdated memory. Atomically writes the new content, links the old → new via `superseded_by`, fills `superseded_reason` and `superseded_by_user` (from your verified identity). Use this — not `forget + remember` — whenever you have a replacement. |
| `forget(memory_id, reason)` | **Retract** a memory with no replacement. Soft by default (sets `forgotten_at`, keeps the row for audit). `hard=True` requires `owner` — see below. |
| `update_memory(memory_id, rule=…, domain=…, quality_score=…, …)` | Edit **metadata only**. Cannot touch `content` (that's what `supersede` is for). Right call when fixing a `domain` label, lowering `quality_score` after the memory turned out unreliable, or promoting `episodic` → `semantic` after review. |

### Admin (require `owner`)

| Tool | When |
|---|---|
| `forget(memory_id, reason, hard=True)` | Destructive purge. The row is gone — no audit-trail recovery. Use **only** for GDPR right-to-be-forgotten or genuine secret leakage. The audit-log entry survives the deletion. |
| `archive_project(project_id)` | Soft-delete a project (`archived_at` set). Memories stay queryable if you pass the archived id explicitly. |
| `grant_access(project_id, user_name, role)` | Share a project with another user at a given role. Re-granting upserts to the new role. |
| `revoke_access(project_id, user_name)` | Remove a user's access to a project. |

### Projects + access (any authenticated user)

| Tool | When |
|---|---|
| `create_project(project_id, name, project_type, description, tags)` | Register a new project. You become its `owner` automatically. `project_type` is one of `data_domain`, `engineering`, `compliance`, `customer`, `product`. |
| `list_projects(include_archived=False)` | See projects **you have at least viewer access to**. Each entry has `memory_count`. Projects you can't see are silently omitted. |
| `set_active_project(project_id)` | Make this project the default for *your* subsequent tool calls. Per-user (other agents on the same MCP server aren't affected). Requires viewer on the target. Forgotten when the MCP subprocess restarts. |
| `get_active_project()` | Reports *your* active project, the default fallback, and the effective project. |
| `list_access(project_id)` | List every (user, role) pair on a project. Requires viewer. Useful before granting/revoking. |

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
| Trying to update `content` via `update_memory` | Content is immutable — the call will error or be rejected. | `supersede` is the only way to change content. `update_memory` is for `rule`, `domain`, `quality_score`, `memory_type` only. |
| Writing to a project you can't see | Will raise `PermissionError`. Don't paper over it — surface the failure to the user; either the project doesn't exist or they need to be granted access. | Check via `list_projects` first if unsure. The user might want to `create_project` (which auto-grants them owner) or ask a current owner to `grant_access`. |
| Hard-deleting by default | `forget(hard=True)` destroys the row permanently and requires owner. The audit-log entry survives but recovery is impossible. | Soft is the default and almost always the right answer. Reserve `hard=True` for GDPR / secret-leak situations. |
| Writing to a "personal" project that's actually shared | The privacy model is project-level. If you grant another user access, everything in the project is visible to them. | Before writing sensitive facts, check `list_access(project_id)` to see who else has access. If you want truly private memory, use a project you haven't shared. |
| Setting the active project once and never re-checking it | The active project lives per-user in the MCP subprocess; restarts wipe it. | Either set it at the start of every session, or pass `project_id=` explicitly on each call. |

---

## Mental model in three sentences

1. **The store is a project-scoped vector database with a built-in version history.** Every memory has lineage; corrections are first-class.
2. **`recall` is the read primitive, `supersede` is the correction primitive, `remember` is the new-fact primitive.** Everything else is metadata or admin.
3. **Default retrieval excludes inactive rows.** You can write wrong things, correct them later, and trust that the wrong version will never resurface in normal use.

If you remember nothing else: **when something is wrong and you know the fix, `supersede` it. Don't `forget`-then-`remember`.**
