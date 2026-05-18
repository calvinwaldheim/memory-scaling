from __future__ import annotations

"""FastMCP server exposing the memory-scaling POC tools as a Databricks App."""

import os
from typing import Any

import httpx
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from pydantic import AnyHttpUrl

from memory_agent import agent as memory_agent
from memory_agent import embeddings, storage

WORKSPACE_URL = os.environ.get(
    "DATABRICKS_WORKSPACE_URL",
    "https://dbc-1223ae6c-4282.cloud.databricks.com",
)
APP_URL = os.environ.get(
    "DATABRICKS_APP_URL",
    "https://memory-scaling-mcp-7474648789573088.aws.databricksapps.com",
)


class DatabricksTokenVerifier(TokenVerifier):
    async def verify_token(self, token: str) -> AccessToken | None:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{WORKSPACE_URL}/api/2.0/preview/scim/v2/Me",
                headers={"Authorization": f"Bearer {token}"},
            )
        if r.status_code != 200:
            return None
        return AccessToken(
            token=token,
            client_id="claude",
            scopes=["all-apis", "offline_access"],
        )


TRANSPORT = os.environ.get("MCP_TRANSPORT", "streamable-http")

# Default project used when no active project has been set and no project_id
# was passed to the tool. Overridable per server instance via the
# DEFAULT_PROJECT_ID env var in claude_desktop_config.json.
DEFAULT_PROJECT_ID = os.environ.get("DEFAULT_PROJECT_ID", "memory-kb-poc")

# Active project for this MCP subprocess. Lives in memory only — quitting
# Claude Desktop resets it. Set via set_active_project; consulted by every
# tool that takes a project_id parameter.
_active_project_id: str | None = None


def _resolve_project_id(explicit: str | None) -> str:
    """Pick the project_id for a single-project tool call.

    Precedence: explicit caller arg > active project > DEFAULT_PROJECT_ID env var.
    Never returns None.
    """
    if explicit:
        return explicit
    if _active_project_id:
        return _active_project_id
    return DEFAULT_PROJECT_ID


if TRANSPORT == "stdio":
    mcp = FastMCP("memory-scaling")
else:
    mcp = FastMCP(
        "memory-scaling",
        token_verifier=DatabricksTokenVerifier(),
        auth=AuthSettings(
            issuer_url=AnyHttpUrl(f"{WORKSPACE_URL}/oidc"),
            resource_server_url=AnyHttpUrl(APP_URL),
            required_scopes=["all-apis", "offline_access"],
        ),
    )


@mcp.tool()
def recall(
    query: str,
    top_k: int = 3,
    project_id: str | None = None,
    project_ids: list[str] | None = None,
    memory_type: str | None = None,
    domain: str | None = None,
    min_quality_score: float | None = None,
    include_inactive: bool = False,
) -> list[dict[str, Any]]:
    """Retrieve the nearest stored memories for a query.

    Call this when an agent needs grounded context from the memory store before answering or planning. The return value is a ranked list of memory dictionaries. Each item contains `id`, `content`, `source_ref`, `memory_type`, `domain`, `rule`, `quality_score`, `distance`, `project_id`, plus lineage fields `superseded_at`, `superseded_by`, and `forgotten_at` (all `None` for live rows). Lower `distance` means a closer match.

    Project targeting (in precedence order):
    - `project_ids` (list): search across multiple projects in one call, e.g. recalling across every customer-type project.
    - `project_id` (single string): explicit override for one project.
    - Active project (set via `set_active_project`).
    - `DEFAULT_PROJECT_ID` env var.

    Default behavior **excludes** rows that have been superseded (replaced by a corrected version) or soft-forgotten (retracted without replacement). This is what you almost always want — it keeps stale/wrong beliefs out of the retrieval window even though they remain in the table for audit. Pass `include_inactive=True` to also surface historical rows (use for audit, lineage inspection, "what did we believe on date X" queries).

    Side effect: each returned row's `retrieval_count` is incremented so future consolidation/pruning can prefer hot memories.

    Optional filters narrow the candidate set before similarity ranking:
    - `memory_type`: `"episodic"` for raw experiences/observations, or `"semantic"` for distilled rules/generalizations.
    - `domain`: exact-match string (e.g. `"architecture"`, `"interactions"`). Inspect `stats` for the available domains.
    - `min_quality_score`: inclusive lower bound (0.0–1.0). Use `0.7` to drop low-confidence memories.
    """
    if project_ids:
        targets = list(project_ids)
    else:
        targets = [_resolve_project_id(project_id)]
    memories = memory_agent.retrieve(
        question=query,
        project_ids=targets,
        top_k=top_k,
        memory_type=memory_type,
        domain=domain,
        min_quality_score=min_quality_score,
        include_inactive=include_inactive,
    )
    return [
        {
            "id": memory.id,
            "content": memory.context,
            "source_ref": memory.source_ref,
            "memory_type": memory.memory_type,
            "domain": memory.domain,
            "rule": memory.rule,
            "quality_score": memory.quality_score,
            "distance": memory.distance,
            "project_id": memory.project_id,
            "superseded_at": memory.superseded_at,
            "superseded_by": memory.superseded_by,
            "forgotten_at": memory.forgotten_at,
        }
        for memory in memories
    ]


@mcp.tool()
def remember(
    content: str,
    source_ref: str,
    memory_type: str = "episodic",
    scope: str = "organizational",
    domain: str | None = None,
    rule: str | None = None,
    project_id: str | None = None,
    quality_score: float = 0.5,
) -> dict[str, str]:
    """Store one new memory item in the project knowledge base.

    Call this when an agent has produced or observed durable context worth retaining across future sessions. The tool embeds `content` inline, writes through the memory storage layer with v0 internal defaults, and returns `{"status": "stored" | "duplicate", "content_hash": "..."}` so callers can tell whether a new row was written or deduplicated by the database.
    """
    project_id = _resolve_project_id(project_id)
    embedding = embeddings.embed(content)
    result = storage.insert_memory(
        project_id=project_id,
        project_type="product",
        memory_type=memory_type,
        scope=scope,
        domain=domain,
        rule=rule,
        context=content,
        source_ref=source_ref,
        embedding=embedding,
        quality_score=quality_score,
    )
    return {"status": result.status, "content_hash": result.content_hash}


@mcp.tool()
def forget(
    memory_id: str,
    reason: str | None = None,
    user_id: str | None = None,
    hard: bool = False,
) -> dict[str, Any]:
    """Retract one memory. Soft by default — the row stays in the table for audit.

    Use this when a memory is wrong, obsolete, or off-scope and should no longer surface in `recall`. The id comes from a prior `recall` result.

    Default (`hard=False`) is a **soft retraction**: sets `forgotten_at`, `forgotten_reason`, and `forgotten_by_user` on the row. The memory is no longer returned by default `recall` queries, but the row stays in the database and can be inspected via `recall(..., include_inactive=True)` or `get_lineage`. This is the audit-preserving path and should be the default choice. Returns `{"status": "forgotten" | "already_forgotten" | "not_found", ...}`.

    `hard=True` is a **destructive purge**: `DELETE FROM memories WHERE id = ?`. Use only for true erasure (GDPR right-to-be-forgotten, secret leaked into a memory, etc.). The row is gone — no audit trail, no recovery. Returns `{"status": "deleted", ...}` or `{"status": "not_found", ...}`.

    To **correct** a memory's content rather than retract it, prefer `supersede` over `forget`+`remember` — it links the old to the new in one atomic transaction and preserves the chain.

    Args:
        memory_id: UUID string from a prior `recall` result.
        reason: Free-text rationale. Optional but strongly recommended for soft retractions — anonymous deletes are noise in audit reports.
        user_id: Attribution string for the actor retracting the memory.
        hard: When True, hard-delete (audit-destroying). Default False (soft).
    """
    if hard:
        deleted = storage.delete_memory(memory_id)
        if deleted is None:
            return {"status": "not_found", "memory_id": memory_id}
        return {"status": "deleted", **deleted}
    return storage.soft_forget_memory(memory_id, reason=reason, user_id=user_id)


@mcp.tool()
def supersede(
    old_id: str,
    new_content: str,
    reason: str,
    user_id: str | None = None,
    rule: str | None = None,
    quality_score: float | None = None,
    source_ref: str | None = None,
    memory_type: str | None = None,
    scope: str | None = None,
    domain: str | None = None,
) -> dict[str, Any]:
    """Atomically replace a memory with a corrected version, preserving lineage.

    Use this whenever a memory's content has become wrong and you have the corrected wording. Compared to `forget` + `remember`, `supersede` is:
    1. **Atomic.** Both rows are written/updated in one transaction — agents never observe a state where the fact is missing or duplicated.
    2. **Audit-preserving.** The old row stays in the table with `superseded_by` pointing at the new row, `superseded_at` set, plus `superseded_reason` and `superseded_by_user`. The full chain is queryable via `get_lineage`.
    3. **Contamination-free.** Default `recall` excludes superseded rows, so the wrong content never surfaces semantically.

    The new memory **inherits** `project_id`, `project_type`, `user_id`, `memory_type`, `scope`, and `domain` from the old row, unless you explicitly override them. `rule`, `quality_score`, and `source_ref` are commonly worth setting — they describe the corrected belief, not the old one.

    Multi-writer race handling: if another agent already superseded `old_id`, this call returns `{"status": "already_superseded", "current_head_id": <uuid>, ...}`. Re-read the head memory before deciding whether to supersede it — the latest version may already incorporate your correction or conflict with it.

    Possible statuses in the return value:
    - `"superseded"` — success. Includes `old_id`, `new_id`, `head_id` (== `new_id`), `superseded_at`, `content_hash`.
    - `"already_superseded"` — old memory was already replaced. Includes `current_head_id`.
    - `"forgotten"` — old memory has been soft-forgotten; you should `remember` the corrected version as a fresh row instead.
    - `"duplicate_content"` — the new content hash collides with an existing memory in the same project. Reword or hard-forget the colliding row first.
    - `"not_found"` — `old_id` doesn't exist.

    Args:
        old_id: UUID of the memory being corrected (from `recall`).
        new_content: Corrected content. Will be embedded fresh.
        reason: Free-text rationale (required). Stored verbatim on the old row.
        user_id: Attribution string for the actor superseding the memory.
        rule: Optional override for the new row's `rule` summary. Defaults to the old row's rule.
        quality_score: Optional override (0.0–1.0). Defaults to 0.5 when the old row had no score.
        source_ref: Optional source pointer for the new memory.
        memory_type, scope, domain: Optional overrides; default to the old row's values.
    """
    embedding = embeddings.embed(new_content)
    return storage.supersede_memory(
        old_id=old_id,
        new_content=new_content,
        embedding=embedding,
        reason=reason,
        user_id=user_id,
        rule=rule,
        quality_score=quality_score,
        source_ref=source_ref,
        memory_type=memory_type,
        scope=scope,
        domain=domain,
    )


@mcp.tool()
def get_lineage(memory_id: str) -> dict[str, Any]:
    """Return the supersede chain and ancestor tree for one memory.

    Use this to answer audit questions like "what did we believe before X, and when did we change our minds, and who changed them". Walks two directions:
    - `chain`: forward from `memory_id` (depth 0) through each `superseded_by` pointer until the live head row (`superseded_at IS NULL`).
    - `ancestors`: backward — every row that (transitively) was replaced by `memory_id`.

    Each node in either list contains `id`, `rule`, `context`, `memory_type`, `domain`, `scope`, `quality_score`, `superseded_by`, `superseded_at`, `superseded_reason`, `superseded_by_user`, `forgotten_at`, `created_at`, and `depth`.

    Returns `{"status": "not_found", "memory_id": "..."}` if the id doesn't exist.
    """
    lineage = storage.get_lineage(memory_id)
    if lineage is None:
        return {"status": "not_found", "memory_id": memory_id}
    return {
        "target_id": lineage.target_id,
        "head_id": lineage.head_id,
        "chain": [_lineage_node_to_dict(n) for n in lineage.chain],
        "ancestors": [_lineage_node_to_dict(n) for n in lineage.ancestors],
    }


def _lineage_node_to_dict(node: storage.LineageNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "rule": node.rule,
        "context": node.context,
        "memory_type": node.memory_type,
        "domain": node.domain,
        "scope": node.scope,
        "quality_score": node.quality_score,
        "superseded_by": node.superseded_by,
        "superseded_at": node.superseded_at,
        "superseded_reason": node.superseded_reason,
        "superseded_by_user": node.superseded_by_user,
        "forgotten_at": node.forgotten_at,
        "created_at": node.created_at,
        "depth": node.depth,
    }


@mcp.tool()
def update_memory(
    memory_id: str,
    rule: str | None = None,
    domain: str | None = None,
    quality_score: float | None = None,
    memory_type: str | None = None,
    scope: str | None = None,
) -> dict[str, Any]:
    """Update lightweight metadata on one memory by id.

    Use this to relabel a memory without changing its content — e.g. fix the `domain` after observing that "interactions" should be "architecture", lower a `quality_score` after the memory turned out to be unreliable, or promote an `episodic` row to `semantic` after manual review. Only fields explicitly passed (non-None) are changed; the rest are left intact.

    `content` is deliberately not updatable here because changing it would invalidate the stored embedding and content_hash. To rewrite content, call `forget` and then `remember` with the new content.

    Returns the updated row's fields, or `{"status": "not_found", "memory_id": "..."}` if the id didn't match.
    """
    fields: dict[str, Any] = {}
    if rule is not None:
        fields["rule"] = rule
    if domain is not None:
        fields["domain"] = domain
    if quality_score is not None:
        fields["quality_score"] = quality_score
    if memory_type is not None:
        fields["memory_type"] = memory_type
    if scope is not None:
        fields["scope"] = scope
    if not fields:
        return {"status": "no_change", "memory_id": memory_id}
    updated = storage.update_memory_fields(memory_id, **fields)
    if updated is None:
        return {"status": "not_found", "memory_id": memory_id}
    return {"status": "updated", **updated}


@mcp.tool()
def stats(project_id: str | None = None) -> dict[str, Any]:
    """Summarize the current contents and usage of the memory store for one project.

    Call this when an agent needs a quick health, inventory, or usage check before relying on or curating the memory base.

    The return value includes inventory fields (`total`, `active_count`, `superseded_count`, `forgotten_count`, `by_memory_type`, `by_domain`, `first_written_at`, `last_written_at`), usage fields driven by `retrieval_count` (`cold_count` — memories never retrieved, `retrieval_count_p50` / `_p90` / `_max`), and a quality summary (`avg_quality_score`). All timestamps are ISO strings or `None`.

    `total = active_count + superseded_count + forgotten_count`. A high `superseded_count + forgotten_count` share signals heavy churn (lots of corrections), while a high `cold_count` relative to `active_count` is a signal that pruning would be productive.
    """
    summary = storage.stats(project_id=_resolve_project_id(project_id))
    return {
        "total": summary.total,
        "active_count": summary.active_count,
        "superseded_count": summary.superseded_count,
        "forgotten_count": summary.forgotten_count,
        "by_memory_type": summary.by_memory_type,
        "by_domain": summary.by_domain,
        "last_written_at": summary.last_written_at,
        "first_written_at": summary.first_written_at,
        "cold_count": summary.cold_count,
        "retrieval_count_p50": summary.retrieval_count_p50,
        "retrieval_count_p90": summary.retrieval_count_p90,
        "retrieval_count_max": summary.retrieval_count_max,
        "avg_quality_score": summary.avg_quality_score,
    }


@mcp.tool()
def list_hot(top_k: int = 10, project_id: str | None = None) -> list[dict[str, Any]]:
    """List the most-retrieved memories, sorted by `retrieval_count` (descending).

    Use this to see what the memory store is actually being used for — i.e. which rules and contexts agents keep pulling back. Memories with `retrieval_count = 0` are excluded; if no memory has been retrieved yet, the list is empty. Pair with `stats` to gauge how concentrated usage is (e.g. p90 ≫ p50 implies a long tail).

    Each item contains `id`, `rule`, `content`, `memory_type`, `domain`, `quality_score`, `retrieval_count`, and `created_at`.
    """
    return storage.list_hot_memories(project_id=_resolve_project_id(project_id), top_k=top_k)


# ----------------------------------------------------------------------------
# Project management tools
# ----------------------------------------------------------------------------


def _project_to_dict(p: storage.Project) -> dict[str, Any]:
    return {
        "project_id": p.project_id,
        "name": p.name,
        "project_type": p.project_type,
        "description": p.description,
        "tags": p.tags,
        "created_at": p.created_at,
        "created_by": p.created_by,
        "archived_at": p.archived_at,
        "memory_count": p.memory_count,
    }


@mcp.tool()
def create_project(
    project_id: str,
    name: str,
    project_type: str,
    description: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Register a new project that memories can be written to.

    Use this when starting work on a new context — a customer engagement, a product surface, a compliance investigation, etc. After creation you'll typically call `set_active_project` to make it the default for the rest of the conversation.

    Args:
        project_id: Slug (lowercase letters, digits, dashes only), e.g. `trackunit-customer`.
        name: Human-readable label, e.g. `Trackunit Customer Engagement`.
        project_type: One of `data_domain`, `engineering`, `compliance`, `customer`, `product`. This controls future type-specific behavior (distillation rules, schemas).
        description: One-paragraph context for what this project is about. Used for human display and (later) semantic project search.
        tags: Optional list of free-text labels for structured filtering, e.g. `["enterprise","emea","priority-account"]`.

    Returns the created project record. Raises if the slug is malformed, the type is invalid, or a project with this id already exists.
    """
    project = storage.create_project(
        project_id=project_id,
        name=name,
        project_type=project_type,
        description=description,
        tags=tags,
    )
    return {"status": "created", **_project_to_dict(project)}


@mcp.tool()
def list_projects(include_archived: bool = False) -> list[dict[str, Any]]:
    """List registered projects, sorted active-first by creation time.

    Each entry includes the current `memory_count` for that project. Pass `include_archived=True` to also see soft-deleted projects.
    """
    return [_project_to_dict(p) for p in storage.list_projects(include_archived=include_archived)]


@mcp.tool()
def archive_project(project_id: str) -> dict[str, Any]:
    """Soft-delete a project (sets `archived_at = NOW()`).

    Memory rows are NOT removed — they stay queryable if you pass the archived project's id explicitly to `recall` or `stats`. Reversible at the DB layer but no MCP tool unarchives yet. Returns `{"status": "not_found_or_already_archived"}` if no active project with this id exists.
    """
    archived = storage.archive_project(project_id)
    if archived is None:
        return {"status": "not_found_or_already_archived", "project_id": project_id}
    return {
        "status": "archived",
        "project_id": archived.project_id,
        "name": archived.name,
        "project_type": archived.project_type,
        "archived_at": archived.archived_at,
    }


@mcp.tool()
def set_active_project(project_id: str) -> dict[str, Any]:
    """Set the active project for this conversation.

    After this call, every subsequent `recall`, `remember`, `forget`, `supersede`, `update_memory`, `stats`, and `list_hot` invocation will default to this project unless the caller passes an explicit `project_id` (or `project_ids` list) override. The active project is forgotten when Claude Desktop quits.

    Validates that the project exists in the registry — pass an unknown id and the call returns `{"status": "not_found"}` without changing state.
    """
    global _active_project_id
    project = storage.get_project(project_id)
    if project is None:
        return {"status": "not_found", "project_id": project_id}
    if project.archived_at:
        return {"status": "archived", "project_id": project_id, "archived_at": project.archived_at}
    _active_project_id = project_id
    return {
        "status": "active",
        "project_id": project.project_id,
        "name": project.name,
        "project_type": project.project_type,
        "memory_count": project.memory_count,
    }


@mcp.tool()
def get_active_project() -> dict[str, Any]:
    """Report which project is currently active for this conversation, plus the effective default if none is set.

    Returns `{"active_project_id": ..., "default_project_id": ..., "effective_project_id": ...}`. The effective id is what every other tool will use if no explicit `project_id` is passed.
    """
    return {
        "active_project_id": _active_project_id,
        "default_project_id": DEFAULT_PROJECT_ID,
        "effective_project_id": _resolve_project_id(None),
    }


if __name__ == "__main__":
    if TRANSPORT != "stdio":
        mcp.settings.host = os.environ.get("DATABRICKS_APP_HOST", "0.0.0.0")
        mcp.settings.port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    mcp.run(transport=TRANSPORT)
