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
) -> list[dict[str, Any]]:
    """Retrieve the nearest stored memories for a query.

    Call this when an agent needs grounded context from the memory store before answering or planning. The return value is a ranked list of memory dictionaries. Each item contains `id`, `content`, `source_ref`, `memory_type`, `domain`, `rule`, `quality_score`, `distance`, and `project_id`; lower `distance` means a closer match.

    Project targeting (in precedence order):
    - `project_ids` (list): search across multiple projects in one call, e.g. recalling across every customer-type project.
    - `project_id` (single string): explicit override for one project.
    - Active project (set via `set_active_project`).
    - `DEFAULT_PROJECT_ID` env var.

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
def forget(memory_id: str) -> dict[str, Any]:
    """Delete one memory by id.

    Use this when a memory is wrong, obsolete, or off-scope and should no longer be retrievable. The id comes from a prior `recall` result. The return value is the deleted row's fields (`id`, `rule`, `content`, `memory_type`, `domain`, `scope`, `quality_score`) so the caller can confirm what was removed; if no row matched the id, the response is `{"status": "not_found", "memory_id": "..."}`.

    To correct a memory's content rather than discard it, call `forget` followed by `remember` with the new content — the embedding has to be recomputed for retrieval to find the new wording.
    """
    deleted = storage.delete_memory(memory_id)
    if deleted is None:
        return {"status": "not_found", "memory_id": memory_id}
    return {"status": "deleted", **deleted}


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

    The return value includes inventory fields (`total`, `by_memory_type`, `by_domain`, `first_written_at`, `last_written_at`), usage fields driven by `retrieval_count` (`cold_count` — memories never retrieved, `retrieval_count_p50` / `_p90` / `_max`), and a quality summary (`avg_quality_score`). All timestamps are ISO strings or `None`. A high `cold_count` relative to `total` is a signal that pruning would be productive.
    """
    summary = storage.stats(project_id=_resolve_project_id(project_id))
    return {
        "total": summary.total,
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

    After this call, every subsequent `recall`, `remember`, `forget`, `update_memory`, `stats`, and `list_hot` invocation will default to this project unless the caller passes an explicit `project_id` (or `project_ids` list) override. The active project is forgotten when Claude Desktop quits.

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
