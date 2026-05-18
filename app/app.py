from __future__ import annotations

"""FastMCP server exposing the memory-scaling POC tools as a Databricks App.

Pass A adds three layers of multi-user safety on top of the original toolset:

1. **Verified-token identity.** Every authenticated request carries a
   ``DatabricksAccessToken`` whose ``user_name`` field was looked up via SCIM
   /Me at verification time. Tools read it via ``_current_user()`` and pass
   it to the storage layer; caller-supplied actor identifiers are no longer
   accepted (they were trust-based and forgeable).

2. **Project ACL gating.** ``_authorize(project_id, role)`` runs at the top
   of every tool. The required role per tool is documented in app/README.md.
   A caller without ``viewer`` on a project gets ``[]`` from reads or a
   ``PermissionError`` from writes — never silently shared data.

3. **Per-connection active project.** Replaced the process-wide
   ``_active_project_id`` global with ``_active_project_by_user`` so two
   users sharing one MCP subprocess can't clobber each other's session state.
"""

import os
from typing import Any

import httpx
from mcp.server.auth.middleware.auth_context import get_access_token
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

TRANSPORT = os.environ.get("MCP_TRANSPORT", "streamable-http")

# Local dev fallback when running under stdio with no auth verifier wired up.
# Production transports (streamable-http) never hit this — every request must
# pass the token verifier first, and the verifier populates a real user_name.
LOCAL_USER = os.environ.get("LOCAL_USER", "local-dev")

# Default project used when no active project has been set and no project_id
# was passed to the tool. Overridable per server instance via the
# DEFAULT_PROJECT_ID env var in claude_desktop_config.json.
DEFAULT_PROJECT_ID = os.environ.get("DEFAULT_PROJECT_ID", "memory-kb-poc")

# Per-user active project. Keyed by the verified user_name so concurrent users
# on the same MCP subprocess no longer overwrite each other's session state.
# In-memory only — restarts wipe it, matching today's UX.
_active_project_by_user: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Auth: verified identity carried through every tool call
# ---------------------------------------------------------------------------


class DatabricksAccessToken(AccessToken):
    """Extension of the SDK's AccessToken that carries the verified user_name.

    The base AccessToken has client_id / scopes / expires_at but no user subject.
    We need the actual Databricks identity for audit attribution and ACL checks,
    so we fetch it from SCIM /Me during verify and stash it here.
    """

    user_name: str


class DatabricksTokenVerifier(TokenVerifier):
    async def verify_token(self, token: str) -> DatabricksAccessToken | None:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{WORKSPACE_URL}/api/2.0/preview/scim/v2/Me",
                headers={"Authorization": f"Bearer {token}"},
            )
        if r.status_code != 200:
            return None
        user_name = r.json().get("userName")
        if not user_name:
            # SCIM /Me succeeded but didn't return a userName — refuse the call
            # rather than fall back to an anonymous identity.
            return None
        return DatabricksAccessToken(
            token=token,
            client_id="claude",
            scopes=["postgres"],
            user_name=user_name,
        )


def _current_user() -> str:
    """Return the verified user_name for this request.

    In production transports the auth middleware always populated a token before
    the tool ran; if it didn't, that's a routing bug — fail loud rather than
    silently downgrade.

    The stdio transport has no verifier, so we fall back to the LOCAL_USER env
    var (default ``"local-dev"``). This path must never run in production —
    streamable-http requires the verifier, so a misconfigured prod deploy
    refuses requests rather than reaching this branch.
    """
    tok = get_access_token()
    if tok is None:
        if TRANSPORT == "stdio":
            return LOCAL_USER
        raise PermissionError("No authenticated user on this request.")
    user_name = getattr(tok, "user_name", None)
    if not user_name:
        raise PermissionError("Authenticated token is missing user_name.")
    return user_name


ROLE_RANK = {"viewer": 1, "contributor": 2, "owner": 3}


def _authorize(project_id: str, role: str) -> None:
    """Raise PermissionError if the current user lacks the required role on project_id."""
    user = _current_user()
    granted = storage.get_user_role(project_id, user)
    required = ROLE_RANK[role]
    if granted is None or ROLE_RANK[granted] < required:
        raise PermissionError(
            f"User {user!r} needs role >= {role!r} on project {project_id!r} "
            f"(has: {granted!r})"
        )


def _resolve_project_id(explicit: str | None) -> str:
    """Pick the project_id for a single-project tool call.

    Precedence: explicit caller arg > caller's active project > DEFAULT_PROJECT_ID env var.
    Never returns None. Does NOT enforce ACL — every tool is responsible for
    calling ``_authorize`` after resolving.
    """
    if explicit:
        return explicit
    user = _current_user()
    if user in _active_project_by_user:
        return _active_project_by_user[user]
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
            # Databricks Apps OAuth retired `all-apis` and `offline_access` in favor of
            # fine-grained per-resource scopes. Our app only needs `postgres` (for
            # Lakebase credential minting via /api/2.0/postgres/credentials); SCIM
            # /Me self-info works on any valid token regardless of scope.
            required_scopes=["postgres"],
        ),
    )


# ---------------------------------------------------------------------------
# Memory operations
# ---------------------------------------------------------------------------


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

    Returns memories the caller has at least ``viewer`` access to. For multi-project
    queries (``project_ids=[...]``), the list is silently intersected with the caller's
    accessible projects — projects the caller can't see contribute no memories rather
    than raising. For single-project queries, an inaccessible project returns ``[]``.

    Each result contains ``id``, ``content``, ``source_ref``, ``memory_type``, ``domain``,
    ``rule``, ``quality_score``, ``distance``, ``project_id``, plus lineage fields
    ``superseded_at``, ``superseded_by``, ``forgotten_at``, and authorship ``created_by``.
    Lower ``distance`` means a closer match.

    Default behavior **excludes** rows that have been superseded or soft-forgotten.
    Pass ``include_inactive=True`` to also surface historical rows (audit use).

    Side effect: each returned row's ``retrieval_count`` is incremented so future
    consolidation/pruning can prefer hot memories.
    """
    user = _current_user()
    if project_ids:
        accessible = storage.accessible_projects_for(user)
        targets = [pid for pid in project_ids if pid in accessible]
    else:
        resolved = _resolve_project_id(project_id)
        # Single-project: ACL miss → empty result (consistent with "you can't see it"
        # rather than "we'll tell you it exists by raising").
        if storage.get_user_role(resolved, user) is None:
            return []
        targets = [resolved]

    if not targets:
        return []

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
            "created_by": memory.created_by,
        }
        for memory in memories
    ]


@mcp.tool()
def remember(
    content: str,
    source_ref: str,
    memory_type: str = "episodic",
    domain: str | None = None,
    rule: str | None = None,
    project_id: str | None = None,
    quality_score: float = 0.5,
) -> dict[str, str]:
    """Store one new memory item in the project knowledge base.

    Requires ``contributor`` access to the target project. The verified user
    identity is stored as ``created_by`` on the row and as ``actor`` on the
    accompanying audit-log entry — caller-supplied attribution is not accepted.

    Returns ``{"status": "stored" | "duplicate", "content_hash": "..."}``.
    """
    project_id = _resolve_project_id(project_id)
    _authorize(project_id, "contributor")
    user = _current_user()
    embedding = embeddings.embed(content)
    result = storage.insert_memory(
        project_id=project_id,
        project_type="product",
        memory_type=memory_type,
        domain=domain,
        rule=rule,
        context=content,
        source_ref=source_ref,
        embedding=embedding,
        quality_score=quality_score,
        created_by=user,
    )
    return {"status": result.status, "content_hash": result.content_hash}


@mcp.tool()
def forget(
    memory_id: str,
    reason: str | None = None,
    hard: bool = False,
) -> dict[str, Any]:
    """Retract one memory. Soft by default — the row stays in the table for audit.

    Permissions:
      - ``hard=False`` (soft retraction, default): requires ``contributor``.
      - ``hard=True`` (destructive purge): requires ``owner``.

    The verified user is the actor on both the row and the audit-log entry.
    """
    user = _current_user()
    project_id = storage.get_memory_project(memory_id)
    if project_id is None:
        return {"status": "not_found", "memory_id": memory_id}
    if hard:
        _authorize(project_id, "owner")
        deleted = storage.delete_memory(memory_id, actor=user, reason=reason)
        if deleted is None:
            return {"status": "not_found", "memory_id": memory_id}
        return {"status": "deleted", **deleted}
    _authorize(project_id, "contributor")
    return storage.soft_forget_memory(memory_id, reason=reason, user_id=user)


@mcp.tool()
def supersede(
    old_id: str,
    new_content: str,
    reason: str,
    rule: str | None = None,
    quality_score: float | None = None,
    source_ref: str | None = None,
    memory_type: str | None = None,
    domain: str | None = None,
) -> dict[str, Any]:
    """Atomically replace a memory with a corrected version, preserving lineage.

    Requires ``contributor`` access to the memory's project. The verified user
    is stored as ``created_by`` on the new row and as ``actor`` on both audit-log
    entries (``created`` for the new memory, ``superseded`` for the old).

    The new memory inherits ``project_id``, ``project_type``, ``memory_type``, and
    ``domain`` from the old row unless explicitly overridden.

    Returns one of: ``superseded`` / ``already_superseded`` / ``forgotten`` /
    ``duplicate_content`` / ``not_found``.
    """
    user = _current_user()
    project_id = storage.get_memory_project(old_id)
    if project_id is None:
        return {"status": "not_found", "memory_id": old_id}
    _authorize(project_id, "contributor")
    embedding = embeddings.embed(new_content)
    return storage.supersede_memory(
        old_id=old_id,
        new_content=new_content,
        embedding=embedding,
        reason=reason,
        user_id=user,
        rule=rule,
        quality_score=quality_score,
        source_ref=source_ref,
        memory_type=memory_type,
        domain=domain,
    )


@mcp.tool()
def update_memory(
    memory_id: str,
    rule: str | None = None,
    domain: str | None = None,
    quality_score: float | None = None,
    memory_type: str | None = None,
) -> dict[str, Any]:
    """Update lightweight metadata on one memory by id. Requires ``contributor``.

    ``content`` is deliberately not updatable here because changing it would invalidate
    the stored embedding and content_hash. To rewrite content, call ``supersede``.

    Writes one ``updated`` audit-log entry inside the same transaction.
    """
    user = _current_user()
    project_id = storage.get_memory_project(memory_id)
    if project_id is None:
        return {"status": "not_found", "memory_id": memory_id}
    _authorize(project_id, "contributor")

    fields: dict[str, Any] = {}
    if rule is not None:
        fields["rule"] = rule
    if domain is not None:
        fields["domain"] = domain
    if quality_score is not None:
        fields["quality_score"] = quality_score
    if memory_type is not None:
        fields["memory_type"] = memory_type
    if not fields:
        return {"status": "no_change", "memory_id": memory_id}
    updated = storage.update_memory_fields(memory_id, actor=user, **fields)
    if updated is None:
        return {"status": "not_found", "memory_id": memory_id}
    return {"status": "updated", **updated}


@mcp.tool()
def get_lineage(memory_id: str) -> dict[str, Any]:
    """Return the supersede chain and ancestor tree for one memory. Requires ``viewer``."""
    project_id = storage.get_memory_project(memory_id)
    if project_id is None:
        return {"status": "not_found", "memory_id": memory_id}
    _authorize(project_id, "viewer")
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
def get_audit_log(memory_id: str, limit: int = 50) -> dict[str, Any]:
    """Return the audit history for one memory, newest first. Requires ``viewer``.

    Each entry contains ``id``, ``memory_id``, ``project_id``, ``action``, ``actor``,
    ``reason``, ``before_state`` (JSON), ``after_state`` (JSON), ``created_at``.
    ``action`` is one of ``created`` / ``superseded`` / ``forgotten`` / ``purged`` / ``updated``.
    """
    project_id = storage.get_memory_project(memory_id)
    if project_id is None:
        # Memory may have been hard-purged. Try to find the project from the audit log
        # itself by reading what we can.
        entries = storage.get_memory_audit_log(memory_id, limit=limit)
        if not entries:
            return {"status": "not_found", "memory_id": memory_id}
        # Authorize against the project recorded in the earliest entry.
        _authorize(entries[-1].project_id, "viewer")
    else:
        _authorize(project_id, "viewer")
        entries = storage.get_memory_audit_log(memory_id, limit=limit)
    return {
        "memory_id": memory_id,
        "entries": [
            {
                "id": e.id,
                "memory_id": e.memory_id,
                "project_id": e.project_id,
                "action": e.action,
                "actor": e.actor,
                "reason": e.reason,
                "before_state": e.before_state,
                "after_state": e.after_state,
                "created_at": e.created_at,
            }
            for e in entries
        ],
    }


@mcp.tool()
def stats(project_id: str | None = None) -> dict[str, Any]:
    """Project health snapshot. Requires ``viewer``."""
    project_id = _resolve_project_id(project_id)
    _authorize(project_id, "viewer")
    summary = storage.stats(project_id=project_id)
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
    """List the most-retrieved memories in a project. Requires ``viewer``."""
    project_id = _resolve_project_id(project_id)
    _authorize(project_id, "viewer")
    return storage.list_hot_memories(project_id=project_id, top_k=top_k)


# ---------------------------------------------------------------------------
# Project management
# ---------------------------------------------------------------------------


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
    """Register a new project. The creator becomes ``owner`` automatically.

    Any authenticated user can create a project. The project is private to the
    creator until they share it via ``grant_access``.
    """
    user = _current_user()
    project = storage.create_project(
        project_id=project_id,
        name=name,
        project_type=project_type,
        description=description,
        tags=tags,
        created_by=user,
    )
    storage.grant_project_access(project_id, user, "owner", granted_by=user)
    return {"status": "created", **_project_to_dict(project)}


@mcp.tool()
def list_projects(include_archived: bool = False) -> list[dict[str, Any]]:
    """List projects the caller has at least ``viewer`` access to.

    Pass ``include_archived=True`` to also see soft-deleted projects you have access to.
    Projects you can't see are silently omitted — no errors.
    """
    user = _current_user()
    accessible = storage.accessible_projects_for(user)
    return [
        _project_to_dict(p)
        for p in storage.list_projects(include_archived=include_archived)
        if p.project_id in accessible
    ]


@mcp.tool()
def archive_project(project_id: str) -> dict[str, Any]:
    """Soft-delete a project. Requires ``owner``.

    Memory rows are NOT removed — they stay queryable if you pass the archived
    project's id explicitly to ``recall`` or ``stats`` (and still have ACL access).
    """
    _authorize(project_id, "owner")
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
    """Set the active project for *this user* for the rest of the conversation.

    After this call, every subsequent tool call from this user that doesn't pass
    an explicit ``project_id`` defaults to this one. Concurrent users on the same
    MCP subprocess have independent active projects.

    Requires ``viewer`` on the target project (you can't activate something you can't read).
    The active project is forgotten when the MCP subprocess restarts.
    """
    user = _current_user()
    project = storage.get_project(project_id)
    if project is None:
        return {"status": "not_found", "project_id": project_id}
    if project.archived_at:
        return {"status": "archived", "project_id": project_id, "archived_at": project.archived_at}
    _authorize(project_id, "viewer")
    _active_project_by_user[user] = project_id
    return {
        "status": "active",
        "project_id": project.project_id,
        "name": project.name,
        "project_type": project.project_type,
        "memory_count": project.memory_count,
    }


@mcp.tool()
def get_active_project() -> dict[str, Any]:
    """Report which project is currently active for *this user*."""
    user = _current_user()
    active = _active_project_by_user.get(user)
    return {
        "active_project_id": active,
        "default_project_id": DEFAULT_PROJECT_ID,
        "effective_project_id": _resolve_project_id(None),
    }


# ---------------------------------------------------------------------------
# Access control (ACL)
# ---------------------------------------------------------------------------


@mcp.tool()
def grant_access(project_id: str, user_name: str, role: str) -> dict[str, Any]:
    """Grant or upgrade a user's role on a project. Requires ``owner``.

    Role must be one of ``viewer``, ``contributor``, ``owner``. Re-granting the same
    user upserts to the new role.
    """
    _authorize(project_id, "owner")
    granter = _current_user()
    access = storage.grant_project_access(project_id, user_name, role, granted_by=granter)
    return {
        "status": "granted",
        "project_id": access.project_id,
        "user_name": access.user_name,
        "role": access.role,
        "granted_at": access.granted_at,
        "granted_by": access.granted_by,
    }


@mcp.tool()
def revoke_access(project_id: str, user_name: str) -> dict[str, Any]:
    """Revoke a user's access to a project. Requires ``owner``."""
    _authorize(project_id, "owner")
    revoked = storage.revoke_project_access(project_id, user_name)
    if revoked is None:
        return {"status": "not_found", "project_id": project_id, "user_name": user_name}
    return {
        "status": "revoked",
        "project_id": revoked.project_id,
        "user_name": revoked.user_name,
        "role": revoked.role,
    }


@mcp.tool()
def list_access(project_id: str) -> list[dict[str, Any]]:
    """List every (user, role) pair on a project. Requires ``viewer``."""
    _authorize(project_id, "viewer")
    return [
        {
            "project_id": a.project_id,
            "user_name": a.user_name,
            "role": a.role,
            "granted_at": a.granted_at,
            "granted_by": a.granted_by,
        }
        for a in storage.list_project_access(project_id)
    ]


if __name__ == "__main__":
    if TRANSPORT != "stdio":
        mcp.settings.host = os.environ.get("DATABRICKS_APP_HOST", "0.0.0.0")
        mcp.settings.port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    mcp.run(transport=TRANSPORT)
