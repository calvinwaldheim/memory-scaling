from __future__ import annotations

"""Parameterized Lakebase storage helpers for memory_agent."""

import hashlib
import json
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import psycopg2
import requests
from databricks.sdk import WorkspaceClient
from psycopg2.extensions import connection as PGConnection

from .config import (
    DEFAULT_BOOTSTRAP_SOURCE_REF,
    DEFAULT_PROJECT_ID,
    DEFAULT_TOP_K,
    get_lakebase_project_name,
    get_lakebase_uri,
)

INSERT_MEMORY_SQL = """
INSERT INTO memories
    (project_id, project_type, memory_type, scope, domain, rule, context, source_ref, content_hash, embedding, quality_score)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT DO NOTHING
"""

RETRIEVE_MEMORIES_SQL = """
SELECT rule, context, quality_score,
       embedding <=> %(emb)s::vector AS distance,
       source_ref, memory_type, domain, id, project_id,
       superseded_at, superseded_by, forgotten_at
FROM memories
WHERE project_id = ANY(%(project_ids)s::text[])
  AND (%(memory_type)s::text IS NULL OR memory_type = %(memory_type)s)
  AND (%(domain)s::text IS NULL OR domain = %(domain)s)
  AND (%(min_quality_score)s::float IS NULL OR quality_score >= %(min_quality_score)s)
  AND (%(include_inactive)s::boolean OR (superseded_at IS NULL AND forgotten_at IS NULL))
ORDER BY distance ASC
LIMIT %(top_k)s
"""

BUMP_RETRIEVAL_COUNT_SQL = """
UPDATE memories
SET retrieval_count = COALESCE(retrieval_count, 0) + 1
WHERE id = ANY(%(ids)s::uuid[])
"""

DELETE_MEMORY_SQL = """
DELETE FROM memories
WHERE id = %(id)s::uuid
RETURNING id, rule, context, memory_type, domain, scope, quality_score
"""

# Soft-forget: mark a memory as retracted without removing the row. Audit fields
# stay populated so we can answer "what did we believe on date X, and when did
# we change our minds". Only flips rows that are currently active; the caller
# disambiguates "not_found" vs "already_forgotten" via a follow-up SELECT.
SOFT_FORGET_MEMORY_SQL = """
UPDATE memories
SET forgotten_at = NOW(),
    forgotten_reason = %(reason)s,
    forgotten_by_user = %(user_id)s,
    updated_at = NOW()
WHERE id = %(id)s::uuid
  AND forgotten_at IS NULL
RETURNING id, rule, context, memory_type, domain, scope, quality_score,
          forgotten_at, forgotten_reason, forgotten_by_user
"""

# Read-only probe used when SOFT_FORGET_MEMORY_SQL updates zero rows: tells us
# whether the id is unknown or already forgotten.
GET_MEMORY_FORGOTTEN_STATE_SQL = """
SELECT id, forgotten_at, forgotten_reason, forgotten_by_user
FROM memories
WHERE id = %(id)s::uuid
"""

# Row-lock the old memory before we decide whether to supersede it.
LOCK_MEMORY_FOR_SUPERSEDE_SQL = """
SELECT id, project_id, project_type, memory_type, scope, user_id, domain,
       rule, superseded_at, superseded_by, forgotten_at
FROM memories
WHERE id = %(id)s::uuid
FOR UPDATE
"""

# Insert the replacement memory inside the same transaction as the old row's
# lock. Mirrors INSERT_MEMORY_SQL but RETURNS the new id so we can patch the
# old row's superseded_by FK in the same transaction. ``derived_from`` is
# nullable — populated by the distillation pipeline when a semantic memory
# supersedes a previous semantic for the same cluster.
INSERT_SUPERSEDE_MEMORY_SQL = """
INSERT INTO memories
    (project_id, project_type, memory_type, scope, user_id, domain, rule,
     context, source_ref, content_hash, embedding, quality_score, derived_from)
VALUES
    (%(project_id)s, %(project_type)s, %(memory_type)s, %(scope)s, %(user_id)s,
     %(domain)s, %(rule)s, %(context)s, %(source_ref)s, %(content_hash)s,
     %(embedding)s, %(quality_score)s, %(derived_from)s::uuid[])
ON CONFLICT (project_id, content_hash) DO NOTHING
RETURNING id
"""

MARK_SUPERSEDED_SQL = """
UPDATE memories
SET superseded_by = %(new_id)s::uuid,
    superseded_at = NOW(),
    superseded_reason = %(reason)s,
    superseded_by_user = %(user_id)s,
    updated_at = NOW()
WHERE id = %(old_id)s::uuid
RETURNING id, superseded_at
"""

# Walk forward from the given id to the chain head (the row with
# superseded_at IS NULL is the live one). Postgres recursive CTE.
LINEAGE_FORWARD_SQL = """
WITH RECURSIVE chain AS (
    SELECT id, superseded_by, superseded_at, superseded_reason, superseded_by_user,
           forgotten_at, rule, context, memory_type, domain, scope, quality_score,
           created_at, 0 AS depth
    FROM memories
    WHERE id = %(id)s::uuid
  UNION ALL
    SELECT m.id, m.superseded_by, m.superseded_at, m.superseded_reason, m.superseded_by_user,
           m.forgotten_at, m.rule, m.context, m.memory_type, m.domain, m.scope, m.quality_score,
           m.created_at, chain.depth + 1
    FROM memories m
    JOIN chain ON m.id = chain.superseded_by
    WHERE chain.depth < 64
)
SELECT id, superseded_by, superseded_at, superseded_reason, superseded_by_user,
       forgotten_at, rule, context, memory_type, domain, scope, quality_score,
       created_at, depth
FROM chain
ORDER BY depth ASC
"""

# Walk backward: every row that (transitively) points its superseded_by at the
# given id is an ancestor. Used for "what beliefs did this one replace".
LINEAGE_BACKWARD_SQL = """
WITH RECURSIVE ancestors AS (
    SELECT id, superseded_by, superseded_at, superseded_reason, superseded_by_user,
           forgotten_at, rule, context, memory_type, domain, scope, quality_score,
           created_at, 1 AS depth
    FROM memories
    WHERE superseded_by = %(id)s::uuid
  UNION ALL
    SELECT m.id, m.superseded_by, m.superseded_at, m.superseded_reason, m.superseded_by_user,
           m.forgotten_at, m.rule, m.context, m.memory_type, m.domain, m.scope, m.quality_score,
           m.created_at, ancestors.depth + 1
    FROM memories m
    JOIN ancestors ON m.superseded_by = ancestors.id
    WHERE ancestors.depth < 64
)
SELECT id, superseded_by, superseded_at, superseded_reason, superseded_by_user,
       forgotten_at, rule, context, memory_type, domain, scope, quality_score,
       created_at, depth
FROM ancestors
ORDER BY depth ASC
"""

# Fields that update_memory_fields() is allowed to overwrite. `context` is
# intentionally excluded because changing it would invalidate the stored
# embedding and content_hash — model that as supersede instead.
UPDATABLE_FIELDS = ("rule", "domain", "quality_score", "memory_type", "scope")

PROJECT_TYPES = ("data_domain", "engineering", "compliance", "customer", "product")
PROJECT_ID_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")

CREATE_PROJECT_SQL = """
INSERT INTO projects (project_id, name, project_type, description, tags, created_by)
VALUES (%(project_id)s, %(name)s, %(project_type)s, %(description)s, %(tags)s, %(created_by)s)
RETURNING project_id, name, project_type, description, tags, created_at, created_by, archived_at
"""

LIST_PROJECTS_SQL = """
SELECT p.project_id, p.name, p.project_type, p.description, p.tags,
       p.created_at, p.created_by, p.archived_at,
       COALESCE(c.memory_count, 0) AS memory_count
FROM projects p
LEFT JOIN (
    SELECT project_id, COUNT(*) AS memory_count
    FROM memories
    GROUP BY project_id
) c ON c.project_id = p.project_id
WHERE %(include_archived)s OR p.archived_at IS NULL
ORDER BY p.archived_at NULLS FIRST, p.created_at ASC
"""

GET_PROJECT_SQL = """
SELECT p.project_id, p.name, p.project_type, p.description, p.tags,
       p.created_at, p.created_by, p.archived_at,
       (SELECT COUNT(*) FROM memories WHERE project_id = p.project_id) AS memory_count
FROM projects p
WHERE p.project_id = %(project_id)s
"""

ARCHIVE_PROJECT_SQL = """
UPDATE projects
SET archived_at = NOW()
WHERE project_id = %(project_id)s AND archived_at IS NULL
RETURNING project_id, name, project_type, archived_at
"""

STATS_SQL = """
WITH scoped AS (
    SELECT memory_type, domain, created_at, retrieval_count, quality_score,
           superseded_at, forgotten_at
    FROM memories
    WHERE project_id = %s
),
by_type AS (
    SELECT COALESCE(jsonb_object_agg(memory_type, count_value), '{}'::jsonb) AS value
    FROM (
        SELECT memory_type, COUNT(*) AS count_value
        FROM scoped
        GROUP BY memory_type
    ) grouped
),
by_domain AS (
    SELECT COALESCE(jsonb_object_agg(domain_key, count_value), '{}'::jsonb) AS value
    FROM (
        SELECT COALESCE(domain, 'unknown') AS domain_key, COUNT(*) AS count_value
        FROM scoped
        GROUP BY COALESCE(domain, 'unknown')
    ) grouped
)
SELECT
    COUNT(*) AS total,
    (SELECT value FROM by_type) AS by_memory_type,
    (SELECT value FROM by_domain) AS by_domain,
    MAX(created_at) AS last_written_at,
    MIN(created_at) AS first_written_at,
    COUNT(*) FILTER (WHERE COALESCE(retrieval_count, 0) = 0) AS cold_count,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY COALESCE(retrieval_count, 0)) AS retrieval_count_p50,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY COALESCE(retrieval_count, 0)) AS retrieval_count_p90,
    MAX(retrieval_count) AS retrieval_count_max,
    AVG(quality_score) AS avg_quality_score,
    COUNT(*) FILTER (WHERE superseded_at IS NOT NULL) AS superseded_count,
    COUNT(*) FILTER (WHERE forgotten_at IS NOT NULL) AS forgotten_count,
    COUNT(*) FILTER (WHERE superseded_at IS NULL AND forgotten_at IS NULL) AS active_count
FROM scoped
"""

LIST_HOT_MEMORIES_SQL = """
SELECT id, rule, context, memory_type, domain, quality_score, retrieval_count, created_at
FROM memories
WHERE project_id = %(project_id)s
  AND COALESCE(retrieval_count, 0) > 0
ORDER BY retrieval_count DESC, created_at DESC
LIMIT %(top_k)s
"""


@dataclass(frozen=True)
class RetrievedMemory:
    """A retrieved memory row returned by pgvector similarity search."""

    rule: str | None
    context: str
    quality_score: float | None
    distance: float
    source_ref: str | None = None
    memory_type: str | None = None
    domain: str | None = None
    id: str | None = None
    project_id: str | None = None
    superseded_at: str | None = None
    superseded_by: str | None = None
    forgotten_at: str | None = None


@dataclass(frozen=True)
class LineageNode:
    """One row in a supersede/forget audit chain."""

    id: str
    rule: str | None
    context: str
    memory_type: str | None
    domain: str | None
    scope: str | None
    quality_score: float | None
    superseded_by: str | None
    superseded_at: str | None
    superseded_reason: str | None
    superseded_by_user: str | None
    forgotten_at: str | None
    created_at: str | None
    depth: int


@dataclass(frozen=True)
class Lineage:
    """Full supersede history for one target memory."""

    target_id: str
    head_id: str | None
    chain: list[LineageNode]
    ancestors: list[LineageNode]


@dataclass(frozen=True)
class InsertMemoryResult:
    """Insert result describing dedupe outcome for a memory write."""

    status: str
    content_hash: str


@dataclass(frozen=True)
class Project:
    """Metadata for one project in the registry."""

    project_id: str
    name: str
    project_type: str
    description: str | None
    tags: list[str]
    created_at: str | None
    created_by: str | None
    archived_at: str | None
    memory_count: int | None = None


@dataclass(frozen=True)
class MemoryStats:
    """Aggregated statistics for a project's stored memories."""

    total: int
    by_memory_type: dict[str, int]
    by_domain: dict[str, int]
    last_written_at: str | None
    first_written_at: str | None = None
    cold_count: int = 0
    retrieval_count_p50: float | None = None
    retrieval_count_p90: float | None = None
    retrieval_count_max: int | None = None
    avg_quality_score: float | None = None
    superseded_count: int = 0
    forgotten_count: int = 0
    active_count: int = 0


def _lakebase_endpoint(project_name: str) -> str:
    """Return the default autoscaling endpoint resource path for a Lakebase project."""
    return f"projects/{project_name}/branches/production/endpoints/primary"


def _get_connection() -> PGConnection:
    """Create a psycopg2 connection to Lakebase using a runtime postgres credential."""
    workspace = WorkspaceClient()
    credential = requests.post(
        f"{workspace.config.host.rstrip('/')}/api/2.0/postgres/credentials",
        headers={
            **workspace.config.authenticate(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={
            "endpoint": _lakebase_endpoint(get_lakebase_project_name()),
            "request_id": str(uuid.uuid4()),
        },
        timeout=30,
    )
    credential.raise_for_status()
    token = credential.json()["token"]
    return psycopg2.connect(get_lakebase_uri(), password=token)


def insert_memory(
    project_id: str,
    project_type: str,
    memory_type: str,
    scope: str,
    domain: str | None,
    rule: str | None,
    context: str,
    source_ref: str,
    embedding: Sequence[float],
    quality_score: float,
) -> InsertMemoryResult:
    """Insert one memory row using the hardened schema defaults.

    Args:
        project_id: Project identifier stored in the memories table.
        project_type: Project type value stored with the memory.
        memory_type: Memory type such as episodic or semantic.
        scope: Scope value stored with the memory.
        domain: Domain value stored with the memory.
        rule: Rule summary stored with the memory.
        context: Full memory content.
        source_ref: Source identifier for the inserted memory.
        embedding: Embedding vector serialized for pgvector.
        quality_score: Quality score stored with the memory.

    Returns:
        Insert metadata including duplicate detection status and content hash.

    Raises:
        psycopg2.Error: If the database connection or insert fails.
    """
    content_hash = hashlib.md5(context.encode()).hexdigest()
    conn = _get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            INSERT_MEMORY_SQL,
            (
                project_id,
                project_type,
                memory_type,
                scope,
                domain,
                rule,
                context,
                source_ref,
                content_hash,
                json.dumps(list(embedding)),
                quality_score,
            ),
        )
        conn.commit()
        status = "stored" if cur.rowcount == 1 else "duplicate"
        return InsertMemoryResult(status=status, content_hash=content_hash)
    finally:
        cur.close()
        conn.close()


def store_bootstrap_memories(
    chunks: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    project_id: str = DEFAULT_PROJECT_ID,
    source_ref: str = DEFAULT_BOOTSTRAP_SOURCE_REF,
) -> int:
    """Insert bootstrapped chunk memories using the notebook's original fields.

    Args:
        chunks: Ordered text chunks to insert.
        embeddings: Ordered embeddings aligned one-to-one with chunks.
        project_id: Project identifier stored in the memories table.
        source_ref: Source identifier written with each inserted chunk.

    Returns:
        The number of chunk and embedding pairs iterated for insertion.

    Raises:
        psycopg2.Error: If the database connection or inserts fail.
    """
    conn = _get_connection()
    cur = conn.cursor()
    stored = 0
    try:
        for chunk, embedding in zip(chunks, embeddings):
            content_hash = hashlib.md5(chunk.encode()).hexdigest()
            cur.execute(
                INSERT_MEMORY_SQL,
                (
                    project_id,
                    "product",
                    "episodic",
                    "organizational",
                    "architecture",
                    chunk[:100],
                    chunk,
                    source_ref,
                    content_hash,
                    json.dumps(list(embedding)),
                    0.8,
                ),
            )
            stored += 1
        conn.commit()
    finally:
        cur.close()
        conn.close()
    return stored


def retrieve_memories(
    query_embedding: Sequence[float],
    project_id: str | None = None,
    project_ids: Sequence[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
    memory_type: str | None = None,
    domain: str | None = None,
    min_quality_score: float | None = None,
    include_inactive: bool = False,
) -> list[RetrievedMemory]:
    """Retrieve the top-k similar memories using pgvector cosine distance.

    Exactly one of ``project_id`` or ``project_ids`` should be provided. If both are
    omitted, defaults to ``DEFAULT_PROJECT_ID``; if both are passed, ``project_ids`` wins.

    Args:
        query_embedding: The embedded user query.
        project_id: Single project filter (most common case).
        project_ids: Multiple-project filter for cross-project recall.
        top_k: Maximum number of rows to return.
        memory_type: Optional exact-match filter on ``memory_type`` (``"episodic"`` or ``"semantic"``).
        domain: Optional exact-match filter on ``domain``.
        min_quality_score: Optional inclusive lower bound on ``quality_score``.
        include_inactive: When True, also return rows that are superseded or soft-forgotten.
            Default False keeps retrieval focused on the live believed-true subset; flip to True
            for audit queries that need to inspect history.

    Returns:
        Retrieved memory rows sorted by ascending cosine distance. Each row carries
        ``project_id`` so callers can attribute when querying across projects.

    Raises:
        psycopg2.Error: If the database connection or query fails.
    """
    if project_ids:
        targets = list(project_ids)
    elif project_id:
        targets = [project_id]
    else:
        targets = [DEFAULT_PROJECT_ID]

    conn = _get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            RETRIEVE_MEMORIES_SQL,
            {
                "emb": json.dumps(list(query_embedding)),
                "project_ids": targets,
                "top_k": top_k,
                "memory_type": memory_type,
                "domain": domain,
                "min_quality_score": min_quality_score,
                "include_inactive": include_inactive,
            },
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()
    return [
        RetrievedMemory(
            rule=row[0],
            context=row[1],
            quality_score=row[2],
            distance=row[3],
            source_ref=row[4],
            memory_type=row[5],
            domain=row[6],
            id=str(row[7]) if row[7] is not None else None,
            project_id=row[8],
            superseded_at=row[9].isoformat() if isinstance(row[9], datetime) else row[9],
            superseded_by=str(row[10]) if row[10] is not None else None,
            forgotten_at=row[11].isoformat() if isinstance(row[11], datetime) else row[11],
        )
        for row in rows
    ]


def bump_retrieval_counts(memory_ids: Sequence[str]) -> int:
    """Increment ``retrieval_count`` on the given memory ids.

    Args:
        memory_ids: UUID strings identifying the memory rows that were just retrieved.

    Returns:
        The number of rows updated. Callers can ignore this; it's surfaced for tests.

    Raises:
        psycopg2.Error: If the database connection or update fails.
    """
    ids = [str(mid) for mid in memory_ids if mid]
    if not ids:
        return 0
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(BUMP_RETRIEVAL_COUNT_SQL, {"ids": ids})
            updated = cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return updated


def delete_memory(memory_id: str) -> dict | None:
    """Hard-delete one memory row by id (audit-destroying).

    Reserved for the ``hard=True`` opt-in path on ``forget`` and the future
    pruning job that reclaims rows past their retention window. Prefer
    ``soft_forget_memory`` for routine retraction — it keeps the audit trail.

    Args:
        memory_id: UUID string of the row to remove.

    Returns:
        A dict describing the deleted row (``id``, ``rule``, ``context``, ``memory_type``,
        ``domain``, ``scope``, ``quality_score``) or ``None`` if no row matched.

    Raises:
        psycopg2.Error: If the database connection or delete fails.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(DELETE_MEMORY_SQL, {"id": memory_id})
            row = cur.fetchone()
        conn.commit()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "id": str(row[0]),
        "rule": row[1],
        "context": row[2],
        "memory_type": row[3],
        "domain": row[4],
        "scope": row[5],
        "quality_score": row[6],
    }


def soft_forget_memory(
    memory_id: str,
    reason: str | None = None,
    user_id: str | None = None,
) -> dict:
    """Mark one memory as retracted without removing the row.

    The row stays in the database but is excluded from default ``retrieve_memories``
    results, so semantically-similar queries no longer surface it. Audit fields
    (``forgotten_at``, ``forgotten_reason``, ``forgotten_by_user``) record the
    retraction.

    Args:
        memory_id: UUID string of the row to retract.
        reason: Free-text rationale (recommended). Stored verbatim.
        user_id: Attribution string for the actor who retracted it.

    Returns:
        A dict with ``status`` ∈ {``"forgotten"``, ``"already_forgotten"``, ``"not_found"``},
        plus the memory's fields when applicable.

    Raises:
        psycopg2.Error: If the database connection or update fails.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                SOFT_FORGET_MEMORY_SQL,
                {"id": memory_id, "reason": reason, "user_id": user_id},
            )
            row = cur.fetchone()
            if row is not None:
                conn.commit()
                forgotten_at = row[7]
                if isinstance(forgotten_at, datetime):
                    forgotten_at = forgotten_at.isoformat()
                return {
                    "status": "forgotten",
                    "id": str(row[0]),
                    "rule": row[1],
                    "context": row[2],
                    "memory_type": row[3],
                    "domain": row[4],
                    "scope": row[5],
                    "quality_score": row[6],
                    "forgotten_at": forgotten_at,
                    "forgotten_reason": row[8],
                    "forgotten_by_user": row[9],
                }
            # Zero rows updated: either unknown id, or already forgotten.
            cur.execute(GET_MEMORY_FORGOTTEN_STATE_SQL, {"id": memory_id})
            probe = cur.fetchone()
    finally:
        conn.close()
    if probe is None:
        return {"status": "not_found", "memory_id": memory_id}
    forgotten_at = probe[1]
    if isinstance(forgotten_at, datetime):
        forgotten_at = forgotten_at.isoformat()
    return {
        "status": "already_forgotten",
        "id": str(probe[0]),
        "forgotten_at": forgotten_at,
        "forgotten_reason": probe[2],
        "forgotten_by_user": probe[3],
    }


def _walk_to_head(cur, start_id: str) -> tuple[str | None, list[tuple]]:
    """Walk the supersede chain forward from ``start_id`` and return (head_id, rows).

    ``head_id`` is the id of the live row in the chain (its ``superseded_at`` is NULL),
    or None if the starting row itself is unknown.
    """
    cur.execute(LINEAGE_FORWARD_SQL, {"id": start_id})
    rows = cur.fetchall()
    if not rows:
        return None, []
    head_id: str | None = None
    for row in rows:
        if row[2] is None:  # superseded_at IS NULL ⇒ live row
            head_id = str(row[0])
    if head_id is None and rows:
        # Defensive: deepest row in the chain. Real data should always end at a live row,
        # but if a chain was truncated by a hard-purge of the head, fall back to deepest.
        head_id = str(rows[-1][0])
    return head_id, rows


def supersede_memory(
    old_id: str,
    new_content: str,
    embedding: Sequence[float],
    reason: str,
    user_id: str | None = None,
    rule: str | None = None,
    quality_score: float | None = None,
    source_ref: str | None = None,
    memory_type: str | None = None,
    scope: str | None = None,
    domain: str | None = None,
    derived_from: Sequence[str] | None = None,
) -> dict:
    """Atomically replace one memory with a corrected version, preserving lineage.

    The replacement memory inherits ``project_id``, ``project_type``, ``user_id``,
    ``memory_type``, ``scope``, and ``domain`` from the old row unless explicitly
    overridden. The ``superseded_by`` FK on the old row is patched to point at
    the new row in the same transaction, so retrieval never sees an interim state.

    Args:
        old_id: UUID string of the memory being corrected.
        new_content: The corrected content. Will be embedded and stored as a new row.
        embedding: Pre-computed embedding for ``new_content``. Caller (MCP layer) owns
            the embedding model so storage stays embedding-model-agnostic.
        reason: Free-text rationale (required — supersedence without a reason is
            indistinguishable from churn in audit reports).
        user_id: Attribution string for the actor who superseded.
        rule, quality_score, source_ref, memory_type, scope, domain: Optional
            per-field overrides on the new row. Unset fields inherit from the old row.

    Returns:
        A dict with ``status`` and supporting fields. Possible statuses:
        - ``"superseded"`` — success; includes ``old_id``, ``new_id``, ``head_id``.
        - ``"not_found"`` — ``old_id`` doesn't exist.
        - ``"already_superseded"`` — old row was already replaced; includes
          ``current_head_id`` so the caller can re-read and decide whether to
          re-supersede the head.
        - ``"forgotten"`` — old row was soft-forgotten; caller should call
          ``remember`` instead of trying to chain off a retracted belief.
        - ``"duplicate_content"`` — the new content_hash collides with an existing
          row in the same project. Caller should reword or forget the colliding row.

    Raises:
        psycopg2.Error: If the database connection or any statement fails. The
            transaction is rolled back automatically on failure.
    """
    content_hash = hashlib.md5(new_content.encode()).hexdigest()
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(LOCK_MEMORY_FOR_SUPERSEDE_SQL, {"id": old_id})
            locked = cur.fetchone()
            if locked is None:
                conn.rollback()
                return {"status": "not_found", "memory_id": old_id}

            (
                _,
                old_project_id,
                old_project_type,
                old_memory_type,
                old_scope,
                old_user_id,
                old_domain,
                old_rule,
                old_superseded_at,
                _old_superseded_by,
                old_forgotten_at,
            ) = locked

            if old_forgotten_at is not None:
                conn.rollback()
                return {"status": "forgotten", "memory_id": old_id}

            if old_superseded_at is not None:
                head_id, _rows = _walk_to_head(cur, old_id)
                conn.rollback()
                return {
                    "status": "already_superseded",
                    "memory_id": old_id,
                    "current_head_id": head_id,
                }

            cur.execute(
                INSERT_SUPERSEDE_MEMORY_SQL,
                {
                    "project_id": old_project_id,
                    "project_type": old_project_type,
                    "memory_type": memory_type if memory_type is not None else old_memory_type,
                    "scope": scope if scope is not None else old_scope,
                    "user_id": old_user_id,
                    "domain": domain if domain is not None else old_domain,
                    "rule": rule if rule is not None else old_rule,
                    "context": new_content,
                    "source_ref": source_ref,
                    "content_hash": content_hash,
                    "embedding": json.dumps(list(embedding)),
                    "quality_score": quality_score if quality_score is not None else 0.5,
                    "derived_from": list(derived_from) if derived_from is not None else None,
                },
            )
            inserted = cur.fetchone()
            if inserted is None:
                conn.rollback()
                return {
                    "status": "duplicate_content",
                    "memory_id": old_id,
                    "content_hash": content_hash,
                }
            new_id = str(inserted[0])

            cur.execute(
                MARK_SUPERSEDED_SQL,
                {
                    "old_id": old_id,
                    "new_id": new_id,
                    "reason": reason,
                    "user_id": user_id,
                },
            )
            marked = cur.fetchone()
            if marked is None:
                conn.rollback()
                return {"status": "not_found", "memory_id": old_id}
            superseded_at = marked[1]
            if isinstance(superseded_at, datetime):
                superseded_at = superseded_at.isoformat()
        conn.commit()
    finally:
        conn.close()
    return {
        "status": "superseded",
        "old_id": old_id,
        "new_id": new_id,
        "head_id": new_id,
        "superseded_at": superseded_at,
        "content_hash": content_hash,
    }


def _lineage_row_to_node(row: tuple) -> LineageNode:
    superseded_at = row[2]
    if isinstance(superseded_at, datetime):
        superseded_at = superseded_at.isoformat()
    forgotten_at = row[5]
    if isinstance(forgotten_at, datetime):
        forgotten_at = forgotten_at.isoformat()
    created_at = row[12]
    if isinstance(created_at, datetime):
        created_at = created_at.isoformat()
    return LineageNode(
        id=str(row[0]),
        superseded_by=str(row[1]) if row[1] is not None else None,
        superseded_at=superseded_at,
        superseded_reason=row[3],
        superseded_by_user=row[4],
        forgotten_at=forgotten_at,
        rule=row[6],
        context=row[7],
        memory_type=row[8],
        domain=row[9],
        scope=row[10],
        quality_score=row[11],
        created_at=created_at,
        depth=int(row[13]),
    )


def get_lineage(memory_id: str) -> Lineage | None:
    """Return the supersede chain and ancestor tree for one memory.

    ``chain`` is the forward walk starting from ``memory_id`` (depth 0) through each
    successive ``superseded_by`` pointer until the live head row. ``ancestors``
    is the recursive backward walk of every row that (transitively) was replaced
    by ``memory_id``.

    Returns ``None`` if ``memory_id`` doesn't exist.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            head_id, forward_rows = _walk_to_head(cur, memory_id)
            if not forward_rows:
                return None
            cur.execute(LINEAGE_BACKWARD_SQL, {"id": memory_id})
            backward_rows = cur.fetchall()
    finally:
        conn.close()
    return Lineage(
        target_id=memory_id,
        head_id=head_id,
        chain=[_lineage_row_to_node(r) for r in forward_rows],
        ancestors=[_lineage_row_to_node(r) for r in backward_rows],
    )


def update_memory_fields(memory_id: str, **fields: object) -> dict | None:
    """Update one memory row's lightweight fields by id.

    Only the fields listed in ``UPDATABLE_FIELDS`` may be changed; passing any
    other key raises ``ValueError``. ``context`` is intentionally not updatable
    because changing it would invalidate the stored embedding and content_hash —
    callers should ``delete_memory`` + re-insert instead.

    Args:
        memory_id: UUID string of the row to update.
        **fields: New values for one or more of ``rule``, ``domain``, ``quality_score``,
            ``memory_type``, ``scope``. Pass ``None`` as a value to set the column to NULL.

    Returns:
        A dict describing the updated row (same shape as ``delete_memory``'s return value)
        or ``None`` if no row matched.

    Raises:
        ValueError: If no updatable fields were provided or unknown keys were passed.
        psycopg2.Error: If the database connection or update fails.
    """
    rejected = set(fields) - set(UPDATABLE_FIELDS)
    if rejected:
        raise ValueError(
            f"Cannot update {sorted(rejected)}; only {list(UPDATABLE_FIELDS)} are allowed."
        )
    if not fields:
        raise ValueError("update_memory_fields requires at least one field to change.")

    # Column names come from the closed UPDATABLE_FIELDS tuple, not user input,
    # so direct string composition is safe. Values are still parameterised.
    set_clause = ", ".join(f"{col} = %({col})s" for col in fields)
    query = (
        f"UPDATE memories SET {set_clause}, updated_at = NOW() "
        "WHERE id = %(id)s::uuid "
        "RETURNING id, rule, context, memory_type, domain, scope, quality_score"
    )
    params = {**fields, "id": memory_id}

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone()
        conn.commit()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "id": str(row[0]),
        "rule": row[1],
        "context": row[2],
        "memory_type": row[3],
        "domain": row[4],
        "scope": row[5],
        "quality_score": row[6],
    }


def stats(project_id: str = DEFAULT_PROJECT_ID) -> MemoryStats:
    """Return aggregate storage statistics for the requested project in one round-trip."""
    conn = _get_connection()
    cur = conn.cursor()
    try:
        cur.execute(STATS_SQL, (project_id,))
        (
            total,
            by_memory_type,
            by_domain,
            last_written_at,
            first_written_at,
            cold_count,
            retrieval_count_p50,
            retrieval_count_p90,
            retrieval_count_max,
            avg_quality_score,
            superseded_count,
            forgotten_count,
            active_count,
        ) = cur.fetchone()
        if isinstance(last_written_at, datetime):
            last_written_at = last_written_at.isoformat()
        if isinstance(first_written_at, datetime):
            first_written_at = first_written_at.isoformat()
        return MemoryStats(
            total=total or 0,
            by_memory_type=dict(by_memory_type or {}),
            by_domain=dict(by_domain or {}),
            last_written_at=last_written_at,
            first_written_at=first_written_at,
            cold_count=int(cold_count or 0),
            retrieval_count_p50=float(retrieval_count_p50) if retrieval_count_p50 is not None else None,
            retrieval_count_p90=float(retrieval_count_p90) if retrieval_count_p90 is not None else None,
            retrieval_count_max=int(retrieval_count_max) if retrieval_count_max is not None else None,
            avg_quality_score=float(avg_quality_score) if avg_quality_score is not None else None,
            superseded_count=int(superseded_count or 0),
            forgotten_count=int(forgotten_count or 0),
            active_count=int(active_count or 0),
        )
    finally:
        cur.close()
        conn.close()


def list_hot_memories(project_id: str = DEFAULT_PROJECT_ID, top_k: int = 10) -> list[dict]:
    """Return memories sorted by ``retrieval_count`` DESC, then ``created_at`` DESC.

    Skips rows with ``retrieval_count = 0`` so callers see only memories that have
    actually been used. Useful for hot-set inspection and as the data feed for
    a future pruning rule (cold-and-low-quality → archive).

    Args:
        project_id: Project identifier filter.
        top_k: Maximum number of rows to return.

    Returns:
        Each item is a dict with ``id``, ``rule``, ``content``, ``memory_type``,
        ``domain``, ``quality_score``, ``retrieval_count``, ``created_at``.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(LIST_HOT_MEMORIES_SQL, {"project_id": project_id, "top_k": top_k})
            rows = cur.fetchall()
    finally:
        conn.close()
    return [
        {
            "id": str(row[0]),
            "rule": row[1],
            "content": row[2],
            "memory_type": row[3],
            "domain": row[4],
            "quality_score": row[5],
            "retrieval_count": row[6] or 0,
            "created_at": row[7].isoformat() if isinstance(row[7], datetime) else row[7],
        }
        for row in rows
    ]


def _project_row_to_obj(row: tuple, with_count: bool = False) -> Project:
    """Convert a raw projects row into a Project dataclass."""
    created_at = row[5]
    if isinstance(created_at, datetime):
        created_at = created_at.isoformat()
    archived_at = row[7]
    if isinstance(archived_at, datetime):
        archived_at = archived_at.isoformat()
    memory_count = row[8] if with_count and len(row) > 8 else None
    return Project(
        project_id=row[0],
        name=row[1],
        project_type=row[2],
        description=row[3],
        tags=list(row[4] or []),
        created_at=created_at,
        created_by=row[6],
        archived_at=archived_at,
        memory_count=memory_count,
    )


def create_project(
    project_id: str,
    name: str,
    project_type: str,
    description: str | None = None,
    tags: Sequence[str] | None = None,
    created_by: str | None = None,
) -> Project:
    """Register a new project. Validates the slug and project_type at the Python layer
    so callers get a clear error message before the DB-layer CHECK constraint fires.

    Raises:
        ValueError: If ``project_id`` is not a slug, or ``project_type`` is not in PROJECT_TYPES.
        psycopg2.errors.UniqueViolation: If ``project_id`` already exists.
    """
    if not PROJECT_ID_SLUG.match(project_id):
        raise ValueError(
            f"project_id {project_id!r} must match {PROJECT_ID_SLUG.pattern} "
            "(lowercase letters, digits, dashes; must start with letter or digit)."
        )
    if project_type not in PROJECT_TYPES:
        raise ValueError(
            f"project_type {project_type!r} must be one of {list(PROJECT_TYPES)}."
        )
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                CREATE_PROJECT_SQL,
                {
                    "project_id": project_id,
                    "name": name,
                    "project_type": project_type,
                    "description": description,
                    "tags": list(tags or []),
                    "created_by": created_by,
                },
            )
            row = cur.fetchone()
        conn.commit()
    finally:
        conn.close()
    return _project_row_to_obj(row)


def list_projects(include_archived: bool = False) -> list[Project]:
    """Return all projects, sorted active-first by creation time.

    Each Project has its current ``memory_count`` populated via a LEFT JOIN.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(LIST_PROJECTS_SQL, {"include_archived": include_archived})
            rows = cur.fetchall()
    finally:
        conn.close()
    return [_project_row_to_obj(row, with_count=True) for row in rows]


def get_project(project_id: str) -> Project | None:
    """Return one project by id, including ``memory_count``, or None if not found."""
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(GET_PROJECT_SQL, {"project_id": project_id})
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return _project_row_to_obj(row, with_count=True)


def archive_project(project_id: str) -> Project | None:
    """Soft-delete a project by setting ``archived_at = NOW()``.

    Memory rows for the project are NOT deleted — they remain queryable via the
    archived project_id. Returns None if the project doesn't exist or is already archived.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(ARCHIVE_PROJECT_SQL, {"project_id": project_id})
            row = cur.fetchone()
        conn.commit()
    finally:
        conn.close()
    if row is None:
        return None
    archived_at = row[3]
    if isinstance(archived_at, datetime):
        archived_at = archived_at.isoformat()
    return Project(
        project_id=row[0],
        name=row[1],
        project_type=row[2],
        description=None,
        tags=[],
        created_at=None,
        created_by=None,
        archived_at=archived_at,
        memory_count=None,
    )
