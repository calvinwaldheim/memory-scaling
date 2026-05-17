from __future__ import annotations

"""Parameterized Lakebase storage helpers for memory_agent."""

import hashlib
import json
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
       source_ref, memory_type, domain
FROM memories
WHERE project_id = %(project_id)s
  AND (%(memory_type)s::text IS NULL OR memory_type = %(memory_type)s)
  AND (%(domain)s::text IS NULL OR domain = %(domain)s)
  AND (%(min_quality_score)s::float IS NULL OR quality_score >= %(min_quality_score)s)
ORDER BY distance ASC
LIMIT %(top_k)s
"""

STATS_SQL = """
WITH scoped AS (
    SELECT memory_type, domain, created_at
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
    MAX(created_at) AS last_written_at
FROM scoped
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


@dataclass(frozen=True)
class InsertMemoryResult:
    """Insert result describing dedupe outcome for a memory write."""

    status: str
    content_hash: str


@dataclass(frozen=True)
class MemoryStats:
    """Aggregated statistics for a project's stored memories."""

    total: int
    by_memory_type: dict[str, int]
    by_domain: dict[str, int]
    last_written_at: str | None


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
    project_id: str = DEFAULT_PROJECT_ID,
    top_k: int = DEFAULT_TOP_K,
    memory_type: str | None = None,
    domain: str | None = None,
    min_quality_score: float | None = None,
) -> list[RetrievedMemory]:
    """Retrieve the top-k similar memories using pgvector cosine distance.

    Args:
        query_embedding: The embedded user query.
        project_id: Project identifier filter for the memories table.
        top_k: Maximum number of rows to return.
        memory_type: Optional exact-match filter on ``memory_type`` (``"episodic"`` or ``"semantic"``).
        domain: Optional exact-match filter on ``domain``.
        min_quality_score: Optional inclusive lower bound on ``quality_score``.

    Returns:
        Retrieved memory rows sorted by ascending cosine distance.

    Raises:
        psycopg2.Error: If the database connection or query fails.
    """
    conn = _get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            RETRIEVE_MEMORIES_SQL,
            {
                "emb": json.dumps(list(query_embedding)),
                "project_id": project_id,
                "top_k": top_k,
                "memory_type": memory_type,
                "domain": domain,
                "min_quality_score": min_quality_score,
            },
        )
        return [RetrievedMemory(*row) for row in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


def stats(project_id: str = DEFAULT_PROJECT_ID) -> MemoryStats:
    """Return aggregate storage statistics for the requested project in one round-trip."""
    conn = _get_connection()
    cur = conn.cursor()
    try:
        cur.execute(STATS_SQL, (project_id,))
        total, by_memory_type, by_domain, last_written_at = cur.fetchone()
        if isinstance(last_written_at, datetime):
            last_written_at = last_written_at.isoformat()
        return MemoryStats(
            total=total or 0,
            by_memory_type=dict(by_memory_type or {}),
            by_domain=dict(by_domain or {}),
            last_written_at=last_written_at,
        )
    finally:
        cur.close()
        conn.close()
