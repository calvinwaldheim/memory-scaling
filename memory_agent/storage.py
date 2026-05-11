from __future__ import annotations

"""Parameterized Lakebase storage helpers for memory_agent."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

import psycopg2
from psycopg2.extensions import connection as PGConnection

from .config import (
    DEFAULT_BOOTSTRAP_SOURCE_REF,
    DEFAULT_PROJECT_ID,
    DEFAULT_TOP_K,
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
       embedding <=> %s::vector AS distance
FROM memories
WHERE project_id = %s
ORDER BY distance ASC
LIMIT %s
"""


@dataclass(frozen=True)
class RetrievedMemory:
    """A retrieved memory row returned by pgvector similarity search."""

    rule: str
    context: str
    quality_score: float | None
    distance: float


def _get_connection() -> PGConnection:
    """Create a psycopg2 connection to Lakebase using the configured secret URI."""
    return psycopg2.connect(get_lakebase_uri())


def insert_memory(
    project_id: str,
    project_type: str,
    memory_type: str,
    scope: str,
    domain: str,
    rule: str,
    context: str,
    source_ref: str,
    embedding: Sequence[float],
    quality_score: float,
) -> None:
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
        None.

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
) -> list[RetrievedMemory]:
    """Retrieve the top-k similar memories using pgvector cosine distance.

    Args:
        query_embedding: The embedded user query.
        project_id: Project identifier filter for the memories table.
        top_k: Maximum number of rows to return.

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
            (json.dumps(list(query_embedding)), project_id, top_k),
        )
        return [RetrievedMemory(*row) for row in cur.fetchall()]
    finally:
        cur.close()
        conn.close()
