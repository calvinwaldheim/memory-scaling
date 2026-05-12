from __future__ import annotations

"""Manual verification script for memories schema hardening.

Run this after applying migrations/001_schema_hardening.sql to the target
Lakebase endpoint. It validates the live memories schema, exercises duplicate
insert behavior through memory_agent.storage.insert_memory(), and confirms the
retrieval shape for the notebook 04 verification question.
"""

import argparse
import hashlib
import json
import sys
import time
from typing import Callable

import psycopg2

sys.path.append("/Workspace/Users/calvin.waldheim@gmail.com/memory-scaling")

import memory_agent.storage as storage
from memory_agent.agent import answer, retrieve
from memory_agent.config import DEFAULT_PROJECT_ID
from memory_agent.storage import _get_connection

DEFAULT_QUESTION = "What did we discuss about end to end implementation?"
DEFAULT_INSERT_CONTEXT = "Q: hardening verification?\nA: duplicate insert should be ignored"
DEFAULT_INSERT_RULE = "hardening verification"
DEFAULT_INSERT_SOURCE_REF = "schema-hardening-verification"


def fetch_schema_state(connect: Callable[[], psycopg2.extensions.connection]) -> dict[str, object]:
    """Return columns, constraints, and indexes for memories."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = 'memories'
                ORDER BY ordinal_position
                """
            )
            columns = cur.fetchall()
            cur.execute(
                """
                SELECT conname, pg_get_constraintdef(oid)
                FROM pg_constraint
                WHERE conrelid = 'memories'::regclass
                ORDER BY conname
                """
            )
            constraints = cur.fetchall()
            cur.execute(
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = 'memories'
                ORDER BY indexname
                """
            )
            indexes = cur.fetchall()
    return {
        "columns": columns,
        "constraints": constraints,
        "indexes": indexes,
    }


def verify_duplicate_noop(connect: Callable[[], psycopg2.extensions.connection], project_id: str) -> dict[str, object]:
    """Insert the same Q+A twice and verify only one row exists for its hash."""
    original_get_connection = storage._get_connection
    storage._get_connection = connect
    try:
        content_hash = hashlib.md5(DEFAULT_INSERT_CONTEXT.encode()).hexdigest()
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM memories WHERE project_id = %s AND source_ref = %s AND content_hash = %s",
                    (project_id, DEFAULT_INSERT_SOURCE_REF, content_hash),
                )
                conn.commit()

        storage.insert_memory(
            project_id=project_id,
            project_type="product",
            memory_type="episodic",
            scope="organizational",
            domain="verification",
            rule=DEFAULT_INSERT_RULE,
            context=DEFAULT_INSERT_CONTEXT,
            source_ref=DEFAULT_INSERT_SOURCE_REF,
            embedding=[0.0] * 1024,
            quality_score=0.5,
        )
        storage.insert_memory(
            project_id=project_id,
            project_type="product",
            memory_type="episodic",
            scope="organizational",
            domain="verification",
            rule=DEFAULT_INSERT_RULE,
            context=DEFAULT_INSERT_CONTEXT,
            source_ref=DEFAULT_INSERT_SOURCE_REF,
            embedding=[0.0] * 1024,
            quality_score=0.5,
        )

        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM memories WHERE project_id = %s AND content_hash = %s",
                    (project_id, content_hash),
                )
                row_count = cur.fetchone()[0]
        return {"content_hash": content_hash, "row_count": row_count}
    finally:
        storage._get_connection = original_get_connection


def verify_retrieval_shape(connect: Callable[[], psycopg2.extensions.connection], project_id: str, question: str) -> dict[str, object]:
    """Run the notebook 04 verification flow and report retrieval shape and latency."""
    original_get_connection = storage._get_connection
    storage._get_connection = connect
    try:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content_hash FROM memories WHERE project_id = %s AND source_ref = 'agent-interaction' AND rule = %s",
                    (project_id, question),
                )
                preexisting_hashes = {row[0] for row in cur.fetchall()}

        start = time.perf_counter()
        response = answer(question)
        retrieved = retrieve(question)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        content = f"Q: {question}\nA: {response}"
        new_hash = hashlib.md5(content.encode()).hexdigest()
        shape = ["fresh" if hashlib.md5(memory.context.encode()).hexdigest() == new_hash else "pre-existing" for memory in retrieved]
        return {
            "retrieved_count": len(retrieved),
            "retrieval_shape": shape,
            "retrieval_latency_ms": elapsed_ms,
            "new_content_hash_preexisted": new_hash in preexisting_hashes,
        }
    finally:
        storage._get_connection = original_get_connection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    connect = _get_connection

    output = {
        "schema_state": fetch_schema_state(connect),
        "duplicate_insert_check": verify_duplicate_noop(connect, args.project_id),
        "retrieval_check": verify_retrieval_shape(connect, args.project_id, args.question),
    }
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
