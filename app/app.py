from __future__ import annotations

"""FastMCP server exposing the memory-scaling POC tools as a Databricks App."""

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from memory_agent import agent as memory_agent
from memory_agent import embeddings, storage

mcp = FastMCP("memory-scaling")


@mcp.tool()
def recall(
    query: str,
    top_k: int = 3,
    project_id: str = "memory-kb-poc",
) -> list[dict[str, Any]]:
    """Retrieve the nearest stored memories for a query.

    Call this when an agent needs grounded context from the memory store before answering or planning. The return value is a ranked list of memory dictionaries. Each item contains only `content`, `source_ref`, `memory_type`, `domain`, `rule`, `quality_score`, and `distance`; lower `distance` means a closer match.
    """
    memories = memory_agent.retrieve(question=query, project_id=project_id, top_k=top_k)
    return [
        {
            "content": memory.context,
            "source_ref": memory.source_ref,
            "memory_type": memory.memory_type,
            "domain": memory.domain,
            "rule": memory.rule,
            "quality_score": memory.quality_score,
            "distance": memory.distance,
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
    project_id: str = "memory-kb-poc",
    quality_score: float = 0.5,
) -> dict[str, str]:
    """Store one new memory item in the project knowledge base.

    Call this when an agent has produced or observed durable context worth retaining across future sessions. The tool embeds `content` inline, writes through the memory storage layer with v0 internal defaults, and returns `{"status": "stored" | "duplicate", "content_hash": "..."}` so callers can tell whether a new row was written or deduplicated by the database.
    """
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
def stats(project_id: str = "memory-kb-poc") -> dict[str, Any]:
    """Summarize the current contents of the memory store for one project.

    Call this when an agent needs a quick health or inventory check before relying on the memory base. The return value is `{"total": N, "by_memory_type": {...}, "by_domain": {...}, "last_written_at": "..."}` where `last_written_at` is an ISO timestamp string or `None` if the project has no memories yet.
    """
    summary = storage.stats(project_id=project_id)
    return {
        "total": summary.total,
        "by_memory_type": summary.by_memory_type,
        "by_domain": summary.by_domain,
        "last_written_at": summary.last_written_at,
    }


if __name__ == "__main__":
    mcp.settings.host = os.environ.get("DATABRICKS_APP_HOST", "0.0.0.0")
    mcp.settings.port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    mcp.run(transport="streamable-http")
