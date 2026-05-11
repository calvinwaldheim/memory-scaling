from __future__ import annotations

import sys
import types

from memory_agent.storage import InsertMemoryResult, MemoryStats, RetrievedMemory


class _FastMCPStub:
    def __init__(self, *_args, **_kwargs) -> None:
        self.settings = types.SimpleNamespace(host="0.0.0.0", port=8000)

    def tool(self):
        def decorator(func):
            return func

        return decorator

    def run(self, *_args, **_kwargs) -> None:
        raise RuntimeError("FastMCP stub should not be run in tests")


fastmcp_module = types.ModuleType("mcp.server.fastmcp")
fastmcp_module.FastMCP = _FastMCPStub
server_module = types.ModuleType("mcp.server")
server_module.fastmcp = fastmcp_module
mcp_module = types.ModuleType("mcp")
mcp_module.server = server_module

sys.modules.setdefault("mcp", mcp_module)
sys.modules.setdefault("mcp.server", server_module)
sys.modules.setdefault("mcp.server.fastmcp", fastmcp_module)

from app.app import recall, remember, stats


def test_recall_returns_filtered_memory_dicts(monkeypatch) -> None:
    def fake_retrieve(question: str, project_id: str, top_k: int):
        assert question == "what matters"
        assert project_id == "memory-kb-poc"
        assert top_k == 2
        return [
            RetrievedMemory(
                rule="design note",
                context="stored memory",
                quality_score=0.9,
                distance=0.12,
                source_ref="doc-1",
                memory_type="episodic",
                domain="architecture",
            )
        ]

    monkeypatch.setattr("app.app.memory_agent.retrieve", fake_retrieve)

    result = recall(query="what matters", top_k=2)

    assert result == [
        {
            "content": "stored memory",
            "source_ref": "doc-1",
            "memory_type": "episodic",
            "domain": "architecture",
            "rule": "design note",
            "quality_score": 0.9,
            "distance": 0.12,
        }
    ]


def test_remember_embeds_and_stores(monkeypatch) -> None:
    monkeypatch.setattr("app.app.embeddings.embed", lambda text: [0.1, 0.2, 0.3])

    captured = {}

    def fake_insert_memory(**kwargs):
        captured.update(kwargs)
        return InsertMemoryResult(status="stored", content_hash="abc123")

    monkeypatch.setattr("app.app.storage.insert_memory", fake_insert_memory)

    result = remember(content="new memory", source_ref="chat-1", domain="product")

    assert result == {"status": "stored", "content_hash": "abc123"}
    assert captured["project_type"] == "product"
    assert captured["context"] == "new memory"
    assert captured["source_ref"] == "chat-1"
    assert captured["embedding"] == [0.1, 0.2, 0.3]


def test_stats_returns_summary_shape(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.app.storage.stats",
        lambda project_id: MemoryStats(
            total=4,
            by_memory_type={"episodic": 3, "semantic": 1},
            by_domain={"architecture": 2, "product": 2},
            last_written_at="2025-01-01T00:00:00+00:00",
        ),
    )

    result = stats()

    assert result["total"] == 4
    assert result["by_memory_type"]["episodic"] == 3
    assert result["by_domain"]["product"] == 2
    assert result["last_written_at"] == "2025-01-01T00:00:00+00:00"
