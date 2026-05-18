from __future__ import annotations

import sys
import types

from memory_agent.storage import (
    InsertMemoryResult,
    Lineage,
    LineageNode,
    MemoryStats,
    RetrievedMemory,
)


class _FastMCPStub:
    def __init__(self, *_args, **_kwargs) -> None:
        self.settings = types.SimpleNamespace(host="0.0.0.0", port=8000)

    def tool(self):
        def decorator(func):
            return func

        return decorator

    def run(self, *_args, **_kwargs) -> None:
        raise RuntimeError("FastMCP stub should not be run in tests")


class _AccessTokenStub:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _TokenVerifierStub:
    pass


class _AuthSettingsStub:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


# Stub the MCP SDK modules that app.app imports. The real package isn't
# installed in the unit-test environment; FastMCP is the only thing app code
# actually exercises beyond import time.
fastmcp_module = types.ModuleType("mcp.server.fastmcp")
fastmcp_module.FastMCP = _FastMCPStub

auth_provider_module = types.ModuleType("mcp.server.auth.provider")
auth_provider_module.AccessToken = _AccessTokenStub
auth_provider_module.TokenVerifier = _TokenVerifierStub

auth_settings_module = types.ModuleType("mcp.server.auth.settings")
auth_settings_module.AuthSettings = _AuthSettingsStub

auth_module = types.ModuleType("mcp.server.auth")
auth_module.provider = auth_provider_module
auth_module.settings = auth_settings_module

server_module = types.ModuleType("mcp.server")
server_module.fastmcp = fastmcp_module
server_module.auth = auth_module

mcp_module = types.ModuleType("mcp")
mcp_module.server = server_module

sys.modules.setdefault("mcp", mcp_module)
sys.modules.setdefault("mcp.server", server_module)
sys.modules.setdefault("mcp.server.fastmcp", fastmcp_module)
sys.modules.setdefault("mcp.server.auth", auth_module)
sys.modules.setdefault("mcp.server.auth.provider", auth_provider_module)
sys.modules.setdefault("mcp.server.auth.settings", auth_settings_module)

from app.app import (  # noqa: E402
    forget,
    get_lineage,
    recall,
    remember,
    stats,
    supersede,
)


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------


def test_recall_returns_filtered_memory_dicts(monkeypatch) -> None:
    captured = {}

    def fake_retrieve(**kwargs):
        captured.update(kwargs)
        return [
            RetrievedMemory(
                rule="design note",
                context="stored memory",
                quality_score=0.9,
                distance=0.12,
                source_ref="doc-1",
                memory_type="episodic",
                domain="architecture",
                id="11111111-1111-1111-1111-111111111111",
                project_id="memory-kb-poc",
            )
        ]

    monkeypatch.setattr("app.app.memory_agent.retrieve", fake_retrieve)

    result = recall(query="what matters", top_k=2)

    assert captured["question"] == "what matters"
    assert captured["top_k"] == 2
    assert captured["include_inactive"] is False
    assert result[0]["content"] == "stored memory"
    assert result[0]["id"] == "11111111-1111-1111-1111-111111111111"
    assert result[0]["project_id"] == "memory-kb-poc"
    assert result[0]["superseded_at"] is None
    assert result[0]["forgotten_at"] is None


def test_recall_passes_include_inactive(monkeypatch) -> None:
    captured = {}

    def fake_retrieve(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("app.app.memory_agent.retrieve", fake_retrieve)
    recall(query="audit pls", include_inactive=True)
    assert captured["include_inactive"] is True


# ---------------------------------------------------------------------------
# remember
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# forget — soft default + hard escape hatch
# ---------------------------------------------------------------------------


def test_forget_defaults_to_soft(monkeypatch) -> None:
    captured = {}

    def fake_soft_forget(memory_id, reason=None, user_id=None):
        captured["memory_id"] = memory_id
        captured["reason"] = reason
        captured["user_id"] = user_id
        return {"status": "forgotten", "id": memory_id}

    def fake_delete(memory_id):
        raise AssertionError("hard delete should not be called when hard=False")

    monkeypatch.setattr("app.app.storage.soft_forget_memory", fake_soft_forget)
    monkeypatch.setattr("app.app.storage.delete_memory", fake_delete)

    result = forget(memory_id="11111111-1111-1111-1111-111111111111", reason="wrong")
    assert result["status"] == "forgotten"
    assert captured["reason"] == "wrong"


def test_forget_hard_routes_to_delete(monkeypatch) -> None:
    called = {"soft": 0, "hard": 0}

    def fake_soft_forget(*args, **kwargs):
        called["soft"] += 1
        raise AssertionError("soft forget should not be called when hard=True")

    def fake_delete(memory_id):
        called["hard"] += 1
        return {
            "id": memory_id,
            "rule": "r",
            "context": "c",
            "memory_type": "episodic",
            "domain": "d",
            "scope": "org",
            "quality_score": 0.5,
        }

    monkeypatch.setattr("app.app.storage.soft_forget_memory", fake_soft_forget)
    monkeypatch.setattr("app.app.storage.delete_memory", fake_delete)

    result = forget(memory_id="11111111-1111-1111-1111-111111111111", hard=True)
    assert result["status"] == "deleted"
    assert called == {"soft": 0, "hard": 1}


# ---------------------------------------------------------------------------
# supersede
# ---------------------------------------------------------------------------


def test_supersede_embeds_and_delegates(monkeypatch) -> None:
    captured = {}

    def fake_embed(text):
        captured["embed_text"] = text
        return [0.1, 0.2, 0.3]

    def fake_supersede(**kwargs):
        captured.update(kwargs)
        return {
            "status": "superseded",
            "old_id": kwargs["old_id"],
            "new_id": "22222222-2222-2222-2222-222222222222",
            "head_id": "22222222-2222-2222-2222-222222222222",
        }

    monkeypatch.setattr("app.app.embeddings.embed", fake_embed)
    monkeypatch.setattr("app.app.storage.supersede_memory", fake_supersede)

    result = supersede(
        old_id="11111111-1111-1111-1111-111111111111",
        new_content="corrected",
        reason="re-tuned",
        user_id="agent-1",
    )
    assert result["status"] == "superseded"
    assert captured["embed_text"] == "corrected"
    assert captured["embedding"] == [0.1, 0.2, 0.3]
    assert captured["reason"] == "re-tuned"
    assert captured["user_id"] == "agent-1"


def test_supersede_already_superseded_passes_through(monkeypatch) -> None:
    monkeypatch.setattr("app.app.embeddings.embed", lambda t: [0.0])
    monkeypatch.setattr(
        "app.app.storage.supersede_memory",
        lambda **kw: {
            "status": "already_superseded",
            "memory_id": kw["old_id"],
            "current_head_id": "33333333-3333-3333-3333-333333333333",
        },
    )
    result = supersede(
        old_id="11111111-1111-1111-1111-111111111111",
        new_content="x",
        reason="r",
    )
    assert result["status"] == "already_superseded"
    assert result["current_head_id"] == "33333333-3333-3333-3333-333333333333"


# ---------------------------------------------------------------------------
# get_lineage
# ---------------------------------------------------------------------------


def _node(row_id, depth, superseded_by=None, superseded_at=None):
    return LineageNode(
        id=row_id,
        rule="r",
        context="c",
        memory_type="episodic",
        domain="d",
        scope="org",
        quality_score=0.5,
        superseded_by=superseded_by,
        superseded_at=superseded_at,
        superseded_reason=None,
        superseded_by_user=None,
        forgotten_at=None,
        created_at="2026-05-01T00:00:00+00:00",
        depth=depth,
    )


def test_get_lineage_returns_chain_and_ancestors(monkeypatch) -> None:
    a = "11111111-1111-1111-1111-111111111111"
    b = "22222222-2222-2222-2222-222222222222"
    fake_lineage = Lineage(
        target_id=a,
        head_id=b,
        chain=[_node(a, 0, superseded_by=b, superseded_at="2026-05-10T00:00:00+00:00"),
               _node(b, 1)],
        ancestors=[],
    )
    monkeypatch.setattr("app.app.storage.get_lineage", lambda mid: fake_lineage)
    result = get_lineage(memory_id=a)
    assert result["target_id"] == a
    assert result["head_id"] == b
    assert [n["id"] for n in result["chain"]] == [a, b]
    assert result["ancestors"] == []


def test_get_lineage_not_found(monkeypatch) -> None:
    monkeypatch.setattr("app.app.storage.get_lineage", lambda mid: None)
    result = get_lineage(memory_id="00000000-0000-0000-0000-000000000000")
    assert result["status"] == "not_found"


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_returns_summary_shape(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.app.storage.stats",
        lambda project_id: MemoryStats(
            total=4,
            by_memory_type={"episodic": 3, "semantic": 1},
            by_domain={"architecture": 2, "product": 2},
            last_written_at="2025-01-01T00:00:00+00:00",
            active_count=2,
            superseded_count=1,
            forgotten_count=1,
        ),
    )

    result = stats()

    assert result["total"] == 4
    assert result["active_count"] == 2
    assert result["superseded_count"] == 1
    assert result["forgotten_count"] == 1
    assert result["by_memory_type"]["episodic"] == 3
    assert result["by_domain"]["product"] == 2
    assert result["last_written_at"] == "2025-01-01T00:00:00+00:00"
