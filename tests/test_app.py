from __future__ import annotations

import sys
import types
from contextlib import contextmanager

import pytest

from memory_agent.storage import (
    AuditLogEntry,
    InsertMemoryResult,
    Lineage,
    LineageNode,
    MemoryStats,
    Project,
    ProjectAccess,
    RetrievedMemory,
)


# ---------------------------------------------------------------------------
# MCP SDK stubs. The real package isn't installed in the unit-test env.
# We stub: server.fastmcp.FastMCP, auth.provider.{AccessToken,TokenVerifier},
# auth.settings.AuthSettings, and auth.middleware.auth_context.get_access_token.
# ---------------------------------------------------------------------------


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
        for k, v in kwargs.items():
            setattr(self, k, v)


class _TokenVerifierStub:
    pass


class _AuthSettingsStub:
    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


# A per-test contextvar would be cleaner; for now a single module-level slot
# the test fixture below sets and clears.
_fake_access_token: dict[str, object] = {"current": None}


def _fake_get_access_token():
    return _fake_access_token["current"]


# Wire the stubs into sys.modules BEFORE importing app.app.
fastmcp_module = types.ModuleType("mcp.server.fastmcp")
fastmcp_module.FastMCP = _FastMCPStub

auth_provider_module = types.ModuleType("mcp.server.auth.provider")
auth_provider_module.AccessToken = _AccessTokenStub
auth_provider_module.TokenVerifier = _TokenVerifierStub

auth_settings_module = types.ModuleType("mcp.server.auth.settings")
auth_settings_module.AuthSettings = _AuthSettingsStub

auth_context_module = types.ModuleType("mcp.server.auth.middleware.auth_context")
auth_context_module.get_access_token = _fake_get_access_token

auth_middleware_module = types.ModuleType("mcp.server.auth.middleware")
auth_middleware_module.auth_context = auth_context_module

auth_module = types.ModuleType("mcp.server.auth")
auth_module.provider = auth_provider_module
auth_module.settings = auth_settings_module
auth_module.middleware = auth_middleware_module

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
sys.modules.setdefault("mcp.server.auth.middleware", auth_middleware_module)
sys.modules.setdefault("mcp.server.auth.middleware.auth_context", auth_context_module)

# Force stdio transport so app.app skips the production auth wiring path.
import os  # noqa: E402

os.environ.setdefault("MCP_TRANSPORT", "stdio")

from app import app as app_module  # noqa: E402
from app.app import (  # noqa: E402
    archive_project,
    create_project,
    find_user,
    forget,
    get_active_project,
    get_audit_log,
    get_lineage,
    grant_access,
    list_access,
    list_hot,
    list_projects,
    recall,
    remember,
    revoke_access,
    set_active_project,
    stats,
    supersede,
    update_memory,
)


@contextmanager
def _as_user(user_name: str):
    """Set the fake authenticated identity for the wrapped block."""
    prev = _fake_access_token["current"]
    _fake_access_token["current"] = _AccessTokenStub(user_name=user_name)
    try:
        yield
    finally:
        _fake_access_token["current"] = prev


@pytest.fixture(autouse=True)
def _reset_active_project_state():
    """Each test starts with an empty per-user active-project dict."""
    app_module._active_project_by_user.clear()
    yield
    app_module._active_project_by_user.clear()


# ---------------------------------------------------------------------------
# Identity + authorization helpers
# ---------------------------------------------------------------------------


def test_current_user_reads_from_token() -> None:
    with _as_user("alice@example.com"):
        assert app_module._current_user() == "alice@example.com"


def test_authorize_passes_when_role_sufficient(monkeypatch) -> None:
    monkeypatch.setattr("app.app.storage.get_user_role", lambda p, u: "owner")
    with _as_user("alice@example.com"):
        app_module._authorize("proj", "contributor")  # owner satisfies contributor


def test_authorize_raises_when_role_missing(monkeypatch) -> None:
    monkeypatch.setattr("app.app.storage.get_user_role", lambda p, u: None)
    with _as_user("bob@example.com"):
        with pytest.raises(PermissionError, match=r"needs role >= 'viewer'"):
            app_module._authorize("proj", "viewer")


def test_authorize_raises_when_role_too_low(monkeypatch) -> None:
    monkeypatch.setattr("app.app.storage.get_user_role", lambda p, u: "viewer")
    with _as_user("carol@example.com"):
        with pytest.raises(PermissionError, match=r"needs role >= 'contributor'"):
            app_module._authorize("proj", "contributor")


# ---------------------------------------------------------------------------
# Per-user active project isolation
# ---------------------------------------------------------------------------


def test_active_project_is_per_user(monkeypatch) -> None:
    # Both users have viewer access to their respective projects.
    fake_project = Project(
        project_id="anything", name="n", project_type="product",
        description=None, tags=[], created_at=None, created_by=None,
        archived_at=None, memory_count=0,
    )
    monkeypatch.setattr("app.app.storage.get_project", lambda pid: fake_project._replace(project_id=pid) if hasattr(fake_project, "_replace") else fake_project)
    # Project dataclass is frozen; rebuild per-call instead of using _replace.
    monkeypatch.setattr(
        "app.app.storage.get_project",
        lambda pid: Project(
            project_id=pid, name="n", project_type="product",
            description=None, tags=[], created_at=None, created_by=None,
            archived_at=None, memory_count=0,
        ),
    )
    monkeypatch.setattr("app.app.storage.get_user_role", lambda p, u: "viewer")

    with _as_user("alice@example.com"):
        set_active_project("proj-a")
    with _as_user("bob@example.com"):
        set_active_project("proj-b")

    # Alice still sees proj-a even after bob switched.
    with _as_user("alice@example.com"):
        assert app_module._resolve_project_id(None) == "proj-a"
    with _as_user("bob@example.com"):
        assert app_module._resolve_project_id(None) == "proj-b"


# ---------------------------------------------------------------------------
# remember
# ---------------------------------------------------------------------------


def test_remember_uses_token_identity_as_created_by(monkeypatch) -> None:
    monkeypatch.setattr("app.app.storage.get_user_role", lambda p, u: "contributor")
    monkeypatch.setattr("app.app.embeddings.embed", lambda t: [0.1, 0.2, 0.3])
    captured = {}

    def fake_insert(**kwargs):
        captured.update(kwargs)
        return InsertMemoryResult(status="stored", content_hash="abc")

    monkeypatch.setattr("app.app.storage.insert_memory", fake_insert)

    with _as_user("alice@example.com"):
        result = remember(content="new fact", source_ref="chat-1")
    assert result["status"] == "stored"
    # created_by came from the verified token, not from any caller param.
    assert captured["created_by"] == "alice@example.com"


def test_remember_requires_contributor(monkeypatch) -> None:
    monkeypatch.setattr("app.app.storage.get_user_role", lambda p, u: "viewer")
    monkeypatch.setattr("app.app.embeddings.embed", lambda t: [0.1])
    with _as_user("bob@example.com"):
        with pytest.raises(PermissionError):
            remember(content="x", source_ref="s")


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------


def test_forget_soft_requires_contributor(monkeypatch) -> None:
    monkeypatch.setattr("app.app.storage.get_memory_project", lambda mid: "proj-a")
    monkeypatch.setattr("app.app.storage.get_user_role", lambda p, u: "viewer")  # below contributor
    with _as_user("bob@example.com"):
        with pytest.raises(PermissionError):
            forget(memory_id="11111111-1111-1111-1111-111111111111")


def test_forget_hard_requires_owner(monkeypatch) -> None:
    monkeypatch.setattr("app.app.storage.get_memory_project", lambda mid: "proj-a")
    monkeypatch.setattr("app.app.storage.get_user_role", lambda p, u: "contributor")  # below owner
    with _as_user("bob@example.com"):
        with pytest.raises(PermissionError):
            forget(memory_id="11111111-1111-1111-1111-111111111111", hard=True)


def test_forget_soft_happy_path(monkeypatch) -> None:
    monkeypatch.setattr("app.app.storage.get_memory_project", lambda mid: "proj-a")
    monkeypatch.setattr("app.app.storage.get_user_role", lambda p, u: "contributor")
    captured = {}

    def fake_soft(memory_id, reason, user_id):
        captured["memory_id"] = memory_id
        captured["reason"] = reason
        captured["user_id"] = user_id
        return {"status": "forgotten", "id": memory_id, "forgotten_reason": reason}

    monkeypatch.setattr("app.app.storage.soft_forget_memory", fake_soft)
    with _as_user("alice@example.com"):
        result = forget(memory_id="11111111-1111-1111-1111-111111111111", reason="wrong")
    assert result["status"] == "forgotten"
    assert captured["user_id"] == "alice@example.com"  # actor from token, not from caller


# ---------------------------------------------------------------------------
# supersede
# ---------------------------------------------------------------------------


def test_supersede_uses_token_identity(monkeypatch) -> None:
    monkeypatch.setattr("app.app.storage.get_memory_project", lambda mid: "proj-a")
    monkeypatch.setattr("app.app.storage.get_user_role", lambda p, u: "contributor")
    monkeypatch.setattr("app.app.embeddings.embed", lambda t: [0.1])
    captured = {}

    def fake_supersede(**kwargs):
        captured.update(kwargs)
        return {"status": "superseded", "old_id": kwargs["old_id"], "new_id": "x", "head_id": "x"}

    monkeypatch.setattr("app.app.storage.supersede_memory", fake_supersede)
    with _as_user("alice@example.com"):
        result = supersede(old_id="11111111-1111-1111-1111-111111111111", new_content="x", reason="r")
    assert result["status"] == "superseded"
    assert captured["user_id"] == "alice@example.com"


# ---------------------------------------------------------------------------
# recall — ACL filtering on single + multi project queries
# ---------------------------------------------------------------------------


def test_recall_single_project_no_access_returns_empty(monkeypatch) -> None:
    monkeypatch.setattr("app.app.storage.get_user_role", lambda p, u: None)  # no access
    with _as_user("bob@example.com"):
        result = recall(query="anything", project_id="proj-secret")
    assert result == []


def test_recall_multi_project_intersects_with_acl(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.app.storage.accessible_projects_for", lambda u: {"proj-a", "proj-b"}
    )
    captured = {}

    def fake_retrieve(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("app.app.memory_agent.retrieve", fake_retrieve)
    with _as_user("alice@example.com"):
        recall(query="x", project_ids=["proj-a", "proj-b", "proj-secret"])
    # proj-secret was filtered out; proj-a + proj-b survived.
    assert set(captured["project_ids"]) == {"proj-a", "proj-b"}


def test_recall_multi_project_empty_intersection_skips_retrieve(monkeypatch) -> None:
    monkeypatch.setattr("app.app.storage.accessible_projects_for", lambda u: set())

    called = {"n": 0}

    def fake_retrieve(**kwargs):
        called["n"] += 1
        return []

    monkeypatch.setattr("app.app.memory_agent.retrieve", fake_retrieve)
    with _as_user("bob@example.com"):
        result = recall(query="x", project_ids=["proj-a", "proj-b"])
    assert result == []
    assert called["n"] == 0  # never called the storage layer at all


# ---------------------------------------------------------------------------
# create_project — auto-grants owner
# ---------------------------------------------------------------------------


def test_create_project_auto_grants_owner(monkeypatch) -> None:
    fake_project = Project(
        project_id="proj-new", name="New", project_type="product",
        description=None, tags=[], created_at="2026-05-18T00:00:00+00:00",
        created_by="alice@example.com", archived_at=None, memory_count=0,
    )
    monkeypatch.setattr("app.app.storage.create_project", lambda **kw: fake_project)
    granted = {}

    def fake_grant(project_id, user_name, role, granted_by):
        granted["project_id"] = project_id
        granted["user_name"] = user_name
        granted["role"] = role
        granted["granted_by"] = granted_by
        return ProjectAccess(
            project_id=project_id, user_name=user_name, role=role,
            granted_at=None, granted_by=granted_by,
        )

    monkeypatch.setattr("app.app.storage.grant_project_access", fake_grant)
    with _as_user("alice@example.com"):
        result = create_project(
            project_id="proj-new", name="New", project_type="product",
        )
    assert result["status"] == "created"
    assert granted == {
        "project_id": "proj-new",
        "user_name": "alice@example.com",
        "role": "owner",
        "granted_by": "alice@example.com",
    }


# ---------------------------------------------------------------------------
# list_projects — ACL filtered
# ---------------------------------------------------------------------------


def test_list_projects_filters_by_acl(monkeypatch) -> None:
    projects = [
        Project("proj-a", "A", "product", None, [], None, None, None, 0),
        Project("proj-b", "B", "product", None, [], None, None, None, 0),
        Project("proj-secret", "S", "product", None, [], None, None, None, 0),
    ]
    monkeypatch.setattr("app.app.storage.list_projects", lambda include_archived: projects)
    monkeypatch.setattr(
        "app.app.storage.accessible_projects_for", lambda u: {"proj-a", "proj-b"}
    )
    with _as_user("alice@example.com"):
        result = list_projects()
    assert {p["project_id"] for p in result} == {"proj-a", "proj-b"}


# ---------------------------------------------------------------------------
# grant/revoke/list_access
# ---------------------------------------------------------------------------


def test_grant_access_requires_owner(monkeypatch) -> None:
    monkeypatch.setattr("app.app.storage.get_user_role", lambda p, u: "contributor")
    with _as_user("bob@example.com"):
        with pytest.raises(PermissionError):
            grant_access(project_id="p", user_name="x", role="viewer")


def test_grant_access_delegates_to_storage(monkeypatch) -> None:
    monkeypatch.setattr("app.app.storage.get_user_role", lambda p, u: "owner")
    captured = {}

    def fake_grant(project_id, user_name, role, granted_by):
        captured["granted_by"] = granted_by
        return ProjectAccess(
            project_id=project_id, user_name=user_name, role=role,
            granted_at="2026-05-18T00:00:00+00:00", granted_by=granted_by,
        )

    monkeypatch.setattr("app.app.storage.grant_project_access", fake_grant)
    with _as_user("alice@example.com"):
        result = grant_access(project_id="p", user_name="bob@example.com", role="contributor")
    assert result["status"] == "granted"
    assert result["role"] == "contributor"
    assert captured["granted_by"] == "alice@example.com"


def test_revoke_access_requires_owner(monkeypatch) -> None:
    monkeypatch.setattr("app.app.storage.get_user_role", lambda p, u: "contributor")
    with _as_user("bob@example.com"):
        with pytest.raises(PermissionError):
            revoke_access(project_id="p", user_name="alice@example.com")


def test_list_access_returns_full_acl(monkeypatch) -> None:
    monkeypatch.setattr("app.app.storage.get_user_role", lambda p, u: "viewer")
    monkeypatch.setattr("app.app.storage.list_project_access", lambda pid: [
        ProjectAccess("p", "alice@example.com", "owner", None, None),
        ProjectAccess("p", "bob@example.com", "viewer", None, None),
    ])
    with _as_user("alice@example.com"):
        result = list_access(project_id="p")
    assert {r["user_name"] for r in result} == {"alice@example.com", "bob@example.com"}


# ---------------------------------------------------------------------------
# find_user (SCIM directory lookup)
# ---------------------------------------------------------------------------


def _scim_resource(user_name: str, display_name: str, active: bool = True) -> dict:
    """SCIM /Users-shaped dict matching what Databricks returns."""
    return {
        "userName": user_name,
        "displayName": display_name,
        "active": active,
    }


def test_find_user_returns_matching_users(monkeypatch) -> None:
    captured: dict = {}

    def fake_scim(query: str, limit: int):
        captured["query"] = query
        captured["limit"] = limit
        return [
            _scim_resource("alice@example.com", "Alice Jones"),
            _scim_resource("alice.smith@example.com", "Alice Smith"),
        ]

    monkeypatch.setattr("app.app._scim_search_users", fake_scim)
    with _as_user("caller@example.com"):
        result = find_user(query="alice")
    assert captured["query"] == "alice"
    assert captured["limit"] == 10  # default
    assert [r["user_name"] for r in result] == [
        "alice@example.com",
        "alice.smith@example.com",
    ]
    assert result[0]["display_name"] == "Alice Jones"
    assert result[0]["active"] is True


def test_find_user_clamps_limit(monkeypatch) -> None:
    captured: dict = {}

    def fake_scim(query: str, limit: int):
        captured["limit"] = limit
        return []

    monkeypatch.setattr("app.app._scim_search_users", fake_scim)
    with _as_user("caller@example.com"):
        find_user(query="x", limit=999)
    assert captured["limit"] == 50  # upper clamp

    with _as_user("caller@example.com"):
        find_user(query="x", limit=0)
    assert captured["limit"] == 1  # lower clamp


def test_find_user_empty_query_short_circuits(monkeypatch) -> None:
    called = {"n": 0}

    def fake_scim(query: str, limit: int):
        called["n"] += 1
        return []

    monkeypatch.setattr("app.app._scim_search_users", fake_scim)
    with _as_user("caller@example.com"):
        assert find_user(query="") == []
        assert find_user(query="   ") == []
    assert called["n"] == 0  # never hit the network


def test_find_user_falls_back_to_user_name_when_display_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.app._scim_search_users",
        lambda q, n: [{"userName": "bob@example.com", "active": True}],
    )
    with _as_user("caller@example.com"):
        result = find_user(query="bob")
    assert result == [
        {"user_name": "bob@example.com", "display_name": "bob@example.com", "active": True}
    ]


def test_find_user_skips_resources_without_user_name(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.app._scim_search_users",
        lambda q, n: [
            {"displayName": "ghost"},  # no userName — skip
            _scim_resource("real@example.com", "Real Person"),
        ],
    )
    with _as_user("caller@example.com"):
        result = find_user(query="x")
    assert [r["user_name"] for r in result] == ["real@example.com"]


# ---------------------------------------------------------------------------
# get_audit_log
# ---------------------------------------------------------------------------


def test_get_audit_log_requires_viewer(monkeypatch) -> None:
    monkeypatch.setattr("app.app.storage.get_memory_project", lambda mid: "proj-a")
    monkeypatch.setattr("app.app.storage.get_user_role", lambda p, u: None)
    with _as_user("bob@example.com"):
        with pytest.raises(PermissionError):
            get_audit_log(memory_id="11111111-1111-1111-1111-111111111111")


def test_get_audit_log_returns_entries(monkeypatch) -> None:
    monkeypatch.setattr("app.app.storage.get_memory_project", lambda mid: "proj-a")
    monkeypatch.setattr("app.app.storage.get_user_role", lambda p, u: "viewer")
    monkeypatch.setattr(
        "app.app.storage.get_memory_audit_log",
        lambda mid, limit: [AuditLogEntry(
            id="aaa", memory_id=mid, project_id="proj-a", action="superseded",
            actor="alice@example.com", reason="fixed",
            before_state={"rule": "old"}, after_state={"superseded_by": "x"},
            created_at="2026-05-18T00:00:00+00:00",
        )],
    )
    with _as_user("alice@example.com"):
        result = get_audit_log(memory_id="11111111-1111-1111-1111-111111111111")
    assert len(result["entries"]) == 1
    assert result["entries"][0]["action"] == "superseded"
    assert result["entries"][0]["actor"] == "alice@example.com"


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_returns_summary_shape(monkeypatch) -> None:
    monkeypatch.setattr("app.app.storage.get_user_role", lambda p, u: "viewer")
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
    with _as_user("alice@example.com"):
        result = stats(project_id="proj-a")
    assert result["total"] == 4
    assert result["active_count"] == 2
    assert result["superseded_count"] == 1
    assert result["forgotten_count"] == 1
