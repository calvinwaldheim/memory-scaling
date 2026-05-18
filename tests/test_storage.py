from __future__ import annotations

import pytest

from memory_agent import config, storage


class FakeCursor:
    """A scriptable fake cursor.

    ``fetch_rows`` is used as the canonical reply: ``fetchall`` returns it whole,
    ``fetchone`` returns its first element. For tests that need different results
    on successive calls (e.g. supersede flows multiple SQL statements), pass
    ``script`` — a list of fetchone/fetchall results consumed in order.
    """

    def __init__(self, fetch_rows=None, script=None):
        self.fetch_rows = fetch_rows or []
        self.script = list(script) if script is not None else None
        self.executed = []
        self.closed = False
        self.rowcount = 1
        self.rowcount_script: list[int] | None = None

    def execute(self, query, params=None):
        self.executed.append((query, params))
        if self.rowcount_script:
            self.rowcount = self.rowcount_script.pop(0)

    def fetchall(self):
        if self.script is not None:
            return list(self.script.pop(0) or [])
        return list(self.fetch_rows)

    def fetchone(self):
        if self.script is not None:
            result = self.script.pop(0)
            if isinstance(result, list):
                return result[0] if result else None
            return result
        return self.fetch_rows[0] if self.fetch_rows else None

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class FakeConnection:
    def __init__(self, fetch_rows=None, script=None):
        self.cursor_obj = FakeCursor(fetch_rows=fetch_rows, script=script)
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


def test_get_connection_requests_postgres_credentials(monkeypatch) -> None:
    captured = {}

    class FakeConfig:
        host = "https://dbc.example.com/"

        @staticmethod
        def authenticate():
            return {"Authorization": "Bearer token"}

    class FakeWorkspaceClient:
        config = FakeConfig()

    class FakeResponse:
        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict[str, str]:
            return {"token": "db-token"}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    def fake_connect(uri, password):
        captured["connect_uri"] = uri
        captured["connect_password"] = password
        return "connection"

    monkeypatch.setattr(storage, "WorkspaceClient", lambda: FakeWorkspaceClient())
    monkeypatch.setattr(storage, "get_lakebase_project_name", lambda: "memory-kb-poc")
    monkeypatch.setattr(storage, "get_lakebase_uri", lambda: "postgresql://user@host:5432/db")
    monkeypatch.setattr(storage.requests, "post", fake_post)
    monkeypatch.setattr(storage.psycopg2, "connect", fake_connect)

    conn = storage._get_connection()

    assert conn == "connection"
    assert captured["url"] == "https://dbc.example.com/api/2.0/postgres/credentials"
    assert captured["headers"]["Authorization"] == "Bearer token"
    assert captured["headers"]["Content-Type"] == "application/json"
    assert captured["headers"]["Accept"] == "application/json"
    assert captured["json"]["endpoint"] == "projects/memory-kb-poc/branches/production/endpoints/primary"
    assert isinstance(captured["json"]["request_id"], str)
    assert captured["timeout"] == 30
    assert captured["connect_uri"] == "postgresql://user@host:5432/db"
    assert captured["connect_password"] == "db-token"


def _retrieve_row(
    rule="rule",
    context="context",
    quality_score=0.8,
    distance=0.25,
    source_ref="src",
    memory_type="episodic",
    domain="architecture",
    row_id="11111111-1111-1111-1111-111111111111",
    project_id="memory-kb-poc",
    superseded_at=None,
    superseded_by=None,
    forgotten_at=None,
    created_by="alice@example.com",
):
    return (
        rule, context, quality_score, distance, source_ref, memory_type,
        domain, row_id, project_id, superseded_at, superseded_by, forgotten_at,
        created_by,
    )


def test_retrieve_memories_returns_typed_rows(monkeypatch) -> None:
    conn = FakeConnection(fetch_rows=[_retrieve_row()])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    results = storage.retrieve_memories([0.1, 0.2], project_id="memory-kb-poc", top_k=3)
    assert len(results) == 1
    assert results[0].rule == "rule"
    assert results[0].project_id == "memory-kb-poc"
    assert results[0].superseded_at is None
    query, params = conn.cursor_obj.executed[0]
    assert "LIMIT %(top_k)s" in query
    assert params["project_ids"] == ["memory-kb-poc"]
    assert params["top_k"] == 3
    assert params["include_inactive"] is False
    assert conn.cursor_obj.closed is True
    assert conn.closed is True


def test_retrieve_memories_default_excludes_inactive(monkeypatch) -> None:
    conn = FakeConnection(fetch_rows=[])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    storage.retrieve_memories([0.1, 0.2], project_id="memory-kb-poc")
    query, params = conn.cursor_obj.executed[0]
    assert "superseded_at IS NULL AND forgotten_at IS NULL" in query
    assert params["include_inactive"] is False


def test_retrieve_memories_include_inactive_flips_predicate(monkeypatch) -> None:
    conn = FakeConnection(fetch_rows=[
        _retrieve_row(superseded_at="2026-05-01T00:00:00+00:00",
                      superseded_by="22222222-2222-2222-2222-222222222222"),
    ])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    results = storage.retrieve_memories(
        [0.1, 0.2], project_id="memory-kb-poc", include_inactive=True
    )
    _, params = conn.cursor_obj.executed[0]
    assert params["include_inactive"] is True
    assert results[0].superseded_at == "2026-05-01T00:00:00+00:00"
    assert results[0].superseded_by == "22222222-2222-2222-2222-222222222222"


def test_insert_memory_uses_parameterized_sql(monkeypatch) -> None:
    # INSERT_MEMORY_SQL now RETURNS the new id so we can write the audit row.
    conn = FakeConnection(script=[("aaaa1111-aaaa-1111-aaaa-111111111111",), None])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    storage.insert_memory(
        project_id="memory-kb-poc",
        project_type="product",
        memory_type="episodic",
        domain="interactions",
        rule="question",
        context="Q: hi\nA: hello",
        source_ref="agent-interaction",
        embedding=[0.1, 0.2],
        quality_score=0.9,
        created_by="alice@example.com",
    )
    # Two statements: the INSERT itself, then the audit row.
    assert len(conn.cursor_obj.executed) == 2
    insert_query, insert_params = conn.cursor_obj.executed[0]
    assert "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)" in insert_query
    assert "ON CONFLICT DO NOTHING" in insert_query
    assert insert_params[0] == "memory-kb-poc"
    assert insert_params[2] == "episodic"
    # Last positional is created_by.
    assert insert_params[-1] == "alice@example.com"
    audit_query, audit_params = conn.cursor_obj.executed[1]
    assert "INSERT INTO memory_audit_log" in audit_query
    assert audit_params["action"] == "created"
    assert audit_params["actor"] == "alice@example.com"
    assert conn.commits == 1
    assert conn.cursor_obj.closed is True
    assert conn.closed is True


def test_insert_memory_returns_duplicate_when_no_row_returned(monkeypatch) -> None:
    # ON CONFLICT DO NOTHING means the INSERT returns no row on dup.
    conn = FakeConnection(script=[None])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    result = storage.insert_memory(
        project_id="memory-kb-poc",
        project_type="product",
        memory_type="episodic",
        domain="interactions",
        rule="question",
        context="Q: hi\nA: hello",
        source_ref="agent-interaction",
        embedding=[0.1, 0.2],
        quality_score=0.9,
        created_by="alice@example.com",
    )
    assert result.status == "duplicate"
    # Only the INSERT ran — no audit row for a no-op write.
    assert len(conn.cursor_obj.executed) == 1


def test_store_bootstrap_memories_uses_conflict_tolerant_inserts(monkeypatch) -> None:
    conn = FakeConnection()
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    stored = storage.store_bootstrap_memories(
        chunks=["chunk one", "chunk two"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        project_id="memory-kb-poc",
        source_ref="concept-doc-v1",
    )
    assert stored == 2
    assert len(conn.cursor_obj.executed) == 2
    assert all("ON CONFLICT DO NOTHING" in query for query, _ in conn.cursor_obj.executed)
    assert conn.commits == 1
    assert conn.cursor_obj.closed is True
    assert conn.closed is True


def test_get_lakebase_project_name_raises_when_unset(monkeypatch) -> None:
    monkeypatch.delenv(config.LAKEBASE_PROJECT_NAME_ENV_VAR, raising=False)

    class FakeSecrets:
        def get(self, scope, key):
            raise RuntimeError("secret missing")

    class FakeDbutils:
        secrets = FakeSecrets()

    class FakeWorkspaceClient:
        dbutils = FakeDbutils()

    monkeypatch.setattr(config, "WorkspaceClient", lambda: FakeWorkspaceClient())

    with pytest.raises(RuntimeError, match=r"Set LAKEBASE_PROJECT_NAME env var or the secret scope memory-scaling/lakebase_project_name\."):
        config.get_lakebase_project_name()


# ---------------------------------------------------------------------------
# Soft-forget
# ---------------------------------------------------------------------------


def _soft_forget_row(
    row_id="11111111-1111-1111-1111-111111111111",
    project_id="memory-kb-poc",
    rule="r",
    context="c",
    memory_type="episodic",
    domain="d",
    quality_score=0.5,
    forgotten_at="2026-05-18T00:00:00+00:00",
    forgotten_reason="wrong",
    forgotten_by_user="agent-1",
):
    return (row_id, project_id, rule, context, memory_type, domain, quality_score,
            forgotten_at, forgotten_reason, forgotten_by_user)


def test_soft_forget_marks_row(monkeypatch) -> None:
    # Two statements when the UPDATE lands: the UPDATE, then the audit insert.
    conn = FakeConnection(script=[_soft_forget_row(), None])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    result = storage.soft_forget_memory(
        "11111111-1111-1111-1111-111111111111",
        reason="wrong",
        user_id="agent-1",
    )
    assert result["status"] == "forgotten"
    assert result["forgotten_reason"] == "wrong"
    assert result["forgotten_by_user"] == "agent-1"
    assert result["project_id"] == "memory-kb-poc"
    assert conn.commits == 1
    # UPDATE + audit insert.
    assert len(conn.cursor_obj.executed) == 2
    audit_query, audit_params = conn.cursor_obj.executed[1]
    assert "INSERT INTO memory_audit_log" in audit_query
    assert audit_params["action"] == "forgotten"
    assert audit_params["actor"] == "agent-1"


def test_soft_forget_already_forgotten(monkeypatch) -> None:
    # Step 1: UPDATE returns no row (already-flagged guard).
    # Step 2: probe SELECT returns the existing forgotten state.
    conn = FakeConnection(script=[
        None,
        ("11111111-1111-1111-1111-111111111111", "2026-05-01T00:00:00+00:00", "stale", "agent-1"),
    ])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    result = storage.soft_forget_memory("11111111-1111-1111-1111-111111111111")
    assert result["status"] == "already_forgotten"
    assert result["forgotten_at"] == "2026-05-01T00:00:00+00:00"
    assert conn.commits == 0


def test_soft_forget_not_found(monkeypatch) -> None:
    conn = FakeConnection(script=[None, None])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    result = storage.soft_forget_memory("00000000-0000-0000-0000-000000000000")
    assert result["status"] == "not_found"
    assert conn.commits == 0


# ---------------------------------------------------------------------------
# Supersede
# ---------------------------------------------------------------------------


def _locked_row(
    row_id="11111111-1111-1111-1111-111111111111",
    project_id="memory-kb-poc",
    project_type="product",
    memory_type="episodic",
    domain="d",
    rule="old rule",
    superseded_at=None,
    superseded_by=None,
    forgotten_at=None,
):
    return (row_id, project_id, project_type, memory_type, domain, rule,
            superseded_at, superseded_by, forgotten_at)


def test_supersede_happy_path(monkeypatch) -> None:
    new_id = "22222222-2222-2222-2222-222222222222"
    conn = FakeConnection(script=[
        _locked_row(),                                  # LOCK_MEMORY_FOR_SUPERSEDE_SQL
        (new_id,),                                      # INSERT_SUPERSEDE_MEMORY_SQL
        (
            "11111111-1111-1111-1111-111111111111",     # MARK_SUPERSEDED_SQL
            "2026-05-18T00:00:00+00:00",
        ),
        None,                                           # audit 'created' for new row
        None,                                           # audit 'superseded' for old row
    ])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)

    result = storage.supersede_memory(
        old_id="11111111-1111-1111-1111-111111111111",
        new_content="corrected content",
        embedding=[0.1, 0.2],
        reason="threshold was wrong",
        user_id="agent-1",
    )
    assert result["status"] == "superseded"
    assert result["old_id"] == "11111111-1111-1111-1111-111111111111"
    assert result["new_id"] == new_id
    assert result["head_id"] == new_id
    assert conn.commits == 1
    assert conn.rollbacks == 0
    # Five SQL statements: lock + insert + mark + two audit writes.
    assert len(conn.cursor_obj.executed) == 5
    # The replacement INSERT inherited project_id from the locked old row.
    _, insert_params = conn.cursor_obj.executed[1]
    assert insert_params["project_id"] == "memory-kb-poc"
    assert insert_params["context"] == "corrected content"
    assert insert_params["created_by"] == "agent-1"
    # The mark statement passed both ids and the reason.
    _, mark_params = conn.cursor_obj.executed[2]
    assert mark_params["old_id"] == "11111111-1111-1111-1111-111111111111"
    assert mark_params["new_id"] == new_id
    assert mark_params["reason"] == "threshold was wrong"
    assert mark_params["user_id"] == "agent-1"
    # Audit rows: 'created' for the new memory, 'superseded' for the old.
    _, audit_new = conn.cursor_obj.executed[3]
    assert audit_new["action"] == "created"
    assert audit_new["memory_id"] == new_id
    assert audit_new["actor"] == "agent-1"
    _, audit_old = conn.cursor_obj.executed[4]
    assert audit_old["action"] == "superseded"
    assert audit_old["memory_id"] == "11111111-1111-1111-1111-111111111111"
    assert audit_old["reason"] == "threshold was wrong"


def test_supersede_carries_derived_from(monkeypatch) -> None:
    new_id = "22222222-2222-2222-2222-222222222222"
    conn = FakeConnection(script=[
        _locked_row(),
        (new_id,),
        ("11111111-1111-1111-1111-111111111111", "2026-05-18T00:00:00+00:00"),
        None,  # audit created
        None,  # audit superseded
    ])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)

    storage.supersede_memory(
        old_id="11111111-1111-1111-1111-111111111111",
        new_content="distilled synthesis",
        embedding=[0.1, 0.2],
        reason="redistillation",
        derived_from=[
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        ],
    )
    _, insert_params = conn.cursor_obj.executed[1]
    assert insert_params["derived_from"] == [
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    ]


def test_supersede_not_found(monkeypatch) -> None:
    conn = FakeConnection(script=[None])  # LOCK returns nothing
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    result = storage.supersede_memory(
        old_id="00000000-0000-0000-0000-000000000000",
        new_content="x",
        embedding=[0.1],
        reason="r",
    )
    assert result["status"] == "not_found"
    assert conn.commits == 0
    assert conn.rollbacks == 1


def test_supersede_already_superseded_returns_head(monkeypatch) -> None:
    # LOCK returns a row whose superseded_at is set.
    old_id = "11111111-1111-1111-1111-111111111111"
    head_id = "33333333-3333-3333-3333-333333333333"
    conn = FakeConnection(script=[
        _locked_row(superseded_at="2026-05-10T00:00:00+00:00",
                    superseded_by="22222222-2222-2222-2222-222222222222"),
        # _walk_to_head's fetchall: chain with head at the end (superseded_at=None).
        [
            (old_id, "22222222-2222-2222-2222-222222222222",
             "2026-05-10T00:00:00+00:00", "r1", "u", None, "old", "ctx",
             "episodic", "d", "org", 0.5, "2026-05-01T00:00:00+00:00", 0),
            ("22222222-2222-2222-2222-222222222222", head_id,
             "2026-05-15T00:00:00+00:00", "r2", "u", None, "mid", "ctx",
             "episodic", "d", "org", 0.5, "2026-05-10T00:00:00+00:00", 1),
            (head_id, None, None, None, None, None, "head", "ctx",
             "episodic", "d", "org", 0.5, "2026-05-15T00:00:00+00:00", 2),
        ],
    ])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    result = storage.supersede_memory(
        old_id=old_id, new_content="x", embedding=[0.1], reason="r",
    )
    assert result["status"] == "already_superseded"
    assert result["current_head_id"] == head_id
    assert conn.rollbacks == 1
    assert conn.commits == 0


def test_supersede_on_forgotten_old_row(monkeypatch) -> None:
    conn = FakeConnection(script=[
        _locked_row(forgotten_at="2026-05-01T00:00:00+00:00"),
    ])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    result = storage.supersede_memory(
        old_id="11111111-1111-1111-1111-111111111111",
        new_content="x", embedding=[0.1], reason="r",
    )
    assert result["status"] == "forgotten"
    assert conn.rollbacks == 1
    assert conn.commits == 0


def test_supersede_duplicate_content(monkeypatch) -> None:
    # LOCK ok, INSERT ON CONFLICT returns nothing.
    conn = FakeConnection(script=[
        _locked_row(),
        None,  # INSERT_SUPERSEDE_MEMORY_SQL: ON CONFLICT DO NOTHING -> no row
    ])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    result = storage.supersede_memory(
        old_id="11111111-1111-1111-1111-111111111111",
        new_content="x", embedding=[0.1], reason="r",
    )
    assert result["status"] == "duplicate_content"
    assert "content_hash" in result
    assert conn.rollbacks == 1
    assert conn.commits == 0


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------


def _lineage_row(
    row_id, superseded_by=None, superseded_at=None, reason=None, by_user=None,
    forgotten_at=None, rule="r", context="c", memory_type="episodic",
    domain="d", quality_score=0.5,
    created_at="2026-05-01T00:00:00+00:00", depth=0,
):
    return (row_id, superseded_by, superseded_at, reason, by_user, forgotten_at,
            rule, context, memory_type, domain, quality_score, created_at, depth)


def test_get_lineage_walks_forward_and_backward(monkeypatch) -> None:
    a = "11111111-1111-1111-1111-111111111111"
    b = "22222222-2222-2222-2222-222222222222"
    c = "33333333-3333-3333-3333-333333333333"
    forward = [
        _lineage_row(a, superseded_by=b, superseded_at="2026-05-10T00:00:00+00:00", depth=0),
        _lineage_row(b, superseded_by=c, superseded_at="2026-05-15T00:00:00+00:00", depth=1),
        _lineage_row(c, depth=2),  # head
    ]
    backward = []  # nothing earlier than a
    conn = FakeConnection(script=[forward, backward])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)

    lineage = storage.get_lineage(a)
    assert lineage is not None
    assert lineage.target_id == a
    assert lineage.head_id == c
    assert [n.id for n in lineage.chain] == [a, b, c]
    assert lineage.ancestors == []


def test_get_lineage_not_found(monkeypatch) -> None:
    conn = FakeConnection(script=[[]])  # forward walk returns nothing
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    assert storage.get_lineage("00000000-0000-0000-0000-000000000000") is None


# ---------------------------------------------------------------------------
# Project ACL
# ---------------------------------------------------------------------------


def test_grant_project_access_upserts(monkeypatch) -> None:
    conn = FakeConnection(script=[
        ("memory-kb-poc", "bob@example.com", "contributor",
         "2026-05-18T00:00:00+00:00", "alice@example.com"),
    ])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    access = storage.grant_project_access(
        "memory-kb-poc", "bob@example.com", "contributor", granted_by="alice@example.com"
    )
    assert access.user_name == "bob@example.com"
    assert access.role == "contributor"
    query, params = conn.cursor_obj.executed[0]
    assert "ON CONFLICT" in query and "DO UPDATE" in query
    assert params["role"] == "contributor"
    assert params["granted_by"] == "alice@example.com"
    assert conn.commits == 1


def test_grant_project_access_rejects_unknown_role() -> None:
    import pytest
    with pytest.raises(ValueError, match="must be one of"):
        storage.grant_project_access("p", "u", "admin", granted_by="a")


def test_get_user_role_returns_role_or_none(monkeypatch) -> None:
    # Match found.
    conn = FakeConnection(script=[("owner",)])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    assert storage.get_user_role("p", "alice@example.com") == "owner"
    # No match.
    conn2 = FakeConnection(script=[None])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn2)
    assert storage.get_user_role("p", "carol@example.com") is None


def test_accessible_projects_for_returns_set(monkeypatch) -> None:
    conn = FakeConnection(fetch_rows=[("proj-a",), ("proj-b",)])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    result = storage.accessible_projects_for("alice@example.com")
    assert result == {"proj-a", "proj-b"}


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


def test_get_memory_audit_log_returns_typed_rows(monkeypatch) -> None:
    audit_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    memory_id = "11111111-1111-1111-1111-111111111111"
    conn = FakeConnection(fetch_rows=[(
        audit_id, memory_id, "memory-kb-poc", "superseded", "alice@example.com",
        "fixed threshold", {"rule": "old"}, {"superseded_by": "22222222-..."},
        "2026-05-18T00:00:00+00:00",
    )])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    entries = storage.get_memory_audit_log(memory_id)
    assert len(entries) == 1
    assert entries[0].action == "superseded"
    assert entries[0].actor == "alice@example.com"
    assert entries[0].before_state == {"rule": "old"}
    assert entries[0].after_state == {"superseded_by": "22222222-..."}


def test_get_memory_project_returns_project_id(monkeypatch) -> None:
    conn = FakeConnection(script=[("memory-kb-poc",)])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    assert storage.get_memory_project("11111111-1111-1111-1111-111111111111") == "memory-kb-poc"
    conn2 = FakeConnection(script=[None])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn2)
    assert storage.get_memory_project("00000000-0000-0000-0000-000000000000") is None


def test_delete_memory_writes_purged_audit_row(monkeypatch) -> None:
    # DELETE_MEMORY_SQL now returns project_id as well.
    conn = FakeConnection(script=[(
        "11111111-1111-1111-1111-111111111111",
        "memory-kb-poc", "r", "c", "episodic", "d", 0.5,
    ), None])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    result = storage.delete_memory(
        "11111111-1111-1111-1111-111111111111",
        actor="alice@example.com",
        reason="GDPR erasure request",
    )
    assert result["project_id"] == "memory-kb-poc"
    # DELETE + audit insert.
    assert len(conn.cursor_obj.executed) == 2
    _, audit_params = conn.cursor_obj.executed[1]
    assert audit_params["action"] == "purged"
    assert audit_params["actor"] == "alice@example.com"
    assert audit_params["reason"] == "GDPR erasure request"
