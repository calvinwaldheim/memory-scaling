from __future__ import annotations

import pytest

from memory_agent import config, storage


class FakeCursor:
    def __init__(self, fetch_rows=None):
        self.fetch_rows = fetch_rows or []
        self.executed = []
        self.closed = False
        self.rowcount = 1

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchall(self):
        return list(self.fetch_rows)

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, fetch_rows=None):
        self.cursor_obj = FakeCursor(fetch_rows=fetch_rows)
        self.commits = 0
        self.closed = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.commits += 1

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


def test_retrieve_memories_returns_typed_rows(monkeypatch) -> None:
    conn = FakeConnection(fetch_rows=[("rule", "context", 0.8, 0.25)])
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    results = storage.retrieve_memories([0.1, 0.2], project_id="memory-kb-poc", top_k=3)
    assert len(results) == 1
    assert results[0].rule == "rule"
    query, params = conn.cursor_obj.executed[0]
    assert "LIMIT %s" in query
    assert params[1] == "memory-kb-poc"
    assert params[2] == 3
    assert conn.cursor_obj.closed is True
    assert conn.closed is True


def test_insert_memory_uses_parameterized_sql(monkeypatch) -> None:
    conn = FakeConnection()
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    storage.insert_memory(
        project_id="memory-kb-poc",
        project_type="product",
        memory_type="episodic",
        scope="organizational",
        domain="interactions",
        rule="question",
        context="Q: hi\nA: hello",
        source_ref="agent-interaction",
        embedding=[0.1, 0.2],
        quality_score=0.9,
    )
    assert len(conn.cursor_obj.executed) == 1
    insert_query, insert_params = conn.cursor_obj.executed[0]
    assert "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)" in insert_query
    assert "ON CONFLICT DO NOTHING" in insert_query
    assert insert_params[0] == "memory-kb-poc"
    assert insert_params[2] == "episodic"
    assert conn.commits == 1
    assert conn.cursor_obj.closed is True
    assert conn.closed is True


def test_insert_memory_returns_duplicate_when_rowcount_zero(monkeypatch) -> None:
    conn = FakeConnection()
    conn.cursor_obj.rowcount = 0
    monkeypatch.setattr(storage, "_get_connection", lambda: conn)
    result = storage.insert_memory(
        project_id="memory-kb-poc",
        project_type="product",
        memory_type="episodic",
        scope="organizational",
        domain="interactions",
        rule="question",
        context="Q: hi\nA: hello",
        source_ref="agent-interaction",
        embedding=[0.1, 0.2],
        quality_score=0.9,
    )
    assert result.status == "duplicate"


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
