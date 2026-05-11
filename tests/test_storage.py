from __future__ import annotations

from memory_agent import storage


class FakeCursor:
    def __init__(self, fetch_rows=None):
        self.fetch_rows = fetch_rows or []
        self.executed = []
        self.closed = False

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
