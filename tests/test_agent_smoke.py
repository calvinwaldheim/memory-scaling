from __future__ import annotations

from memory_agent import agent
from memory_agent.storage import RetrievedMemory


def test_answer_runs_retrieve_generate_writeback(monkeypatch) -> None:
    inserted = {}

    def fake_embed_text(text: str) -> list[float]:
        return [0.1, 0.2] if text.startswith("What") else [0.3, 0.4]

    def fake_retrieve_memories(
        query_embedding,
        project_id: str,
        top_k: int,
        project_ids=None,
        memory_type=None,
        domain=None,
        min_quality_score=None,
        include_inactive: bool = False,
    ):
        assert query_embedding == [0.1, 0.2]
        assert project_id == "memory-kb-poc"
        assert top_k == 3
        assert include_inactive is False
        return [
            RetrievedMemory("r1", "ctx one", 0.8, 0.2),
            RetrievedMemory("r2", "ctx two", 0.7, 0.3),
        ]

    def fake_generate_answer(question: str, context: str) -> str:
        assert question == "What did we discuss?"
        assert context == "ctx one\n\nctx two"
        return "A grounded answer"

    def fake_insert_memory(**kwargs) -> None:
        inserted.update(kwargs)

    monkeypatch.setattr(agent, "embed_text", fake_embed_text)
    monkeypatch.setattr(agent, "retrieve_memories", fake_retrieve_memories)
    monkeypatch.setattr(agent, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(agent, "insert_memory", fake_insert_memory)

    result = agent.answer("What did we discuss?")

    assert result == "A grounded answer"
    assert inserted["memory_type"] == "episodic"
    assert inserted["project_id"] == "memory-kb-poc"
    assert inserted["source_ref"] == "agent-interaction"
    assert inserted["context"] == "Q: What did we discuss?\nA: A grounded answer"
