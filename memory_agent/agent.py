from __future__ import annotations

"""Public retrieval and answer entry points for the memory agent."""

from .config import AGENT_SOURCE_REF, DEFAULT_PROJECT_ID, DEFAULT_TOP_K
from .embeddings import embed_text
from .llm import generate_answer
from .storage import RetrievedMemory, insert_memory, retrieve_memories


def retrieve(
    question: str,
    project_id: str = DEFAULT_PROJECT_ID,
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievedMemory]:
    """Embed a question and retrieve the top-k similar memories.

    Args:
        question: The natural-language query to retrieve against.
        project_id: Project identifier filter for the memories table.
        top_k: Maximum number of memories to return.

    Returns:
        Retrieved memory rows sorted by ascending cosine distance.

    Raises:
        Exception: Propagates embedding or database errors.
    """
    return retrieve_memories(
        query_embedding=embed_text(question),
        project_id=project_id,
        top_k=top_k,
    )


def answer(
    question: str,
    project_id: str = DEFAULT_PROJECT_ID,
    top_k: int = DEFAULT_TOP_K,
) -> str:
    """Answer a question using retrieved memory context and store the interaction.

    Args:
        question: The question to answer.
        project_id: Project identifier filter and write target for memories.
        top_k: Maximum number of memories to retrieve.

    Returns:
        The model-generated answer text.

    Raises:
        Exception: Propagates embedding, LLM, or database errors.
    """
    memories = retrieve(question=question, project_id=project_id, top_k=top_k)
    context = "\n\n".join(memory.context for memory in memories)
    response = generate_answer(question=question, context=context)
    content = f"Q: {question}\nA: {response}"
    insert_memory(
        project_id=project_id,
        project_type="product",
        memory_type="episodic",
        scope="organizational",
        domain="interactions",
        rule=question[:100],
        context=content,
        source_ref=AGENT_SOURCE_REF,
        embedding=embed_text(content),
        quality_score=0.9,
    )
    return response
