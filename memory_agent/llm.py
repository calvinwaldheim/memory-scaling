from __future__ import annotations

"""LLM helpers backed by Databricks Foundation Model endpoints."""

from typing import Any

from mlflow.deployments import get_deploy_client

from .config import LLM_ENDPOINT


def _get_client() -> Any:
    return get_deploy_client("databricks")


def generate_answer(question: str, context: str) -> str:
    """Generate an answer grounded in retrieved memory context.

    Args:
        question: The user question to answer.
        context: The retrieved memory context injected into the system prompt.

    Returns:
        The generated answer text.

    Raises:
        Exception: Propagates client errors from the endpoint invocation.
    """
    response = _get_client().predict(
        endpoint=LLM_ENDPOINT,
        inputs={
            "messages": [
                {
                    "role": "system",
                    "content": f"""You are a helpful assistant. 
Answer questions using the context below. If the context doesn't contain the answer, say so.

CONTEXT:
{context}""",
                },
                {"role": "user", "content": question},
            ]
        },
    )
    return response["choices"][0]["message"]["content"]
