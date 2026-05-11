from __future__ import annotations

"""Embedding helpers backed by Databricks Foundation Model endpoints."""

from collections.abc import Sequence
from typing import Any

from mlflow.deployments import get_deploy_client

from .config import EMBEDDING_ENDPOINT


def _get_client() -> Any:
    return get_deploy_client("databricks")


def embed_text(text: str) -> list[float]:
    """Embed a single text string with the configured Foundation Model endpoint.

    Args:
        text: The input text to embed.

    Returns:
        The embedding vector for the input text.

    Raises:
        Exception: Propagates client errors from the endpoint invocation.
    """
    response = _get_client().predict(
        endpoint=EMBEDDING_ENDPOINT,
        inputs={"input": [text]},
    )
    return response["data"][0]["embedding"]


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed texts one at a time to preserve notebook bootstrap behavior.

    Args:
        texts: Ordered text strings to embed.

    Returns:
        A list of embedding vectors in the same order as the inputs.

    Raises:
        Exception: Propagates client errors from individual embedding requests.
    """
    return [embed_text(text) for text in texts]


def embed(text: str) -> list[float]:
    """Compatibility alias for app-facing callers that expect embeddings.embed()."""
    return embed_text(text)
