from __future__ import annotations

"""Configuration constants and secret-backed accessors for memory_agent."""

from databricks.sdk import WorkspaceClient

EMBEDDING_ENDPOINT = "databricks-gte-large-en"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
DEFAULT_CHUNK_SIZE = 150
DEFAULT_CHUNK_OVERLAP = 20
DEFAULT_TOP_K = 3
DEFAULT_PROJECT_ID = "memory-kb-poc"
DEFAULT_BOOTSTRAP_SOURCE_REF = "concept-doc-v1"
AGENT_SOURCE_REF = "agent-interaction"
SECRET_SCOPE = "memory-scaling"
LAKEBASE_URI_SECRET_KEY = "lakebase_uri"


def get_lakebase_uri() -> str:
    """Return the Lakebase PostgreSQL connection URI from Databricks secrets.

    Returns:
        The secret-backed PostgreSQL connection URI.

    Raises:
        Exception: Propagates Databricks SDK exceptions if the secret lookup fails.
    """
    return WorkspaceClient().dbutils.secrets.get(
        scope=SECRET_SCOPE,
        key=LAKEBASE_URI_SECRET_KEY,
    )
