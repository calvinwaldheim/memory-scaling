from __future__ import annotations

"""Configuration constants and secret-backed accessors for memory_agent."""

import os

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
LAKEBASE_URI_ENV_VAR = "LAKEBASE_URI"
LAKEBASE_PROJECT_NAME_SECRET_KEY = "lakebase_project_name"
LEGACY_LAKEBASE_INSTANCE_NAME_SECRET_KEY = "lakebase_instance_name"
LAKEBASE_PROJECT_NAME_ENV_VAR = "LAKEBASE_PROJECT_NAME"


def get_lakebase_uri() -> str:
    """Return the Lakebase PostgreSQL connection URI from env or Databricks secrets.

    Returns:
        The PostgreSQL connection URI for the active runtime.

    Raises:
        Exception: Propagates Databricks SDK exceptions if the secret lookup fails.
    """
    lakebase_uri = os.environ.get(LAKEBASE_URI_ENV_VAR)
    if lakebase_uri:
        return lakebase_uri
    return WorkspaceClient().dbutils.secrets.get(
        scope=SECRET_SCOPE,
        key=LAKEBASE_URI_SECRET_KEY,
    )


def get_lakebase_project_name() -> str:
    """Return the Lakebase autoscaling project name from env or Databricks secrets.

    Raises:
        RuntimeError: If neither the environment variable nor a configured secret is set.
    """
    project_name = os.environ.get(LAKEBASE_PROJECT_NAME_ENV_VAR)
    if project_name:
        return project_name

    secrets = WorkspaceClient().dbutils.secrets
    for secret_key in (
        LAKEBASE_PROJECT_NAME_SECRET_KEY,
        LEGACY_LAKEBASE_INSTANCE_NAME_SECRET_KEY,
    ):
        try:
            project_name = secrets.get(scope=SECRET_SCOPE, key=secret_key)
        except Exception:
            continue
        if project_name:
            return project_name

    raise RuntimeError(
        "Set LAKEBASE_PROJECT_NAME env var or the secret scope memory-scaling/lakebase_project_name."
    )
