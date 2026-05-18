# memory-scaling MCP app

A Databricks App that exposes the entire `memory_agent` package over MCP. Connect any MCP-aware client (Claude Desktop, Claude Code, Cursor, custom agents) and you get a project-scoped memory store with read, write, correct, retract, and audit capabilities.

The operating manual for the agent *using* this server is [`../AGENT.md`](../AGENT.md). This file covers the **server** — its tool surface, configuration, and how to deploy it.

---

## Tool surface

Thirteen MCP tools, grouped by concern.

### Reading

- **`recall(query, top_k=3, project_id=None, project_ids=None, memory_type=None, domain=None, min_quality_score=None, include_inactive=False)`**
  Embed the query and return the top-k most-similar memories. Each result carries `id`, `content`, `source_ref`, `memory_type`, `domain`, `rule`, `quality_score`, `distance`, `project_id`, `superseded_at`, `superseded_by`, `forgotten_at`. Default excludes superseded and forgotten rows — flip `include_inactive=True` for audit.
- **`get_lineage(memory_id)`**
  Walks the supersede chain forward (target → head) and backward (ancestors). Returns `target_id`, `head_id`, `chain[]`, `ancestors[]`.
- **`stats(project_id=None)`**
  Project health snapshot: `total`, `active_count`, `superseded_count`, `forgotten_count`, by-type/by-domain breakdowns, retrieval percentiles, `avg_quality_score`.
- **`list_hot(top_k=10, project_id=None)`**
  Most-retrieved memories in the project, sorted by `retrieval_count DESC`.

### Writing

- **`remember(content, source_ref, memory_type="episodic", scope="organizational", domain=None, rule=None, project_id=None, quality_score=0.5)`**
  Write a new memory. Dedupes on `(project_id, content_hash)` — duplicate calls return `{"status": "duplicate"}`.
- **`supersede(old_id, new_content, reason, user_id=None, rule=None, quality_score=None, source_ref=None, memory_type=None, scope=None, domain=None)`**
  Atomically replace a wrong memory. Inherits `project_id`, `project_type`, `user_id`, `memory_type`, `scope`, `domain` from the old row by default. Five outcomes: `superseded`, `already_superseded` (with `current_head_id`), `forgotten`, `duplicate_content`, `not_found`.
- **`forget(memory_id, reason=None, user_id=None, hard=False)`**
  Soft-retract by default (sets `forgotten_at`, keeps the row). `hard=True` permanently deletes — only for GDPR or secret erasure.
- **`update_memory(memory_id, rule=None, domain=None, quality_score=None, memory_type=None, scope=None)`**
  Edit metadata only. Cannot touch `content` — that's what `supersede` is for.

### Projects

- **`create_project(project_id, name, project_type, description=None, tags=None)`**
  Register a new project. `project_type` ∈ `data_domain`, `engineering`, `compliance`, `customer`, `product`. `project_id` is a slug.
- **`list_projects(include_archived=False)`**
  Returns all projects with `memory_count` populated.
- **`archive_project(project_id)`**
  Soft-delete (sets `archived_at`). Memories remain queryable by explicit `project_id`.
- **`set_active_project(project_id)`**
  Make this project the default for all subsequent tool calls in this MCP subprocess. Lives in memory only — restarts wipe it.
- **`get_active_project()`**
  Reports `active_project_id`, `default_project_id`, `effective_project_id`.

### Project precedence

For every tool that takes `project_id`, the effective project is resolved in this order:
1. The `project_ids` list arg (if the tool accepts one).
2. The explicit `project_id` arg.
3. The active project from `set_active_project(...)`.
4. The `DEFAULT_PROJECT_ID` env var (defaults to `memory-kb-poc`).

---

## Configuration

The server reads its config from environment variables. Defaults are set in [`app.yaml`](app.yaml):

| Variable | Purpose |
|---|---|
| `LAKEBASE_URI` | PostgreSQL connection string for the Lakebase instance. Resolved from the `memory-scaling/lakebase_uri` Databricks secret by default. |
| `LAKEBASE_PROJECT_NAME` | Lakebase project name (for OAuth-scoped credentials). Resolved from the `memory-scaling/lakebase_project_name` secret by default. |
| `DEFAULT_PROJECT_ID` | Fallback project id when no active project is set and no `project_id` is passed. Default: `memory-kb-poc`. |
| `DATABRICKS_APP_PORT` | HTTP port. Default `8000`. Databricks Apps sets this automatically. |
| `DATABRICKS_APP_HOST` | Bind host. Default `0.0.0.0`. |
| `DATABRICKS_WORKSPACE_URL` | Workspace URL for the token verifier. |
| `DATABRICKS_APP_URL` | This app's public URL — used as the OAuth resource_server_url. |
| `MCP_TRANSPORT` | `streamable-http` (default) for Databricks Apps deployment, or `stdio` for local development. |

---

## Authentication

When `MCP_TRANSPORT != "stdio"`, the server runs with `DatabricksTokenVerifier`. Every request must carry a `Bearer <token>` header where the token is valid for the configured Databricks workspace. The verifier hits `/api/2.0/preview/scim/v2/Me` to confirm. Tokens are caller-scoped — Lakebase rejects connections where the OAuth identity doesn't match the URI's user (see the URI substitution pattern in [`05_distillation.py`](../05_distillation.py)).

In `stdio` mode, no auth is performed — suitable for local development only.

---

## Running locally

```bash
# From the repo root
pip install -e .
pip install mcp httpx psycopg2-binary databricks-sdk mlflow
export LAKEBASE_URI=...        # set or rely on the SDK to resolve from the secret
export LAKEBASE_PROJECT_NAME=memory-kb-poc
export MCP_TRANSPORT=stdio     # for direct stdio client connection
python app/app.py
```

Connect from Claude Desktop by adding to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memory-scaling": {
      "command": "python",
      "args": ["/path/to/app/app.py"],
      "env": {
        "MCP_TRANSPORT": "stdio",
        "DEFAULT_PROJECT_ID": "memory-kb-poc"
      }
    }
  }
}
```

---

## Deploying to Databricks Apps

1. Rebuild the vendored wheel so the deployed app has the latest `memory_agent`:

   ```bash
   bash scripts/build_app_wheel.sh
   ```

   This produces `app/vendor/memory_agent-0.1.0-py3-none-any.whl` and updates the last line of [`requirements.txt`](requirements.txt) to point at it.

2. Through the Databricks Apps UI, deploy the `app/` folder. The runtime reads [`app.yaml`](app.yaml) for the start command and env-var bindings.

3. Confirm the workspace secrets exist:
   - `memory-scaling/lakebase_uri`
   - `memory-scaling/lakebase_project_name`

   These are referenced by `valueFrom` in `app.yaml`.

4. After deploy, run the smoke checks from the operating manual ([`../AGENT.md`](../AGENT.md)) — `list_projects`, `stats`, a probe `recall` — to confirm the server is wired up.

---

## What `memory_agent.storage` does under the hood

This server is a thin MCP wrapper over [`memory_agent.storage`](../memory_agent/storage.py). The interesting work — atomic supersede transactions, recursive lineage CTEs, soft-forget guards, the partial active-row index — all lives in that module and is unit-tested in [`tests/test_storage.py`](../tests/test_storage.py). Read those two files together if you want to understand or extend the behavior.
