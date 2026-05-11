# memory-scaling MCP app

This Databricks App exposes three MCP tools for the memory-scaling POC:

* `recall(query, top_k=3, project_id="memory-kb-poc")` retrieves the nearest stored memories and returns only agent-relevant fields.
* `remember(content, source_ref, ...)` embeds content, writes it through `memory_agent.storage`, and reports whether the row was stored or deduplicated.
* `stats(project_id="memory-kb-poc")` returns memory counts by type and domain plus the latest write timestamp.

Run locally from the repo root after installing the app requirements:

* `mcp dev app/app.py`

If your local `mcp` CLI version does not provide `dev`, run the server directly instead:

* `python app/app.py`

Deployment is manual via the Databricks Apps UI when you are ready. No deploy commands are included here.
