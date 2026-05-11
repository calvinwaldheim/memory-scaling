# Databricks notebook source
# MAGIC %pip install -e /Workspace/Users/calvin.waldheim@gmail.com/memory-scaling
# COMMAND ----------
from memory_agent.agent import retrieve

query = "How does memory scaling reduce reasoning steps?"
results = retrieve(query)
for i, memory in enumerate(results, start=1):
    print(f"\n--- Result {i} (distance: {memory.distance:.3f}) ---")
    print(memory.context[:300])

