# Databricks notebook source
# MAGIC %pip install -e /Workspace/Users/calvin.waldheim@gmail.com/memory-scaling
# COMMAND ----------
from memory_agent.chunking import chunk_text
from memory_agent.embeddings import embed_texts
from memory_agent.storage import store_bootstrap_memories

with open("/Workspace/Users/calvin.waldheim@gmail.com/memory-scaling/concept.txt", "r") as handle:
    source_text = handle.read()
chunks = chunk_text(source_text)
embeddings = embed_texts(chunks)
stored = store_bootstrap_memories(chunks, embeddings)
print(f"Loaded {len(source_text)} characters")
print(f"{len(chunks)} chunks created")
print(f"{len(embeddings)} embeddings created")
print(f"Stored {stored} memories.")

