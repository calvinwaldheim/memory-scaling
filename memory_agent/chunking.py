from __future__ import annotations

"""Text chunking helpers for memory bootstrap flows."""

from .config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into word chunks using the notebook's original overlap logic.

    Args:
        text: Source text to chunk.
        chunk_size: Maximum number of words per chunk.
        overlap: Number of overlapping words between adjacent chunks.

    Returns:
        A list of chunk strings in original order.

    Raises:
        No explicit exceptions are raised by this function.
    """
    words = text.split()
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks
