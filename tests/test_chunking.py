from __future__ import annotations

from memory_agent.chunking import chunk_text


def test_chunk_text_preserves_word_overlap() -> None:
    words = [f"w{i}" for i in range(1, 201)]
    chunks = chunk_text(" ".join(words))
    assert len(chunks) == 2
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert len(first_words) == 150
    assert len(second_words) == 70
    assert first_words[-20:] == second_words[:20]


def test_chunk_text_empty_string_returns_no_chunks() -> None:
    assert chunk_text("") == []
