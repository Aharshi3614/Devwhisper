"""Unit tests for the chunk statistics report (issue #214).

`compute_chunk_statistics` aggregates chunk counts and sizes after
indexing so the frontend can display them. A chunk's "size" is its
number of lines.
"""

from indexer import compute_chunk_statistics


def _chunk(text, file="a.py", start_line=1):
    """Build a minimal chunk dict like the indexer produces."""
    return {"text": text, "file": file, "start_line": start_line}


def test_empty_chunks_produce_zero_stats():
    """No chunks → all stats are zero/None, never an error."""
    result = compute_chunk_statistics([])
    assert result["total_chunks"] == 0
    assert result["average_size"] == 0
    assert result["largest"] is None
    assert result["smallest"] is None


def test_total_chunk_count():
    """Total chunk count equals the number of input chunks."""
    chunks = [_chunk("a\n"), _chunk("b\n"), _chunk("c\n")]
    result = compute_chunk_statistics(chunks)
    assert result["total_chunks"] == 3


def test_average_size_is_mean_line_count():
    """Average size is the mean of each chunk's line count."""
    chunks = [
        _chunk("1\n2\n"),            # 2 lines
        _chunk("only-one-line"),      # 1 line
        _chunk("1\n2\n3\n4\n5\n"),    # 5 lines
    ]
    result = compute_chunk_statistics(chunks)
    assert result["average_size"] == round((2 + 1 + 5) / 3, 1)


def test_identifies_largest_and_smallest():
    """Largest/smallest describe the chunk's file, start_line, and size."""
    chunks = [
        _chunk("small", file="tiny.py", start_line=3),       # 1 line
        _chunk("1\n2\n3\n4\n5\n6\n7\n8\n", file="big.py", start_line=10),  # 8 lines
        _chunk("mid\n", file="mid.py", start_line=1),        # 1 line, ties small
    ]
    result = compute_chunk_statistics(chunks)

    assert result["largest"] == {"file": "big.py", "start_line": 10, "size": 8}
    # smallest is the first chunk with the minimum size (tiny.py).
    assert result["smallest"] == {"file": "tiny.py", "start_line": 3, "size": 1}


def test_single_chunk_is_both_largest_and_smallest():
    """With one chunk, largest == smallest == that chunk."""
    chunks = [_chunk("hello world", file="only.py", start_line=7)]
    result = compute_chunk_statistics(chunks)

    assert result["total_chunks"] == 1
    assert result["average_size"] == 1
    assert result["largest"] == {"file": "only.py", "start_line": 7, "size": 1}
    assert result["smallest"] == result["largest"]


def test_line_count_counts_newlines_not_characters():
    """A chunk's size is line-based, not character-based."""
    one_line = _chunk("a very long single line with lots of characters")
    three_lines = _chunk("1\n2\n3")
    result = compute_chunk_statistics([one_line, three_lines])

    assert result["smallest"] == {"file": "a.py", "start_line": 1, "size": 1}
    assert result["largest"] == {"file": "a.py", "start_line": 1, "size": 3}
    assert result["average_size"] == 2
