"""
Unit tests for indexing dry run mode (Issue #222).
"""

import os
import tempfile
import sys
from unittest.mock import MagicMock
import indexer


def test_index_directory_dry_run_mode(monkeypatch):
    """Dry run mode processes repository files and calculates statistics without uploading vectors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "test.py")
        with open(file1, "w") as f:
            f.write("def hello():\n    return 'world'\n")

        # Mock qdrant client upsert to ensure it's not called
        mock_upsert = MagicMock()
        monkeypatch.setattr(indexer.client, "upsert", mock_upsert)

        summary = indexer.index_directory(tmpdir, dry_run=True)

        assert summary["dry_run"] is True
        assert summary["total_files"] == 1
        assert summary["estimated_chunks"] >= 1
        assert summary["total_symbols"] >= 1
        assert summary["vectors_uploaded"] == 0

        # Assert no vector upserts occurred
        mock_upsert.assert_not_called()
