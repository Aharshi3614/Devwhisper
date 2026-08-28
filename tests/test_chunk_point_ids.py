"""Regression tests for chunk point id uniqueness (issue #292).

The Qdrant point id used to be ``uuid5(NAMESPACE_OID, f"{path}_{start_line}")``.
``start_line`` is not unique within a file — ``_chunk_source_region()`` splits
an oversized class at ``start + n * (chunk_size - overlap)``, and those offsets
land on the first line of the class's own methods often enough that this
repository loses three chunks to it. Qdrant's ``upsert`` treats a repeated id
as an update, so the loss is silent.

These tests pin down the three properties the id scheme has to have:

  * unique per chunk, so nothing is overwritten
  * stable across runs, so incremental re-indexing does not orphan points
  * scoped by repository and full path, so shared-collection mode is safe
"""

import collections
import os
import uuid

import pytest

import indexer
from indexer import chunk_point_id, drop_duplicate_points, get_file_chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ids_for(path):
    """Return the point ids the indexer would mint for every chunk of *path*."""
    return [chunk_point_id(chunk, path, "repo") for chunk in get_file_chunks(path)]


def _duplicates(ids):
    """Return {id: count} for ids appearing more than once."""
    counts = collections.Counter(ids)
    return {key: count for key, count in counts.items() if count > 1}


@pytest.fixture
def colliding_module(tmp_path):
    """A module whose class is long enough to be split across its own methods.

    With ``INDEX_CHUNK_SIZE=15`` / ``INDEX_CHUNK_OVERLAP=3`` the class body is
    chunked at a stride of 12 lines, so part 2 begins on a line that is also
    the first line of one of the methods. That is exactly the shape that
    collided under the old scheme.
    """
    source = ["class Widget:", '    """A class long enough to be split."""', ""]
    for n in range(6):
        source.append(f"    def method_{n}(self):")
        source.append(f'        """Docstring for method {n}."""')
        source.append(f"        value = {n}")
        source.append("        for i in range(value):")
        source.append("            value += i")
        source.append("        return value")
        source.append("")

    path = tmp_path / "widget.py"
    path.write_text("\n".join(source), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Uniqueness
# ---------------------------------------------------------------------------
def test_split_class_and_its_methods_get_distinct_ids(colliding_module):
    """A class part and a method starting on the same line must not collide."""
    chunks = get_file_chunks(colliding_module)

    by_line = collections.defaultdict(list)
    for chunk in chunks:
        by_line[chunk["start_line"]].append(chunk)
    shared = [line for line, group in by_line.items() if len(group) > 1]
    assert shared, (
        "fixture no longer produces two chunks on one start_line; "
        "the regression it guards can no longer be reproduced"
    )

    ids = [chunk_point_id(chunk, colliding_module, "repo") for chunk in chunks]
    assert not _duplicates(ids)
    assert len(set(ids)) == len(chunks)


def test_old_scheme_collided_on_the_same_fixture(colliding_module):
    """Document the defect: the previous key really does lose chunks here."""
    chunks = get_file_chunks(colliding_module)
    legacy = [
        str(uuid.uuid5(uuid.NAMESPACE_OID, f"{colliding_module}_{c['start_line']}"))
        for c in chunks
    ]
    assert _duplicates(legacy), "expected the old (path, start_line) key to collide"

    fixed = [chunk_point_id(chunk, colliding_module, "repo") for chunk in chunks]
    assert not _duplicates(fixed)


@pytest.mark.parametrize(
    "source_file",
    ["symbol_parser.py", "pipeline_validator.py", "retriever.py", "main.py"],
)
def test_real_source_files_produce_unique_ids(source_file):
    """The three known collisions in this repository's own source are gone."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), source_file)
    ids = _ids_for(path)
    assert ids, f"{source_file} produced no chunks"
    assert not _duplicates(ids)


def test_module_and_symbol_chunks_on_one_line_differ():
    """The is_symbol flag alone is enough to separate two otherwise equal chunks."""
    base = {"start_line": 10, "end_line": 24, "symbol_name": None}
    module_chunk = dict(base, is_symbol=False, chunk_type="module_context")
    symbol_chunk = dict(base, is_symbol=True, symbol_name="handler")

    assert chunk_point_id(module_chunk, "a.py", "repo") != chunk_point_id(
        symbol_chunk, "a.py", "repo"
    )


def test_symbol_parts_of_one_symbol_differ():
    """Parts 1..n of a split symbol are distinct points, not one overwritten one."""
    ids = {
        chunk_point_id(
            {
                "start_line": 10,
                "end_line": 24,
                "symbol_name": "Widget",
                "is_symbol": True,
                "symbol_part": part,
                "symbol_parts": 4,
            },
            "a.py",
            "repo",
        )
        for part in (1, 2, 3, 4)
    }
    assert len(ids) == 4


def test_methods_of_different_classes_differ():
    """parent_class participates, so Foo.run and Bar.run are separate points."""
    def make(parent):
        return {
            "start_line": 10,
            "end_line": 20,
            "symbol_name": "run",
            "parent_class": parent,
            "is_symbol": True,
        }

    assert chunk_point_id(make("Foo"), "a.py", "repo") != chunk_point_id(
        make("Bar"), "a.py", "repo"
    )


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------
def test_id_is_stable_across_calls(colliding_module):
    """Re-indexing an unchanged file must reuse the same ids, not orphan points."""
    first = _ids_for(colliding_module)
    second = _ids_for(colliding_module)
    assert first == second


def test_id_is_a_valid_uuid_string():
    """Qdrant accepts a UUID string or an unsigned integer, nothing else."""
    point_id = chunk_point_id(
        {"start_line": 1, "end_line": 5, "is_symbol": False}, "a.py", "repo"
    )
    assert isinstance(point_id, str)
    assert str(uuid.UUID(point_id)) == point_id


# ---------------------------------------------------------------------------
# Scoping
# ---------------------------------------------------------------------------
def test_same_relative_path_in_two_repositories_differs():
    """Shared-collection mode: two repos with a common path must not collide."""
    chunk = {"start_line": 1, "end_line": 12, "is_symbol": False}
    assert chunk_point_id(chunk, "src/utils.py", "alpha") != chunk_point_id(
        chunk, "src/utils.py", "beta"
    )


def test_same_basename_in_two_directories_differs():
    """generate_embeddings() keyed on the basename; two utils.py collided."""
    chunk = {"start_line": 1, "end_line": 12, "is_symbol": False}
    assert chunk_point_id(chunk, "pkg_a/utils.py", "repo") != chunk_point_id(
        chunk, "pkg_b/utils.py", "repo"
    )


def test_separator_cannot_be_forged_from_a_path():
    """Field boundaries are unambiguous, so 'a' + 'b' cannot equal 'ab' + ''."""
    left = chunk_point_id(
        {"start_line": 1, "end_line": 2, "symbol_name": "b", "is_symbol": True},
        "a",
        "repo",
    )
    right = chunk_point_id(
        {"start_line": 1, "end_line": 2, "symbol_name": "", "is_symbol": True},
        "ab",
        "repo",
    )
    assert left != right


def test_missing_repository_is_accepted():
    """repo_id is None for a single-repository install; that must still work."""
    point_id = chunk_point_id(
        {"start_line": 1, "end_line": 5, "is_symbol": False}, "a.py", None
    )
    assert str(uuid.UUID(point_id)) == point_id


# ---------------------------------------------------------------------------
# The duplicate guard
# ---------------------------------------------------------------------------
class _FakePoint:
    def __init__(self, point_id, payload=None):
        self.id = point_id
        self.payload = payload or {}


def test_drop_duplicate_points_keeps_order_and_first_occurrence():
    points = [
        _FakePoint("a", {"file": "x.py", "start_line": 1}),
        _FakePoint("b", {"file": "x.py", "start_line": 20}),
        _FakePoint("a", {"file": "x.py", "start_line": 1, "symbol_name": "dup"}),
        _FakePoint("c", {"file": "y.py", "start_line": 3}),
    ]
    kept = drop_duplicate_points(points)

    assert [point.id for point in kept] == ["a", "b", "c"]
    assert kept[0].payload["start_line"] == 1
    assert "symbol_name" not in kept[0].payload


def test_drop_duplicate_points_warns_about_each_collision(caplog):
    points = [_FakePoint("a"), _FakePoint("a"), _FakePoint("a")]
    with caplog.at_level("WARNING"):
        kept = drop_duplicate_points(points)

    assert len(kept) == 1
    assert caplog.text.count("Duplicate point id") == 2


def test_drop_duplicate_points_is_a_noop_for_real_chunks():
    """The guard should never fire for ids minted by chunk_point_id()."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "symbol_parser.py")
    points = [
        _FakePoint(chunk_point_id(chunk, path, "repo"), chunk)
        for chunk in get_file_chunks(path)
    ]
    assert len(drop_duplicate_points(points)) == len(points)


def test_upload_vectors_deduplicates_before_upserting(monkeypatch):
    """upload_vectors() must not hand Qdrant a batch with a repeated id."""
    sent = []

    class _Client:
        def upsert(self, collection_name, points):
            sent.extend(points)

    monkeypatch.setattr(indexer, "client", _Client())

    uploaded = indexer.upload_vectors(
        "col",
        [_FakePoint("a"), _FakePoint("a"), _FakePoint("b")],
        batch_size=10,
    )

    assert uploaded == 2
    assert [point.id for point in sent] == ["a", "b"]
