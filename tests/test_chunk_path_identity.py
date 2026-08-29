"""
Chunk path identity — regression tests for issue #307.

Chunks used to carry only ``os.path.basename(filepath)``. That single missing
field broke ``chunk_point_id()`` in two opposite ways depending on which of
the two call sites computed the id:

  * ``generate_embeddings()`` fell back to the basename, so two files with the
    same name in different directories produced one id and ``upsert()`` kept
    only one of them — requirement (2), uniqueness, violated;
  * ``index_directory()`` keyed on the absolute walked path, so moving the
    repository re-keyed every point in the collection — requirement (1),
    stability, violated.

Both now key on ``chunk["path"]``, a repository-relative, forward-slash
separated path. These tests pin down both requirements at once, since a fix
for either one alone is easy to write and wrong.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from indexer import (  # noqa: E402
    chunk_file,
    chunk_point_id,
    chunk_relative_path,
    generate_chunks,
    get_file_chunks,
)


SIMPLE_MODULE = "def helper():\n    return 1\n"

MODULE_WITH_CLASS = (
    "import os\n"
    "\n"
    "\n"
    "class Thing:\n"
    "    \"\"\"A thing.\"\"\"\n"
    "\n"
    "    def run(self):\n"
    "        return 2\n"
    "\n"
    "\n"
    "def helper():\n"
    "    return 1\n"
)


@pytest.fixture
def repo(tmp_path):
    """A repository with the same filename in two different packages."""
    for package in ("a", "b"):
        directory = tmp_path / package
        directory.mkdir()
        (directory / "utils.py").write_text(SIMPLE_MODULE)
    return tmp_path


# ---------------------------------------------------------------------------
# chunk_relative_path()
# ---------------------------------------------------------------------------
def test_relative_path_is_taken_relative_to_the_root(tmp_path):
    filepath = os.path.join(str(tmp_path), "pkg", "mod.py")
    assert chunk_relative_path(filepath, str(tmp_path)) == "pkg/mod.py"


def test_relative_path_normalises_separators_to_forward_slashes(tmp_path):
    filepath = os.path.join(str(tmp_path), "a", "b", "c.py")
    result = chunk_relative_path(filepath, str(tmp_path))
    assert "\\" not in result
    assert result == "a/b/c.py"


def test_relative_path_without_a_root_falls_back_to_the_basename():
    assert chunk_relative_path("/somewhere/deep/mod.py") == "mod.py"


def test_relative_path_outside_the_root_falls_back_to_the_basename(tmp_path):
    """
    ``os.path.relpath`` would express this with ``..`` segments, which are not
    a repository-relative path and would leak the layout above the root.
    """
    outside = os.path.join(os.path.dirname(str(tmp_path)), "elsewhere", "mod.py")
    assert chunk_relative_path(outside, str(tmp_path)) == "mod.py"


def test_relative_path_of_a_file_at_the_root_is_just_its_name(tmp_path):
    filepath = os.path.join(str(tmp_path), "main.py")
    assert chunk_relative_path(filepath, str(tmp_path)) == "main.py"


def test_relative_path_of_an_empty_path_is_empty():
    assert chunk_relative_path("", "/root") == ""


# ---------------------------------------------------------------------------
# Every chunk producer records a path
# ---------------------------------------------------------------------------
def test_chunk_file_records_the_relative_path(tmp_path):
    directory = tmp_path / "pkg"
    directory.mkdir()
    filepath = directory / "notes.md"
    filepath.write_text("# heading\n\nbody\n")

    chunks = chunk_file(str(filepath), rel_root=str(tmp_path))

    assert chunks
    for chunk in chunks:
        assert chunk["path"] == "pkg/notes.md"
        assert chunk["file"] == "notes.md"


def test_symbol_chunks_record_the_relative_path(tmp_path):
    directory = tmp_path / "pkg"
    directory.mkdir()
    filepath = directory / "mod.py"
    filepath.write_text(MODULE_WITH_CLASS)

    chunks = get_file_chunks(str(filepath), rel_root=str(tmp_path))

    assert any(chunk.get("is_symbol") for chunk in chunks), "expected symbol chunks"
    for chunk in chunks:
        assert chunk["path"] == "pkg/mod.py"


def test_module_context_chunks_record_the_relative_path(tmp_path):
    """The non-symbol gaps between top-level definitions carry it too."""
    directory = tmp_path / "pkg"
    directory.mkdir()
    filepath = directory / "mod.py"
    filepath.write_text(MODULE_WITH_CLASS)

    chunks = get_file_chunks(str(filepath), rel_root=str(tmp_path))
    module_chunks = [c for c in chunks if c.get("chunk_type") == "module_context"]

    assert module_chunks, "expected at least one module_context chunk"
    for chunk in module_chunks:
        assert chunk["path"] == "pkg/mod.py"


def test_the_basename_is_still_recorded_alongside_the_path(tmp_path):
    """``file`` is used for display and by existing payloads; it must stay."""
    directory = tmp_path / "deep" / "nested"
    directory.mkdir(parents=True)
    filepath = directory / "mod.py"
    filepath.write_text(SIMPLE_MODULE)

    for chunk in get_file_chunks(str(filepath), rel_root=str(tmp_path)):
        assert chunk["file"] == "mod.py"
        assert chunk["path"] == "deep/nested/mod.py"


def test_chunks_without_a_root_still_carry_a_path(tmp_path):
    """A standalone caller gets the basename rather than a missing key."""
    filepath = tmp_path / "mod.py"
    filepath.write_text(SIMPLE_MODULE)

    for chunk in get_file_chunks(str(filepath)):
        assert chunk["path"] == "mod.py"


# ---------------------------------------------------------------------------
# Requirement (2): unique per chunk
# ---------------------------------------------------------------------------
def test_same_basename_in_two_packages_no_longer_collides(repo):
    """The failure reported in #307, reproduced directly."""
    a = get_file_chunks(str(repo / "a" / "utils.py"), rel_root=str(repo))[0]
    b = get_file_chunks(str(repo / "b" / "utils.py"), rel_root=str(repo))[0]

    assert a["path"] == "a/utils.py"
    assert b["path"] == "b/utils.py"

    id_a = chunk_point_id(a, a["path"], "repo")
    id_b = chunk_point_id(b, b["path"], "repo")
    assert id_a != id_b


def test_the_basename_alone_would_still_collide(repo):
    """
    Guards the fix rather than the symptom: keying on ``file`` reproduces the
    bug, so a future change that quietly reverts to it fails here.
    """
    a = get_file_chunks(str(repo / "a" / "utils.py"), rel_root=str(repo))[0]
    b = get_file_chunks(str(repo / "b" / "utils.py"), rel_root=str(repo))[0]

    assert chunk_point_id(a, a["file"], "repo") == chunk_point_id(b, b["file"], "repo")


def test_generate_chunks_gives_every_file_a_distinct_identity(repo):
    chunks, _cache, _imports = generate_chunks(
        [str(repo / "a" / "utils.py"), str(repo / "b" / "utils.py")],
        rel_root=str(repo),
    )

    ids = [chunk_point_id(c, c["path"], "repo") for c in chunks]
    assert len(ids) == len(set(ids)), "generate_chunks produced colliding ids"


def test_deeply_nested_files_with_one_name_are_all_distinct(tmp_path):
    """__init__.py in a package tree is the common real-world case."""
    paths = []
    for package in ("one", "two", "three"):
        directory = tmp_path / "src" / package
        directory.mkdir(parents=True)
        filepath = directory / "__init__.py"
        filepath.write_text(SIMPLE_MODULE)
        paths.append(str(filepath))

    chunks, _cache, _imports = generate_chunks(paths, rel_root=str(tmp_path))
    ids = {chunk_point_id(c, c["path"], "repo") for c in chunks}
    assert len(ids) == len(chunks)


def test_two_repositories_sharing_a_relative_path_do_not_collide(repo):
    """Shared-collection mode: the repository name still separates them."""
    chunk = get_file_chunks(str(repo / "a" / "utils.py"), rel_root=str(repo))[0]

    assert chunk_point_id(chunk, chunk["path"], "repo-one") != chunk_point_id(
        chunk, chunk["path"], "repo-two"
    )


# ---------------------------------------------------------------------------
# Requirement (1): stable across checkouts
# ---------------------------------------------------------------------------
def test_ids_survive_the_repository_moving(tmp_path):
    """
    The same repository content, checked out at two different absolute paths,
    must produce identical ids — otherwise every re-index orphans the previous
    run's points instead of updating them.
    """
    ids = []
    for checkout in ("/first/location", "/second/location"):
        root = tmp_path / checkout.strip("/").replace("/", "_")
        package = root / "pkg"
        package.mkdir(parents=True)
        filepath = package / "mod.py"
        filepath.write_text(MODULE_WITH_CLASS)

        chunks = get_file_chunks(str(filepath), rel_root=str(root))
        ids.append([chunk_point_id(c, c["path"], "repo") for c in chunks])

    assert ids[0] == ids[1]


def test_the_absolute_path_would_not_survive_the_move(tmp_path):
    """The other half of the regression: pins why ``rel_root`` is required."""
    absolute_ids = []
    for name in ("first", "second"):
        root = tmp_path / name
        root.mkdir()
        filepath = root / "mod.py"
        filepath.write_text(SIMPLE_MODULE)
        chunk = get_file_chunks(str(filepath), rel_root=str(root))[0]
        absolute_ids.append(chunk_point_id(chunk, str(filepath), "repo"))

    assert absolute_ids[0] != absolute_ids[1]


def test_reindexing_an_unchanged_file_reproduces_the_same_ids(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    filepath = package / "mod.py"
    filepath.write_text(MODULE_WITH_CLASS)

    first = [
        chunk_point_id(c, c["path"], "repo")
        for c in get_file_chunks(str(filepath), rel_root=str(tmp_path))
    ]
    second = [
        chunk_point_id(c, c["path"], "repo")
        for c in get_file_chunks(str(filepath), rel_root=str(tmp_path))
    ]
    assert first == second


def test_editing_a_file_changes_only_the_chunks_that_moved(tmp_path):
    """
    Appending a function must not re-key the ones above it, or an incremental
    re-index rewrites the whole file's worth of points.
    """
    package = tmp_path / "pkg"
    package.mkdir()
    filepath = package / "mod.py"
    filepath.write_text(MODULE_WITH_CLASS)

    before = {
        chunk_point_id(c, c["path"], "repo")
        for c in get_file_chunks(str(filepath), rel_root=str(tmp_path))
    }

    filepath.write_text(MODULE_WITH_CLASS + "\n\ndef added():\n    return 3\n")
    after = {
        chunk_point_id(c, c["path"], "repo")
        for c in get_file_chunks(str(filepath), rel_root=str(tmp_path))
    }

    assert before & after, "expected the untouched symbols to keep their ids"
    assert after - before, "expected the new symbol to get a new id"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------
def test_a_payload_without_a_path_still_gets_an_id():
    """Points indexed before this field existed must not raise."""
    legacy = {"text": "x", "file": "mod.py", "start_line": 1, "is_symbol": False}
    assert chunk_point_id(legacy, legacy.get("path") or legacy["file"], "repo")


def test_rel_root_is_optional_for_every_public_chunker(tmp_path):
    filepath = tmp_path / "mod.py"
    filepath.write_text(SIMPLE_MODULE)

    assert chunk_file(str(filepath))
    assert get_file_chunks(str(filepath))
    assert generate_chunks([str(filepath)])[0]
