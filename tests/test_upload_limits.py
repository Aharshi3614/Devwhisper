"""
Upload size limits and temp-archive cleanup — regression tests for issue #310.

``/index/upload`` accepted an archive of unlimited size, wrote it to the
working directory, and handed it to ``extractall()`` with no bound on what it
expanded to. Nothing checked the request body, the archive on disk, the
extracted total, the per-entry size, or the entry count. ``MAX_FILE_SIZE_BYTES``
exists but is applied by ``collect_indexable_files()``, *after* extraction has
already written the bytes.

The archive that motivated this is 199 KB, expands to 200 MB at a 1027:1 ratio,
and passes all three checks the endpoint already performed — it is a
well-formed ZIP, it contains no traversing member, and it contains a ``.py``.

Cleanup was a single unguarded ``os.remove()`` placed *after* the extraction,
so an extraction that raised left the archive in the working directory forever.
"""

import io
import os
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from archive_safety import (  # noqa: E402
    ArchiveTooLarge,
    UnsafeArchiveMember,
    inspect_archive,
    is_safe_member,
    safe_extract_all,
    stream_to_file,
    sweep_orphan_uploads,
    validate_archive_limits,
)

KB = 1024
MB = 1024 * 1024


def _build_zip(path, members, compress=zipfile.ZIP_DEFLATED):
    with zipfile.ZipFile(path, "w", compress) as archive:
        for name, body in members:
            archive.writestr(name, body)
    return str(path)


# ---------------------------------------------------------------------------
# stream_to_file()
# ---------------------------------------------------------------------------
def test_a_stream_under_the_limit_is_written(tmp_path):
    destination = str(tmp_path / "out.bin")
    written = stream_to_file(io.BytesIO(b"x" * 1000), destination, 10 * KB)

    assert written == 1000
    assert os.path.getsize(destination) == 1000


def test_a_stream_over_the_limit_is_refused(tmp_path):
    destination = str(tmp_path / "out.bin")

    with pytest.raises(ArchiveTooLarge):
        stream_to_file(io.BytesIO(b"x" * (3 * MB)), destination, 1 * MB)


def test_a_refused_stream_leaves_no_partial_file(tmp_path):
    """The point of streaming is to not keep what we just refused."""
    destination = str(tmp_path / "out.bin")

    with pytest.raises(ArchiveTooLarge):
        stream_to_file(io.BytesIO(b"x" * (3 * MB)), destination, 1 * MB)

    assert not os.path.exists(destination)


def test_a_stream_exactly_at_the_limit_is_accepted(tmp_path):
    destination = str(tmp_path / "out.bin")
    assert stream_to_file(io.BytesIO(b"x" * 1000), destination, 1000) == 1000


def test_a_non_positive_limit_disables_the_check(tmp_path):
    destination = str(tmp_path / "out.bin")
    assert stream_to_file(io.BytesIO(b"x" * (2 * MB)), destination, 0) == 2 * MB


def test_the_limit_message_names_the_limit(tmp_path):
    """"Too large" alone gives the uploader nothing to act on."""
    with pytest.raises(ArchiveTooLarge, match="1 MB"):
        stream_to_file(io.BytesIO(b"x" * (2 * MB)), str(tmp_path / "o.bin"), 1 * MB)


# ---------------------------------------------------------------------------
# inspect_archive() / validate_archive_limits()
# ---------------------------------------------------------------------------
def test_inspect_reports_the_expansion(tmp_path):
    path = _build_zip(tmp_path / "a.zip", [("a.py", "\0" * (4 * MB)), ("b.md", "#")])
    stats = inspect_archive(path)

    assert stats["entry_count"] == 2
    assert stats["declared_total"] >= 4 * MB
    assert stats["ratio"] > 50


def test_the_reported_bomb_is_rejected(tmp_path):
    """
    The archive from the report: ~199 KB on disk, 200 MB declared, 1027:1.
    Passes is_zipfile, the Zip Slip scan, and the .py/.md scan.
    """
    path = _build_zip(
        tmp_path / "bomb.zip", [("a.py", "\0" * (200 * MB)), ("keep.md", "#")]
    )

    assert zipfile.is_zipfile(path)
    assert os.path.getsize(path) < MB

    with pytest.raises(ArchiveTooLarge):
        validate_archive_limits(
            path,
            max_extracted_bytes=50 * MB,
            max_entries=20000,
            max_ratio=100.0,
        )


def test_an_ordinary_archive_passes(tmp_path):
    path = _build_zip(
        tmp_path / "ok.zip",
        [("src/app.py", "def f():\n    return 1\n" * 50), ("README.md", "# hi\n")],
    )

    stats = validate_archive_limits(
        path, max_extracted_bytes=500 * MB, max_entries=20000, max_ratio=100.0
    )
    assert stats["entry_count"] == 2


def test_too_many_entries_is_rejected(tmp_path):
    path = _build_zip(
        tmp_path / "many.zip", [(f"f{i}.py", "x") for i in range(50)]
    )

    with pytest.raises(ArchiveTooLarge, match="entries"):
        validate_archive_limits(
            path, max_extracted_bytes=500 * MB, max_entries=10, max_ratio=1000.0
        )


def test_the_ratio_is_checked_independently_of_the_total(tmp_path):
    """
    A small upload engineered to expand enormously is the shape of a bomb
    rather than of a large project, even when it fits the absolute budget.
    """
    path = _build_zip(
        tmp_path / "ratio.zip", [("a.py", "\0" * (8 * MB)), ("b.py", "\0" * MB)]
    )

    with pytest.raises(ArchiveTooLarge, match="ratio"):
        validate_archive_limits(
            path, max_extracted_bytes=500 * MB, max_entries=20000, max_ratio=10.0
        )


def test_a_single_member_archive_is_exempt_from_the_ratio_check(tmp_path):
    """One highly compressible file is unremarkable; the totals still bound it."""
    path = _build_zip(tmp_path / "one.zip", [("a.py", "\0" * (2 * MB))])

    validate_archive_limits(
        path, max_extracted_bytes=500 * MB, max_entries=20000, max_ratio=2.0
    )


def test_an_empty_archive_does_not_divide_by_zero(tmp_path):
    path = _build_zip(tmp_path / "empty.zip", [])
    stats = validate_archive_limits(
        path, max_extracted_bytes=MB, max_entries=10, max_ratio=10.0
    )
    assert stats["entry_count"] == 0


def test_nothing_is_decompressed_to_validate(tmp_path):
    """Validation reads the central directory only."""
    path = _build_zip(tmp_path / "b.zip", [("a.py", "\0" * (400 * MB))])
    before = tmp_path.stat().st_size

    with pytest.raises(ArchiveTooLarge):
        validate_archive_limits(
            path, max_extracted_bytes=10 * MB, max_entries=100, max_ratio=1e9
        )

    assert tmp_path.stat().st_size == before
    assert os.path.getsize(path) < 2 * MB


# ---------------------------------------------------------------------------
# is_safe_member()
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name", ["src/app.py", "a/b/c.py", "README.md", "./x.py", "deep/../ok.py"]
)
def test_safe_members_are_accepted(name):
    assert is_safe_member(name)


@pytest.mark.parametrize(
    "name",
    [
        "../evil.py",
        "../../etc/passwd",
        "/etc/passwd",
        "a/../../evil.py",
        "..\\..\\evil.py",
        "",
    ],
)
def test_traversing_members_are_rejected(name):
    assert not is_safe_member(name)


def test_a_windows_style_traversal_is_caught_on_posix():
    """
    os.path.normpath treats backslashes as ordinary characters on POSIX, so a
    check that does not fold them lets this through as one long filename.
    """
    assert not is_safe_member("..\\..\\evil.py")


# ---------------------------------------------------------------------------
# safe_extract_all()
# ---------------------------------------------------------------------------
def test_extraction_writes_the_members(tmp_path):
    path = _build_zip(tmp_path / "a.zip", [("src/app.py", "code\n"), ("r.md", "# hi")])
    destination = tmp_path / "out"

    written = safe_extract_all(path, str(destination), max_extracted_bytes=10 * MB)

    assert (destination / "src" / "app.py").read_text() == "code\n"
    assert (destination / "r.md").read_text() == "# hi"
    assert written == len("code\n") + len("# hi")


def test_extraction_stops_at_the_budget(tmp_path):
    """
    The declared sizes were already checked; this counts the bytes actually
    written, because a ZIP may declare one size and deliver another.
    """
    path = _build_zip(tmp_path / "big.zip", [("a.py", "\0" * (8 * MB))])

    with pytest.raises(ArchiveTooLarge):
        safe_extract_all(path, str(tmp_path / "out"), max_extracted_bytes=MB)


def test_a_failed_extraction_leaves_no_partial_tree(tmp_path):
    """Half a repository is worse than none — the indexer would index it."""
    path = _build_zip(tmp_path / "big.zip", [("a.py", "\0" * (8 * MB))])
    destination = tmp_path / "out"

    with pytest.raises(ArchiveTooLarge):
        safe_extract_all(path, str(destination), max_extracted_bytes=MB)

    assert not destination.exists()


def test_extraction_refuses_a_traversing_member(tmp_path):
    path = str(tmp_path / "evil.zip")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escaped.py", "pwned")

    with pytest.raises(UnsafeArchiveMember):
        safe_extract_all(path, str(tmp_path / "out"), max_extracted_bytes=10 * MB)


def test_a_traversing_member_writes_nothing_outside(tmp_path):
    path = str(tmp_path / "evil.zip")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escaped.py", "pwned")

    with pytest.raises(UnsafeArchiveMember):
        safe_extract_all(path, str(tmp_path / "out"), max_extracted_bytes=10 * MB)

    assert not (tmp_path / "escaped.py").exists()


def test_directory_members_are_created(tmp_path):
    path = str(tmp_path / "d.zip")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("pkg/", "")
        archive.writestr("pkg/mod.py", "x = 1\n")

    destination = tmp_path / "out"
    safe_extract_all(path, str(destination), max_extracted_bytes=10 * MB)

    assert (destination / "pkg" / "mod.py").exists()


def test_a_non_positive_budget_disables_the_extraction_limit(tmp_path):
    path = _build_zip(tmp_path / "a.zip", [("a.py", "x" * 5000)])
    assert safe_extract_all(path, str(tmp_path / "out"), max_extracted_bytes=0) == 5000


# ---------------------------------------------------------------------------
# sweep_orphan_uploads()
# ---------------------------------------------------------------------------
def test_the_sweep_removes_orphaned_archives(tmp_path):
    for name in ("temp_upload_a.zip", "temp_upload_b.zip"):
        (tmp_path / name).write_bytes(b"x")

    assert sweep_orphan_uploads(str(tmp_path)) == 2
    assert not list(tmp_path.glob("temp_upload_*.zip"))


def test_the_sweep_leaves_everything_else_alone(tmp_path):
    (tmp_path / "temp_upload_a.zip").write_bytes(b"x")
    (tmp_path / "main.py").write_text("x = 1\n")
    (tmp_path / "archive.zip").write_bytes(b"x")
    (tmp_path / "temp_upload_notes.txt").write_text("keep")

    assert sweep_orphan_uploads(str(tmp_path)) == 1
    assert (tmp_path / "main.py").exists()
    assert (tmp_path / "archive.zip").exists()
    assert (tmp_path / "temp_upload_notes.txt").exists()


def test_the_sweep_ignores_directories(tmp_path):
    (tmp_path / "temp_upload_dir.zip").mkdir()
    assert sweep_orphan_uploads(str(tmp_path)) == 0


def test_the_sweep_on_a_clean_directory_is_a_no_op(tmp_path):
    assert sweep_orphan_uploads(str(tmp_path)) == 0


def test_the_sweep_on_a_missing_directory_does_not_raise(tmp_path):
    assert sweep_orphan_uploads(str(tmp_path / "nope")) == 0


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    import main

    return TestClient(main.app)


@pytest.fixture(autouse=True)
def _sweep_between_tests():
    import glob

    before = set(glob.glob("temp_upload_*.zip"))
    yield
    for path in set(glob.glob("temp_upload_*.zip")) - before:
        try:
            os.remove(path)
        except OSError:
            pass


def _post(client, payload, filename="project.zip"):
    return client.post(
        "/index/upload",
        files={"file": (filename, payload, "application/zip")},
    )


def test_the_endpoint_rejects_a_bomb(client, tmp_path):
    path = _build_zip(
        tmp_path / "bomb.zip", [("a.py", "\0" * (600 * MB)), ("keep.md", "#")]
    )
    response = _post(client, open(path, "rb").read())

    assert response.status_code == 413


def test_a_rejected_bomb_leaves_no_temp_archive(client, tmp_path):
    import glob

    before = set(glob.glob("temp_upload_*.zip"))
    path = _build_zip(
        tmp_path / "bomb.zip", [("a.py", "\0" * (600 * MB)), ("keep.md", "#")]
    )
    _post(client, open(path, "rb").read())

    assert set(glob.glob("temp_upload_*.zip")) == before


def test_an_ordinary_archive_is_still_accepted(client, tmp_path):
    path = _build_zip(
        tmp_path / "ok.zip",
        [("src/app.py", "def f():\n    return 1\n"), ("README.md", "# hi\n")],
    )
    response = _post(client, open(path, "rb").read())

    assert response.status_code == 200, response.json()


def test_an_invalid_archive_leaves_no_temp_file(client):
    import glob

    before = set(glob.glob("temp_upload_*.zip"))
    _post(client, b"this is not a zip file at all")

    assert set(glob.glob("temp_upload_*.zip")) == before


def test_a_rejected_upload_reports_the_limit(client, tmp_path):
    path = _build_zip(
        tmp_path / "bomb.zip", [("a.py", "\0" * (600 * MB)), ("keep.md", "#")]
    )
    response = _post(client, open(path, "rb").read())

    assert "MB" in response.json()["message"]
