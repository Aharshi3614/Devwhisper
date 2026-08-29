"""
Language support wiring — regression tests for issue #309.

``SUPPORTED_EXTENSIONS`` was ``{".py", ".md"}`` and ``collect_indexable_files()``
filters on it *before* any chunking happens, so the JavaScript/TypeScript
extractor merged in #288 could never run: those files were recorded as
``unsupported_extension`` and ``get_file_chunks()`` was never called on them.

The list was written out four times — in `config.py`, in `get_file_chunks()`'s
dispatch, in `symbol_parser`, and again as a literal in the `/index/upload`
validator. These tests pin the wiring end to end rather than each copy, since
the bug was the disagreement between them and not any one of the four.
"""

import io
import os
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import indexer  # noqa: E402
from config import (  # noqa: E402
    EXTENSION_LANGUAGES,
    SUPPORTED_EXTENSIONS,
    SYMBOL_EXTENSIONS,
)
from indexer import (  # noqa: E402
    collect_indexable_files,
    detect_language,
    get_file_chunks,
)


TS_SOURCE = """\
export function greet(name: string): string {
    return `hello ${name}`;
}

export class Greeter {
    greet(name: string): string {
        return greet(name);
    }
}
"""

JS_SOURCE = """\
function add(a, b) {
    return a + b;
}

class Calculator {
    multiply(a, b) {
        return a * b;
    }
}
"""

PY_SOURCE = "def helper():\n    return 1\n"


# ---------------------------------------------------------------------------
# The allowlist no longer gates off the extractor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("extension", sorted(SYMBOL_EXTENSIONS))
def test_every_symbol_extension_is_indexable(extension):
    """
    The invariant behind the bug: an extension the extractor handles must not
    be filtered out before the extractor is reached.
    """
    assert extension in SUPPORTED_EXTENSIONS


def test_typescript_files_are_collected_for_indexing(tmp_path):
    (tmp_path / "app.ts").write_text(TS_SOURCE)
    (tmp_path / "notes.md").write_text("# notes\n")

    files, skipped = collect_indexable_files(str(tmp_path))

    assert any(f.endswith("app.ts") for f in files)
    assert not [s for s in skipped if s["reason"] == "unsupported_extension"]


@pytest.mark.parametrize(
    "filename,source",
    [
        ("app.ts", TS_SOURCE),
        ("app.tsx", TS_SOURCE),
        ("app.js", JS_SOURCE),
        ("app.jsx", JS_SOURCE),
        ("app.mjs", JS_SOURCE),
    ],
)
def test_symbols_are_extracted_from_js_and_ts(tmp_path, filename, source):
    """The extractor was already written and merged; it just never ran."""
    filepath = tmp_path / filename
    filepath.write_text(source)

    chunks = get_file_chunks(str(filepath))
    symbol_names = {c.get("symbol_name") for c in chunks if c.get("is_symbol")}

    assert symbol_names, f"no symbols extracted from {filename}"


def test_a_typescript_file_is_chunked_on_symbol_boundaries(tmp_path):
    filepath = tmp_path / "app.ts"
    filepath.write_text(TS_SOURCE)

    chunks = get_file_chunks(str(filepath))
    symbols = {c["symbol_name"] for c in chunks if c.get("is_symbol")}

    assert "greet" in symbols
    assert "Greeter" in symbols


def test_python_and_markdown_are_still_indexable():
    assert ".py" in SUPPORTED_EXTENSIONS
    assert ".md" in SUPPORTED_EXTENSIONS


def test_an_unsupported_extension_is_still_skipped(tmp_path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "keep.py").write_text(PY_SOURCE)

    files, skipped = collect_indexable_files(str(tmp_path))

    assert not any(f.endswith(".png") for f in files)
    assert any(s["path"].endswith(".png") for s in skipped)


def test_markdown_is_not_symbol_chunked(tmp_path):
    """Markdown is indexable but has no extractor; it stays line-based."""
    assert ".md" not in SYMBOL_EXTENSIONS

    filepath = tmp_path / "notes.md"
    filepath.write_text("# heading\n\nbody text\n")

    chunks = get_file_chunks(str(filepath))
    assert chunks
    assert not any(c.get("is_symbol") for c in chunks)


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "filename,expected",
    [
        ("mod.py", "python"),
        ("app.js", "javascript"),
        ("app.jsx", "javascript"),
        ("app.mjs", "javascript"),
        ("app.ts", "typescript"),
        ("app.tsx", "typescript"),
        ("notes.md", "markdown"),
    ],
)
def test_detect_language(filename, expected):
    assert detect_language(f"/somewhere/{filename}") == expected


def test_detect_language_is_case_insensitive():
    assert detect_language("/somewhere/App.TS") == "typescript"


def test_detect_language_of_an_unknown_extension_is_none():
    assert detect_language("/somewhere/image.png") is None


def test_every_supported_extension_has_a_language():
    """A file we index but cannot name the language of is a wiring gap."""
    for extension in SUPPORTED_EXTENSIONS:
        assert extension in EXTENSION_LANGUAGES, extension


@pytest.mark.parametrize(
    "filename,source,expected",
    [("app.ts", TS_SOURCE, "typescript"), ("mod.py", PY_SOURCE, "python")],
)
def test_symbol_chunks_record_the_language(tmp_path, filename, source, expected):
    filepath = tmp_path / filename
    filepath.write_text(source)

    chunks = get_file_chunks(str(filepath))
    assert chunks
    for chunk in chunks:
        assert chunk["language"] == expected


def test_line_chunks_record_the_language(tmp_path):
    filepath = tmp_path / "notes.md"
    filepath.write_text("# heading\n\nbody\n")

    for chunk in indexer.chunk_file(str(filepath)):
        assert chunk["language"] == "markdown"


# ---------------------------------------------------------------------------
# The upload validator uses the same constant
# ---------------------------------------------------------------------------
def _zip_with(*members):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, body in members:
            archive.writestr(name, body)
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    import main

    return TestClient(main.app)


@pytest.fixture(autouse=True)
def _sweep_queued_uploads():
    """
    Remove the archives the accepted uploads leave behind.

    A successful POST queues the job and returns; the worker thread that would
    delete the temp archive is not running under the test client, so the file
    stays in the working directory. That leak is issue #310 and is fixed
    separately — this fixture only keeps these tests from adding to it.
    """
    import glob

    before = set(glob.glob("temp_upload_*.zip"))
    yield
    for path in set(glob.glob("temp_upload_*.zip")) - before:
        try:
            os.remove(path)
        except OSError:
            pass


def test_a_typescript_only_archive_is_accepted(client):
    """
    The reported symptom: a TypeScript project was rejected at upload for
    containing no supported files, by an endpoint whose indexer could read it.
    """
    payload = _zip_with(("src/app.ts", TS_SOURCE))

    response = client.post(
        "/index/upload",
        files={"file": ("project.zip", payload, "application/zip")},
    )

    assert response.status_code == 200, response.json()


def test_a_javascript_only_archive_is_accepted(client):
    payload = _zip_with(("src/index.js", JS_SOURCE))

    response = client.post(
        "/index/upload",
        files={"file": ("project.zip", payload, "application/zip")},
    )

    assert response.status_code == 200, response.json()


def test_an_archive_with_no_supported_file_is_still_rejected(client):
    payload = _zip_with(("readme.txt", "text"), ("logo.png", "binary"))

    response = client.post(
        "/index/upload",
        files={"file": ("project.zip", payload, "application/zip")},
    )

    assert response.status_code == 400


def test_the_rejection_message_lists_the_configured_extensions(client):
    """
    It used to say "(.py, .md)" from a literal beside the check. Deriving it
    means the message cannot disagree with what the endpoint accepts.
    """
    payload = _zip_with(("readme.txt", "text"))

    response = client.post(
        "/index/upload",
        files={"file": ("project.zip", payload, "application/zip")},
    )

    message = response.json()["message"]
    for extension in SUPPORTED_EXTENSIONS:
        assert extension in message


# ---------------------------------------------------------------------------
# The override
# ---------------------------------------------------------------------------
def test_the_extension_set_can_be_narrowed_by_configuration(monkeypatch):
    from config import _env_extensions

    monkeypatch.setenv("SUPPORTED_EXTENSIONS", ".py,.md")
    assert _env_extensions("SUPPORTED_EXTENSIONS", frozenset()) == frozenset(
        {".py", ".md"}
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("py,md", {".py", ".md"}),
        (".PY, .Md", {".py", ".md"}),
        (".py,,.md,", {".py", ".md"}),
        ("  .ts  ", {".ts"}),
    ],
)
def test_the_override_normalises_its_entries(monkeypatch, raw, expected):
    from config import _env_extensions

    monkeypatch.setenv("SUPPORTED_EXTENSIONS", raw)
    assert _env_extensions("SUPPORTED_EXTENSIONS", frozenset()) == frozenset(expected)


def test_an_unset_override_keeps_the_default(monkeypatch):
    from config import _env_extensions

    monkeypatch.delenv("SUPPORTED_EXTENSIONS", raising=False)
    default = frozenset({".py"})
    assert _env_extensions("SUPPORTED_EXTENSIONS", default) is default


def test_a_blank_override_keeps_the_default(monkeypatch):
    from config import _env_extensions

    monkeypatch.setenv("SUPPORTED_EXTENSIONS", "   ")
    default = frozenset({".py"})
    assert _env_extensions("SUPPORTED_EXTENSIONS", default) is default
