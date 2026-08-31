"""
Search directives — regression tests for issue #308.

``extract_filters()`` matched four directive keywords but mapped only three.
``repo:`` was in the regex, so ``re.sub`` deleted it from the query text, but
it had no branch in the if/elif chain, so it never became a filter. The user
lost the words and got no scoping — silently, and against whichever repository
happened to be active.

A second, quieter mismatch sat in the same six lines: ``re.findall`` was called
with ``re.IGNORECASE`` and ``re.sub`` was not, so an upper-case directive was
extracted as a filter *and* left in the query text, where it went on to be
embedded as though it were part of the question.

The tests below pin the directive table, the query text it produces, and the
way ``retrieve()`` merges the result with a caller-supplied filter.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import retriever  # noqa: E402
from query_normalizer import (  # noqa: E402
    QueryNormalizer,
    extract_query_filters,
)


# ---------------------------------------------------------------------------
# The directive table
# ---------------------------------------------------------------------------
def test_repo_directive_becomes_a_filter():
    """The reported failure: stripped from the query, returned as nothing."""
    clean, filters = extract_query_filters("where is login handled repo:backend")
    assert clean == "where is login handled"
    assert filters == {"repository": "backend"}


def test_repo_directive_alongside_a_symbol_directive():
    """
    The worst case in the report: this used to reduce to the query "show"
    with the repository scoping thrown away.
    """
    clean, filters = extract_query_filters("show symbol:retrieve repo:devwhisper")
    assert clean == "show"
    assert filters == {"symbol_name": "retrieve", "repository": "devwhisper"}


def test_the_other_three_directives_are_unchanged():
    clean, filters = extract_query_filters("find file:main.py type:function auth")
    assert clean == "find auth"
    assert filters == {"file": "main.py", "symbol_type": "function"}


def test_all_four_directives_at_once():
    clean, filters = extract_query_filters(
        "explain file:retriever.py type:function symbol:retrieve repo:devwhisper"
    )
    assert clean == "explain"
    assert filters == {
        "file": "retriever.py",
        "symbol_type": "function",
        "symbol_name": "retrieve",
        "repository": "devwhisper",
    }


def test_every_stripped_keyword_is_also_mapped():
    """
    The invariant behind the bug, asserted directly: a keyword that the
    pattern removes from the query must map to a payload field. Adding a
    keyword to the pattern without mapping it fails here.
    """
    normalizer = QueryNormalizer()
    for keyword in QueryNormalizer.DIRECTIVE_FIELDS:
        clean, filters = normalizer.extract_filters(f"question {keyword}:value")
        assert clean == "question", f"{keyword}: left in the query text"
        assert filters, f"{keyword}: stripped from the query but not mapped"


def test_directives_map_to_payload_field_names_not_keywords():
    """`type:` filters `symbol_type`; the payload has no `type` field."""
    _clean, filters = extract_query_filters("q type:class symbol:Thing repo:r")
    assert set(filters) == {"symbol_type", "symbol_name", "repository"}


# ---------------------------------------------------------------------------
# Case handling
# ---------------------------------------------------------------------------
def test_an_uppercase_directive_is_removed_from_the_query():
    """
    `findall` had IGNORECASE and `sub` did not, so this was extracted as a
    filter *and* left in the query, to be embedded as part of the question.
    """
    clean, filters = extract_query_filters("find FILE:main.py auth")
    assert clean == "find auth"
    assert filters == {"file": "main.py"}


@pytest.mark.parametrize("directive", ["REPO:backend", "Repo:backend", "rEpO:backend"])
def test_repo_directive_is_case_insensitive(directive):
    clean, filters = extract_query_filters(f"where is login {directive}")
    assert clean == "where is login"
    assert filters == {"repository": "backend"}


def test_the_directive_value_keeps_its_case():
    """Repository and symbol names are case-sensitive; only the key is not."""
    _clean, filters = extract_query_filters("q REPO:BackEnd symbol:MyClass")
    assert filters == {"repository": "BackEnd", "symbol_name": "MyClass"}


# ---------------------------------------------------------------------------
# Things that are not directives
# ---------------------------------------------------------------------------
def test_a_bare_colon_is_not_a_directive():
    clean, filters = extract_query_filters("what is the ratio 3:4")
    assert filters == {}
    assert "3:4" in clean


def test_an_unknown_keyword_is_left_alone():
    clean, filters = extract_query_filters("look at branch:main")
    assert filters == {}
    assert "branch:main" in clean


def test_an_empty_query_returns_empty():
    assert extract_query_filters("") == ("", {})


# ---------------------------------------------------------------------------
# retrieve(): the caller's dict is not mutated
# ---------------------------------------------------------------------------
@pytest.fixture
def stub_retrieval(monkeypatch):
    """A retriever whose backends return one chunk and record their filters."""
    embedder = MagicMock()
    embedder.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]
    monkeypatch.setattr(retriever, "embedder", embedder)

    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(
                payload={"file": "main.py", "start_line": 1, "text": "def f(): pass"},
                score=0.9,
            )
        ]
    )
    monkeypatch.setattr(retriever, "client", client)
    monkeypatch.setattr(retriever, "_get_bm25", lambda _repo_id: None)
    monkeypatch.setattr(retriever, "check_embedding_version", lambda _repo_id=None: None)
    return client


def test_retrieve_does_not_mutate_the_callers_filter(stub_retrieval):
    """
    A caller that builds one standing filter and reuses it would otherwise
    accumulate every directive any user has ever typed.
    """
    standing = {"symbol_type": "function"}
    original = dict(standing)

    retriever.retrieve("question file:main.py repo:backend", metadata_filter=standing)

    assert standing == original


def test_a_reused_filter_does_not_leak_between_queries(stub_retrieval):
    """The consequence of the mutation, exercised end to end."""
    standing = {"symbol_type": "function"}

    retriever.retrieve("first question file:one.py", metadata_filter=standing)
    retriever.retrieve("second question", metadata_filter=standing)

    assert "file" not in standing


def test_the_explicit_filter_wins_over_an_inline_directive(stub_retrieval, monkeypatch):
    """A caller that passed a filter meant it."""
    captured = {}

    def record(metadata_filter=None, repository_names=None):
        captured["metadata_filter"] = metadata_filter
        return None

    monkeypatch.setattr(
        retriever,
        "_build_qdrant_filter",
        lambda metadata_filter, repository_names=None: record(
            metadata_filter, repository_names
        ),
    )

    retriever.retrieve("question file:inline.py", metadata_filter={"file": "explicit.py"})

    assert captured["metadata_filter"]["file"] == "explicit.py"


# ---------------------------------------------------------------------------
# retrieve(): repo: reaches the repository filtering
# ---------------------------------------------------------------------------
def test_the_repo_directive_reaches_the_repository_filter(stub_retrieval, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        retriever,
        "_build_qdrant_filter",
        lambda metadata_filter, repository_names=None: captured.update(
            metadata_filter=metadata_filter, repository_names=repository_names
        ),
    )

    retriever.retrieve("where is login repo:backend")

    assert captured["repository_names"] == ["backend"]


def test_the_repo_directive_does_not_land_in_the_metadata_filter(
    stub_retrieval, monkeypatch
):
    """
    It routes to `repository_names`, which knows shared-collection mode from
    repo-isolated mode; a raw `repository` key in `metadata_filter` would not.
    """
    captured = {}
    monkeypatch.setattr(
        retriever,
        "_build_qdrant_filter",
        lambda metadata_filter, repository_names=None: captured.update(
            metadata_filter=metadata_filter, repository_names=repository_names
        ),
    )

    retriever.retrieve("where is login repo:backend")

    assert not (captured["metadata_filter"] or {}).get("repository")


def test_an_explicit_repositories_argument_wins_over_the_directive(
    stub_retrieval, monkeypatch
):
    captured = {}
    monkeypatch.setattr(
        retriever,
        "_build_qdrant_filter",
        lambda metadata_filter, repository_names=None: captured.update(
            repository_names=repository_names
        ),
    )

    retriever.retrieve("where is login repo:inline", repositories="explicit")

    assert captured["repository_names"] == ["explicit"]


def test_a_query_without_a_repo_directive_is_unscoped(stub_retrieval, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        retriever,
        "_build_qdrant_filter",
        lambda metadata_filter, repository_names=None: captured.update(
            repository_names=repository_names
        ),
    )

    retriever.retrieve("where is login")

    assert captured["repository_names"] is None


def test_the_directive_is_not_embedded_as_part_of_the_question(stub_retrieval):
    """
    The directive must not reach the embedder — it is scoping, not meaning.
    """
    retriever.retrieve("where is login handled repo:backend")

    encoded = retriever.embedder.encode.call_args[0][0]
    assert "repo:backend" not in encoded
    assert "backend" not in encoded
