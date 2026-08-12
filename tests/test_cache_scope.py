"""Repository scoping tests for the response cache (issue #258).

`cache.py` keyed entries on the normalized query text alone, in one
process-wide dict with no notion of which repository the answer came from.
After a `POST /repos/switch` the user was served an answer generated from a
different codebase — confidently wrong, and with a "Sources used:" footer
naming files that are not in the active repository.

Generic questions make it easy to hit: "what does this project do", "how do
i run the tests", "where is the entry point" get asked against every
repository, and whichever one was asked first won for as long as the entry
survived in the LRU. The near-duplicate Jaccard matching widened it further
— an entry cached for repo A was reachable from repo B by any query that was
merely 70% similar.

These tests pin the isolation, the unscoped fallback, and the fact that the
LRU bound stays global rather than per-repository.
"""

import pytest

import cache
import repositories


@pytest.fixture(autouse=True)
def clean_cache():
    """Start every test from an empty cache and no active repository."""
    original_repo = repositories.get_current_repo_id()
    cache.clear()
    repositories.set_current_repo(None)
    yield
    cache.clear()
    repositories.set_current_repo(original_repo)


# ---------------------------------------------------------------------------
# Isolation between repositories
# ---------------------------------------------------------------------------

def test_answer_does_not_leak_across_repositories():
    """The regression: repo A's answer served while repo B is active."""
    repositories.set_current_repo("repo_aaaa")
    cache.put("what does main.py do?", "ANSWER FROM REPO A")

    repositories.set_current_repo("repo_bbbb")

    assert cache.get("what does main.py do?") is None


def test_each_repository_keeps_its_own_answer():
    repositories.set_current_repo("repo_aaaa")
    cache.put("what does this project do?", "answer A")

    repositories.set_current_repo("repo_bbbb")
    cache.put("what does this project do?", "answer B")

    repositories.set_current_repo("repo_aaaa")
    assert cache.get("what does this project do?") == "answer A"

    repositories.set_current_repo("repo_bbbb")
    assert cache.get("what does this project do?") == "answer B"

    assert cache.size() == 2, "the same query in two repos is two entries"


def test_near_duplicate_matching_is_scoped_too():
    """Similarity matching must not reach across the repository boundary.

    These two queries are similar enough to match inside one repository, so
    if the scan were unscoped this would return repo A's answer.
    """
    repositories.set_current_repo("repo_aaaa")
    cache.put("how do i run the tests in this project", "answer A")

    repositories.set_current_repo("repo_bbbb")

    assert cache.get("how do i run the tests for this project") is None

    # Same lookup inside repo A still matches — the scan itself still works.
    repositories.set_current_repo("repo_aaaa")
    assert cache.get("how do i run the tests for this project") == "answer A"


def test_switching_away_and_back_restores_the_hit():
    repositories.set_current_repo("repo_aaaa")
    cache.put("where is the entry point?", "answer A")

    repositories.set_current_repo("repo_bbbb")
    assert cache.get("where is the entry point?") is None

    repositories.set_current_repo("repo_aaaa")
    assert cache.get("where is the entry point?") == "answer A"


def test_unscoped_entries_are_not_visible_from_a_repository():
    """An answer cached before any repository was registered stays separate."""
    repositories.set_current_repo(None)
    cache.put("a question", "unscoped answer")

    repositories.set_current_repo("repo_aaaa")
    assert cache.get("a question") is None

    repositories.set_current_repo(None)
    assert cache.get("a question") == "unscoped answer"


# ---------------------------------------------------------------------------
# Single-repository behaviour is unchanged
# ---------------------------------------------------------------------------

def test_exact_match_round_trip():
    cache.put("what is devwhisper?", "it is a voice agent")

    assert cache.get("what is devwhisper?") == "it is a voice agent"


@pytest.mark.parametrize(
    "variant",
    [
        "  what is devwhisper?  ",
        "WHAT IS DEVWHISPER?",
        "what   is    devwhisper?",
    ],
)
def test_normalization_still_applies(variant):
    cache.put("what is devwhisper?", "it is a voice agent")

    assert cache.get(variant) == "it is a voice agent"


def test_near_duplicate_still_matches_within_one_repository():
    repositories.set_current_repo("repo_aaaa")
    cache.put("how do i run the tests in this project", "run pytest")

    assert cache.get("how do i run the tests in the project") == "run pytest"


def test_miss_returns_none():
    cache.put("what is devwhisper?", "an answer")

    assert cache.get("something completely unrelated") is None


def test_empty_response_is_not_cached():
    cache.put("a question", "")
    cache.put("another question", "   ")

    assert cache.size() == 0
    assert cache.get("a question") is None


def test_empty_query_does_not_match_by_similarity():
    cache.put("a real question", "an answer")

    assert cache.get("") is None
    assert cache.get("   ") is None


def test_put_overwrites_the_same_key_in_the_same_repo():
    repositories.set_current_repo("repo_aaaa")
    cache.put("a question", "old answer")
    cache.put("a question", "new answer")

    assert cache.get("a question") == "new answer"
    assert cache.size() == 1


# ---------------------------------------------------------------------------
# LRU bound
# ---------------------------------------------------------------------------

def test_capacity_is_global_not_per_repository():
    """The memory ceiling must not scale with the number of repositories."""
    for i in range(cache.MAX_CACHE_SIZE + 10):
        repositories.set_current_repo(f"repo_{i % 5}")
        cache.put(f"question number {i}", f"answer {i}")

    assert cache.size() == cache.MAX_CACHE_SIZE


def test_least_recently_used_entry_is_evicted_first():
    repositories.set_current_repo("repo_aaaa")

    for i in range(cache.MAX_CACHE_SIZE):
        cache.put(f"unique question {i}", f"answer {i}")

    # Touch the oldest entry so it is no longer least-recently-used.
    assert cache.get("unique question 0") == "answer 0"

    cache.put("one more question that overflows", "overflow answer")

    assert cache.size() == cache.MAX_CACHE_SIZE
    assert cache.get("unique question 0") == "answer 0", "recently used entry survived"
    assert cache.get("unique question 1") is None, "true LRU entry was evicted"


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------

def test_invalidate_repo_drops_only_that_repository():
    repositories.set_current_repo("repo_aaaa")
    cache.put("shared question", "answer A")

    repositories.set_current_repo("repo_bbbb")
    cache.put("shared question", "answer B")

    removed = cache.invalidate_repo("repo_aaaa")

    assert removed == 1
    assert cache.get("shared question") == "answer B"

    repositories.set_current_repo("repo_aaaa")
    assert cache.get("shared question") is None


def test_invalidate_unknown_repo_is_a_no_op():
    repositories.set_current_repo("repo_aaaa")
    cache.put("a question", "an answer")

    assert cache.invalidate_repo("repo_that_does_not_exist") == 0
    assert cache.get("a question") == "an answer"


def test_clear_drops_everything():
    repositories.set_current_repo("repo_aaaa")
    cache.put("q1", "a1")
    repositories.set_current_repo("repo_bbbb")
    cache.put("q2", "a2")

    assert cache.clear() == 2
    assert cache.size() == 0


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_a_broken_registry_does_not_break_answering(monkeypatch):
    """A cache lookup must never raise out into the request path."""
    def boom():
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(repositories, "get_current_repo_id", boom)

    cache.put("a question", "an answer")
    assert cache.get("a question") == "an answer"
