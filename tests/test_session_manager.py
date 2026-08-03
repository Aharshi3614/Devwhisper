"""Tests for the centralized conversation session manager."""

import pytest

from session_manager import SessionManager


class FakeClock:
    """Return deterministic, increasing timestamps."""

    def __init__(self):
        self.value = 0.0

    def __call__(self):
        self.value += 1.0
        return self.value


def test_update_and_get_preserve_existing_history_format():
    manager = SessionManager(clock=FakeClock())

    manager.update("session-a", "hello", "hi")
    manager.update("session-a", "question", "answer")

    assert manager.get("session-a") == (
        "User: hello\nAssistant: hi\n\n"
        "User: question\nAssistant: answer"
    )


def test_history_is_trimmed_to_configured_limit():
    manager = SessionManager(
        max_history_per_session=2,
        clock=FakeClock(),
    )

    manager.update("session-a", "one", "1")
    manager.update("session-a", "two", "2")
    manager.update("session-a", "three", "3")

    assert manager.sessions["session-a"]["history"] == [
        "User: two\nAssistant: 2",
        "User: three\nAssistant: 3",
    ]


def test_least_recently_used_session_is_evicted():
    manager = SessionManager(max_sessions=2, clock=FakeClock())

    manager.update("session-a", "a", "a")
    manager.update("session-b", "b", "b")
    assert manager.get("session-a")

    manager.update("session-c", "c", "c")

    assert list(manager.sessions) == ["session-a", "session-c"]
    assert "session-b" not in manager.sessions


def test_missing_session_does_not_create_an_entry():
    manager = SessionManager(clock=FakeClock())

    assert manager.get("missing") == ""
    assert manager.sessions == {}


def test_clear_keeps_shared_store_reference_valid():
    manager = SessionManager(clock=FakeClock())
    shared_reference = manager.sessions

    manager.update("session-a", "hello", "hi")
    manager.clear()

    assert shared_reference is manager.sessions
    assert shared_reference == {}


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("max_sessions", 0),
        ("max_history_per_session", 0),
    ],
)
def test_invalid_limits_are_rejected(keyword, value):
    with pytest.raises(ValueError, match="must be at least 1"):
        SessionManager(**{keyword: value})
