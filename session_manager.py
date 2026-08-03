"""Thread-safe in-memory conversation session management."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from threading import RLock
from typing import Any


SessionData = dict[str, Any]


class SessionManager:
    """Maintain bounded per-session conversation history using LRU ordering."""

    def __init__(
        self,
        max_sessions: int = 100,
        max_history_per_session: int = 5,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        if max_history_per_session < 1:
            raise ValueError("max_history_per_session must be at least 1")

        self.max_sessions = max_sessions
        self.max_history_per_session = max_history_per_session
        self._clock = clock
        self.sessions: OrderedDict[str, SessionData] = OrderedDict()
        self.lock = RLock()

    def update(self, session_id: str, user: str, assistant: str) -> None:
        """Append an exchange and promote the session to most recently used."""
        with self.lock:
            session = self.sessions.setdefault(
                session_id,
                {"history": [], "last_used": self._clock()},
            )
            session["history"].append(
                f"User: {user}\nAssistant: {assistant}"
            )

            overflow = len(session["history"]) - self.max_history_per_session
            if overflow > 0:
                del session["history"][:overflow]

            session["last_used"] = self._clock()
            self.sessions.move_to_end(session_id)
            self._evict_if_needed_locked()

    def get(self, session_id: str) -> str:
        """Return formatted history and promote an existing session."""
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                return ""

            session["last_used"] = self._clock()
            self.sessions.move_to_end(session_id)
            return "\n\n".join(session["history"])

    def clear(self) -> None:
        """Remove all active sessions without replacing the shared store."""
        with self.lock:
            self.sessions.clear()

    def evict_if_needed(self) -> None:
        """Remove least-recently-used sessions until capacity is respected."""
        with self.lock:
            self._evict_if_needed_locked()

    def _evict_if_needed_locked(self) -> None:
        while len(self.sessions) > self.max_sessions:
            self.sessions.popitem(last=False)
