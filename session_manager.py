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
        self.lock = RLock()  # Thread-safe lock to prevent race conditions during concurrent ASGI requests

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

    def delete_session(self, session_id: str) -> bool:
        """Delete a single session by session_id. Returns True if existed and deleted."""
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
            return False

    def export_session(self, session_id: str) -> dict[str, Any] | None:
        """Export session data with metadata and structured messages."""
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                return None
            history_list = list(session.get("history", []))
            return {
                "session_id": session_id,
                "last_used": session.get("last_used", self._clock()),
                "message_count": len(history_list),
                "history": history_list,
                "exported_at": self._clock(),
            }

    def import_session(self, session_data: dict[str, Any]) -> str:
        """Import a session into the manager, validating required fields."""
        if not isinstance(session_data, dict):
            raise ValueError("Session data must be a valid dictionary")
        session_id = session_data.get("session_id")
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Session data must contain a non-empty string 'session_id'")

        raw_history = session_data.get("history", [])
        if not isinstance(raw_history, list):
            raw_history = []

        history = [str(item) for item in raw_history]
        overflow = len(history) - self.max_history_per_session
        if overflow > 0:
            history = history[overflow:]

        last_used = float(session_data.get("last_used", self._clock()))

        with self.lock:
            self.sessions[session_id] = {
                "history": history,
                "last_used": last_used,
            }
            self.sessions.move_to_end(session_id)
            self._evict_if_needed_locked()

        return session_id

    def evict_if_needed(self) -> None:
        """Remove least-recently-used sessions until capacity is respected."""
        with self.lock:
            self._evict_if_needed_locked()

    def _evict_if_needed_locked(self) -> None:
        while len(self.sessions) > self.max_sessions:
            self.sessions.popitem(last=False)
