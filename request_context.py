"""request_context.py — Central Request Context Object for DevWhisper.

Encapsulates request lifecycle data, metadata, session information, and transient pipeline state.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import uuid

@dataclass
class RequestContext:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_query: str = ""
    session_id: Optional[str] = None
    repo_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)

    def set_state(self, key: str, value: Any) -> None:
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)
