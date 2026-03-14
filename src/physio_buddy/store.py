from __future__ import annotations

from dataclasses import dataclass, field

from .models import Event


@dataclass
class SessionStore:
    _events: dict[str, list[Event]] = field(default_factory=dict)

    def add_event(self, session_id: str, event: Event) -> None:
        self._events.setdefault(session_id, []).append(event)

    def get_events(self, session_id: str) -> list[Event]:
        return list(self._events.get(session_id, []))

    def clear(self, session_id: str) -> None:
        self._events.pop(session_id, None)
