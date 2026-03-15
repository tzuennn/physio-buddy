"""Session lifecycle manager.

Single responsibility: own every per-session data structure so that no router
or route handler needs to touch a bare dict.  Routers ask for a SessionManager
via the dependencies layer and call create / get / close — that's it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .coaching import AdaptiveCoachAgent, LLMCoachAgent
from .config import Settings
from .fatigue import FatigueEstimator
from .models import Event, FrameMetrics
from .store import SessionStore
from .vision import RepStateMachine, VisionThresholds, profile_state_machine, profile_thresholds
from uuid import uuid4


@dataclass
class SessionComponents:
    """All mutable state belonging to one active session."""

    profile: str
    rep_tracker: RepStateMachine
    fatigue_estimator: FatigueEstimator
    coach_agent: AdaptiveCoachAgent | LLMCoachAgent
    thresholds: VisionThresholds
    last_good_frame: Optional[FrameMetrics] = None
    last_vision_ts: float | None = None
    last_audio_ts: float | None = None


class SessionManager:
    """Creates, retrieves, and destroys sessions.

    The underlying event store is also owned here so the manager is the single
    source of truth for everything session-scoped.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._store = SessionStore()
        self._sessions: dict[str, SessionComponents] = {}

    @property
    def store(self) -> SessionStore:
        return self._store

    def create(self, profile: str) -> tuple[str, SessionComponents]:
        """Allocate a new session and return its ID and components."""
        session_id = str(uuid4())
        components = SessionComponents(
            profile=profile,
            rep_tracker=profile_state_machine(profile),
            fatigue_estimator=FatigueEstimator(),
            coach_agent=self._make_coach_agent(profile),
            thresholds=profile_thresholds(profile),
        )
        self._sessions[session_id] = components
        return session_id, components

    def get(self, session_id: str) -> SessionComponents | None:
        return self._sessions.get(session_id)

    def close(self, session_id: str) -> list[Event]:
        """Tear down a session and return its recorded events."""
        events = self._store.get_events(session_id)
        self._sessions.pop(session_id, None)
        self._store.clear(session_id)
        return events

    def _make_coach_agent(self, profile: str) -> AdaptiveCoachAgent | LLMCoachAgent:
        s = self._settings
        if s.llm_api_url:
            return LLMCoachAgent(
                profile=profile,
                api_url=s.llm_api_url,
                api_key=s.llm_api_key or "",
                model=s.llm_model,
                timeout_s=s.llm_timeout_s,
            )
        return AdaptiveCoachAgent()
