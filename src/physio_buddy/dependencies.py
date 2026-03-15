"""Application-level singletons.

All three long-lived objects (pose analyzer, session manager, MERaLiON client)
live here.  Routers import the getter functions — never the backing globals —
which keeps them easily monkeypatchable in tests.
"""

from __future__ import annotations

from .config import settings
from .mediapipe_pose import MediaPipePoseAnalyzer
from .meralion import MeralionClient
from .session_manager import SessionManager

_pose_analyzer: MediaPipePoseAnalyzer | None = None
_session_manager: SessionManager | None = None
_meralion: MeralionClient | None = None


def get_pose_analyzer() -> MediaPipePoseAnalyzer:
    global _pose_analyzer
    if _pose_analyzer is None:
        _pose_analyzer = MediaPipePoseAnalyzer()
    return _pose_analyzer


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(settings)
    return _session_manager


def get_meralion() -> MeralionClient:
    global _meralion
    if _meralion is None:
        _meralion = MeralionClient(
            base_url=settings.meralion_base_url,
            api_key=settings.meralion_api_key or "",
            timeout_s=settings.meralion_timeout_s,
        )
    return _meralion
