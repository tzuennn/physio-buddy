"""Application factory.

Creates the FastAPI app, mounts routers and static files.
Nothing else — no route handlers, no business logic, no state.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .routers import audio, health, integrations, sessions

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(
        title="Physio Buddy API",
        version="0.2.0",
        description="AI-powered physiotherapy squat coaching.",
    )

    app.include_router(health.router)
    app.include_router(sessions.router)
    app.include_router(audio.router)
    app.include_router(integrations.router)

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    return app


# Module-level instance so `uvicorn physio_buddy.app:app` works directly.
app = create_app()
