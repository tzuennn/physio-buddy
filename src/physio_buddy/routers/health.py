"""Health and static-file routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

router = APIRouter()


@router.get("/")
def demo() -> FileResponse:
    index = _STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Demo UI not available")
    return FileResponse(index)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
