"""Third-party integration status routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ..dependencies import get_meralion

router = APIRouter(prefix="/integrations")


@router.get("/meralion/status")
def meralion_status() -> dict[str, Any]:
    m = get_meralion()
    if not m.enabled:
        raise HTTPException(status_code=503, detail="MERALION_API_KEY is not configured")
    try:
        return m.status()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MERaLiON error: {exc}") from exc
