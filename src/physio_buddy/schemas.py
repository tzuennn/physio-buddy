"""API request/response schemas.

Pydantic models that belong to the HTTP layer only — completely separate from
the domain models in models.py.  No business logic here, just shape contracts.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .models import AudioSignal, FrameMetrics, LandmarkInput, PhysioProfile


class StartSessionRequest(BaseModel):
    profile: PhysioProfile = PhysioProfile.GENERAL_MOBILITY


class StartSessionResponse(BaseModel):
    session_id: str
    safety_notice: str
    profile: str


class IngestPayload(BaseModel):
    frame: FrameMetrics | None = None
    landmarks: LandmarkInput | None = None
    image_base64: str | None = None
    audio: AudioSignal
    tempo_rps: float = Field(..., gt=0)
    rest_gap_s: float = Field(..., ge=0)


class UploadUrlRequest(BaseModel):
    filename: str
    content_type: str = "audio/wav"
    file_size: int | None = Field(None, gt=0)


class FileKeyRequest(BaseModel):
    file_key: str


class TranslateAudioRequest(FileKeyRequest):
    language: str


class ProcessAudioRequest(FileKeyRequest):
    instruction: str
