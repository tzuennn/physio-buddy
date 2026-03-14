from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .coaching import choose_coaching
from .config import settings
from .fatigue import FatigueEstimator
from .mediapipe_pose import MediaPipePoseAnalyzer, calculate_angles_from_landmarks
from .meralion import MeralionClient
from .models import AudioSignal, Event, FatigueLevel, FrameMetrics, LandmarkInput, RepPhase, SessionReport
from .store import SessionStore
from .summary import build_summary
from .vision import RepStateMachine, assess_form


class StartSessionResponse(BaseModel):
    session_id: str
    safety_notice: str


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
    file_size: int = Field(..., gt=0)


class FileKeyRequest(BaseModel):
    file_key: str


class TranslateAudioRequest(FileKeyRequest):
    language: str


class ProcessAudioRequest(FileKeyRequest):
    instruction: str


app = FastAPI(title="Physio Buddy API", version="0.2.0")
_store = SessionStore()
_rep_trackers: dict[str, RepStateMachine] = {}
_fatigue_estimators: dict[str, FatigueEstimator] = {}
_meralion = MeralionClient(
    base_url=settings.meralion_base_url,
    api_key=settings.meralion_api_key or "",
    timeout_s=settings.meralion_timeout_s,
)
_ALLOWED_AUDIO_TYPES = {"audio/wav", "audio/mpeg", "audio/flac", "audio/mp4"}
_MAX_AUDIO_BYTES = 100 * 1024 * 1024


def _resolve_frame_metrics(payload: IngestPayload) -> FrameMetrics:
    if payload.frame:
        return payload.frame

    if payload.landmarks:
        angles = calculate_angles_from_landmarks(payload.landmarks.model_dump())
        return FrameMetrics(
            knee_angle_deg=angles.knee_angle_deg,
            torso_lean_deg=angles.torso_lean_deg,
            knee_inward_offset=angles.knee_inward_offset,
        )

    if payload.image_base64:
        analyzer = MediaPipePoseAnalyzer()
        angles = analyzer.from_image_base64(payload.image_base64)
        return FrameMetrics(
            knee_angle_deg=angles.knee_angle_deg,
            torso_lean_deg=angles.torso_lean_deg,
            knee_inward_offset=angles.knee_inward_offset,
        )

    raise HTTPException(status_code=422, detail="Provide one of: frame, landmarks, or image_base64")


def _require_meralion() -> None:
    if not _meralion.enabled:
        raise HTTPException(status_code=503, detail="MERALION_API_KEY is not configured")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/integrations/meralion/status")
def meralion_status() -> dict[str, Any]:
    _require_meralion()
    try:
        return _meralion.status()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MERaLiON error: {exc}") from exc


@app.post("/sessions/start", response_model=StartSessionResponse)
def start_session() -> StartSessionResponse:
    session_id = str(uuid4())
    _rep_trackers[session_id] = RepStateMachine()
    _fatigue_estimators[session_id] = FatigueEstimator()
    return StartSessionResponse(
        session_id=session_id,
        safety_notice="Assistive wellness support only, not diagnostic advice. Stop if pain occurs.",
    )


@app.post("/sessions/{session_id}/ingest", response_model=Event)
def ingest(session_id: str, payload: IngestPayload) -> Event:
    if session_id not in _rep_trackers:
        raise HTTPException(status_code=404, detail="Unknown session_id")

    rep_tracker = _rep_trackers[session_id]
    fatigue_estimator = _fatigue_estimators[session_id]

    frame = _resolve_frame_metrics(payload)
    phase = rep_tracker.update(frame.knee_angle_deg)
    form = assess_form(
        knee_angle_deg=frame.knee_angle_deg,
        knee_inward_offset=frame.knee_inward_offset,
        torso_lean_deg=frame.torso_lean_deg,
    )

    fatigue = fatigue_estimator.update(
        tempo_rps=payload.tempo_rps,
        depth_angle=frame.knee_angle_deg,
        rest_gap_s=payload.rest_gap_s,
        strain_score=payload.audio.strain_score,
    )

    if payload.audio.command_intent == "pause":
        fatigue = FatigueLevel.HIGH

    coaching = choose_coaching(form=form, fatigue=fatigue)

    event = Event(
        ts=time.time(),
        frame=frame,
        phase=phase if isinstance(phase, RepPhase) else RepPhase(phase),
        rep_count=rep_tracker.rep_count,
        form=form,
        fatigue_level=fatigue,
        coaching_message=coaching.message,
    )
    _store.add_event(session_id, event)
    return event


@app.post("/audio/upload-url")
def audio_upload_url(payload: UploadUrlRequest) -> dict[str, Any]:
    _require_meralion()
    if payload.file_size > _MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file exceeds 100MB limit")
    if payload.content_type not in _ALLOWED_AUDIO_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported audio content type")
    try:
        return _meralion.upload_url(
            filename=payload.filename,
            content_type=payload.content_type,
            file_size=payload.file_size,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MERaLiON error: {exc}") from exc


@app.post("/audio/upload-file")
def audio_upload_file(file: UploadFile = File(...)) -> dict[str, Any]:
    _require_meralion()
    content_type = file.content_type or "application/octet-stream"
    if content_type not in _ALLOWED_AUDIO_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported audio content type")
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size <= 0:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if file_size > _MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file exceeds 100MB limit")
    try:
        upload = _meralion.upload_url(
            filename=file.filename or "audio",
            content_type=content_type,
            file_size=file_size,
        )
        url = upload.get("url")
        file_key = upload.get("fileKey")
        if not url or not file_key:
            raise HTTPException(status_code=502, detail="MERaLiON upload URL response missing fields")
        _meralion.upload_file(url=url, content=file.file, content_type=content_type)
        return {"fileKey": file_key, "uploaded": True}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MERaLiON error: {exc}") from exc


@app.post("/audio/upload-status")
def audio_upload_status(payload: FileKeyRequest) -> dict[str, Any]:
    _require_meralion()
    try:
        return _meralion.upload_status(file_key=payload.file_key)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MERaLiON error: {exc}") from exc


@app.post("/audio/transcribe")
def audio_transcribe(payload: FileKeyRequest) -> dict[str, Any]:
    _require_meralion()
    try:
        return _meralion.transcribe(file_key=payload.file_key)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MERaLiON error: {exc}") from exc


@app.post("/audio/summarize")
def audio_summarize(payload: FileKeyRequest) -> dict[str, Any]:
    _require_meralion()
    try:
        return _meralion.summarize(file_key=payload.file_key)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MERaLiON error: {exc}") from exc


@app.post("/audio/translate")
def audio_translate(payload: TranslateAudioRequest) -> dict[str, Any]:
    _require_meralion()
    try:
        return _meralion.translate(file_key=payload.file_key, language=payload.language)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MERaLiON error: {exc}") from exc


@app.post("/audio/analyze")
def audio_analyze(payload: FileKeyRequest) -> dict[str, Any]:
    _require_meralion()
    try:
        return _meralion.analyze(file_key=payload.file_key)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MERaLiON error: {exc}") from exc


@app.post("/audio/process")
def audio_process(payload: ProcessAudioRequest) -> dict[str, Any]:
    _require_meralion()
    try:
        return _meralion.process(file_key=payload.file_key, instruction=payload.instruction)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MERaLiON error: {exc}") from exc


@app.get("/sessions/{session_id}/summary", response_model=SessionReport)
def summary(session_id: str) -> SessionReport:
    events = _store.get_events(session_id)
    if not events:
        raise HTTPException(status_code=404, detail="No events found for session_id")
    return build_summary(session_id=session_id, events=events)


@app.post("/sessions/{session_id}/stop", response_model=SessionReport)
def stop(session_id: str) -> SessionReport:
    report = summary(session_id)
    _rep_trackers.pop(session_id, None)
    _fatigue_estimators.pop(session_id, None)
    _store.clear(session_id)
    return report
