"""Audio / MERaLiON routes."""

from __future__ import annotations

import logging
from typing import Any, NoReturn

import httpx

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..dependencies import get_meralion
from ..schemas import FileKeyRequest, ProcessAudioRequest, TranslateAudioRequest, UploadUrlRequest

_ALLOWED_AUDIO_TYPES = {
    "audio/wav",
    "audio/mpeg",
    "audio/flac",
    "audio/mp4",
    "audio/m4a",
}
_MAX_AUDIO_BYTES = 100 * 1024 * 1024  # 100 MB

router = APIRouter(prefix="/audio")
logger = logging.getLogger(__name__)


def _normalize_content_type(content_type: str | None, filename: str | None = None) -> str:
    ct = (content_type or "").split(";", 1)[0].strip().lower()

    # MERaLiON expects audio/m4a specifically when extension is .m4a.
    if filename and filename.lower().endswith(".m4a") and ct in {"audio/mp4", "audio/m4a", ""}:
        return "audio/m4a"

    if ct:
        return ct

    if filename:
        lower = filename.lower()
        if lower.endswith(".wav"):
            return "audio/wav"
        if lower.endswith(".mp3"):
            return "audio/mpeg"
        if lower.endswith(".flac"):
            return "audio/flac"
        if lower.endswith(".m4a"):
            return "audio/m4a"
        if lower.endswith(".webm"):
            return "audio/webm"
        if lower.endswith(".ogg"):
            return "audio/ogg"

    return "application/octet-stream"


def _require_meralion():
    """Return the MeralionClient or raise 503 if not configured."""
    m = get_meralion()
    if not m.enabled:
        raise HTTPException(status_code=503, detail="MERALION_API_KEY is not configured")
    return m


def _raise_meralion_error(exc: Exception, operation: str, context: dict[str, Any]) -> NoReturn:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code if exc.response is not None else "unknown"
        upstream_body = exc.response.text[:300] if exc.response is not None and exc.response.text else ""
        logger.error(
            "MERaLiON %s failed: status=%s context=%s upstream=%s",
            operation,
            status,
            context,
            upstream_body,
        )
        raise HTTPException(
            status_code=502,
            detail=f"MERaLiON {operation} failed ({status}): {upstream_body or str(exc)}",
        ) from exc

    logger.exception("MERaLiON %s failed: context=%s", operation, context)
    raise HTTPException(status_code=502, detail=f"MERaLiON {operation} error: {exc}") from exc


@router.post("/upload-url")
def upload_url(payload: UploadUrlRequest) -> dict[str, Any]:
    m = _require_meralion()
    content_type = _normalize_content_type(payload.content_type)
    if payload.file_size is not None and payload.file_size > _MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file exceeds 100MB limit")
    if content_type not in _ALLOWED_AUDIO_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported audio content type")
    try:
        kwargs: dict[str, Any] = {
            "filename": payload.filename,
            "content_type": content_type,
        }
        if payload.file_size is not None:
            kwargs["file_size"] = payload.file_size
        return m.upload_url(**kwargs)
    except Exception as exc:
        _raise_meralion_error(
            exc,
            operation="upload-url",
            context={
                "filename": payload.filename,
                "content_type": content_type,
                "file_size": payload.file_size,
            },
        )


@router.post("/upload-file")
def upload_file(file: UploadFile = File(...)) -> dict[str, Any]:
    m = _require_meralion()
    content_type = _normalize_content_type(file.content_type, file.filename)
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
        upload = m.upload_url(
            filename=file.filename or "audio",
            content_type=content_type,
            file_size=file_size,
        )
        url = upload.get("url")
        file_key = upload.get("fileKey")
        if not url or not file_key:
            raise HTTPException(status_code=502, detail="MERaLiON upload URL response missing fields")
        m.upload_file(url=url, content=file.file, content_type=content_type)
        return {"fileKey": file_key, "uploaded": True}
    except HTTPException:
        raise
    except Exception as exc:
        _raise_meralion_error(
            exc,
            operation="upload-file",
            context={
                "filename": file.filename,
                "content_type": content_type,
                "file_size": file_size,
            },
        )


@router.post("/upload-status")
def upload_status(payload: FileKeyRequest) -> dict[str, Any]:
    m = _require_meralion()
    try:
        return m.upload_status(file_key=payload.file_key)
    except Exception as exc:
        _raise_meralion_error(exc, operation="upload-status", context={"file_key": payload.file_key})


@router.post("/transcribe")
def transcribe(payload: FileKeyRequest) -> dict[str, Any]:
    m = _require_meralion()
    try:
        return m.transcribe(file_key=payload.file_key)
    except Exception as exc:
        _raise_meralion_error(exc, operation="transcribe", context={"file_key": payload.file_key})


@router.post("/summarize")
def summarize(payload: FileKeyRequest) -> dict[str, Any]:
    m = _require_meralion()
    try:
        return m.summarize(file_key=payload.file_key)
    except Exception as exc:
        _raise_meralion_error(exc, operation="summarize", context={"file_key": payload.file_key})


@router.post("/translate")
def translate(payload: TranslateAudioRequest) -> dict[str, Any]:
    m = _require_meralion()
    try:
        return m.translate(file_key=payload.file_key, language=payload.language)
    except Exception as exc:
        _raise_meralion_error(
            exc,
            operation="translate",
            context={"file_key": payload.file_key, "language": payload.language},
        )


@router.post("/analyze")
def analyze(payload: FileKeyRequest) -> dict[str, Any]:
    m = _require_meralion()
    try:
        return m.analyze(file_key=payload.file_key)
    except Exception as exc:
        _raise_meralion_error(exc, operation="analyze", context={"file_key": payload.file_key})


@router.post("/process")
def process(payload: ProcessAudioRequest) -> dict[str, Any]:
    m = _require_meralion()
    try:
        return m.process(file_key=payload.file_key, instruction=payload.instruction)
    except Exception as exc:
        _raise_meralion_error(
            exc,
            operation="process",
            context={"file_key": payload.file_key, "instruction_len": len(payload.instruction)},
        )
