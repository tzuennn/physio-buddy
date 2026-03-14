from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
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
_pose_analyzer: MediaPipePoseAnalyzer | None = None


def _get_pose_analyzer() -> MediaPipePoseAnalyzer:
    global _pose_analyzer
    if _pose_analyzer is None:
        _pose_analyzer = MediaPipePoseAnalyzer()
    return _pose_analyzer


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
        analyzer = _get_pose_analyzer()
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



@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Physio Buddy Live MVP Demo</title>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: system-ui, sans-serif; max-width: 1080px; margin: 1rem auto; padding: 0 1rem; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    .card { border: 1px solid #8a8a8a55; border-radius: 10px; padding: .9rem; }
    button { margin: .25rem .35rem .25rem 0; padding: .45rem .75rem; }
    video { width: 100%; border-radius: 8px; background: #000; aspect-ratio: 4/3; }
    pre { max-height: 340px; overflow: auto; border-radius: 8px; padding: .75rem; background: #00000012; }
    .row { display: flex; gap: .75rem; flex-wrap: wrap; }
    .metric { font-size: .95rem; padding: .2rem .55rem; border-radius: 999px; border: 1px solid #8a8a8a55; }
  </style>
</head>
<body>
  <h1>Physio Buddy — Live Webcam + Mic MVP</h1>
  <p>This page uses your <strong>real camera and microphone</strong> in-browser and streams frame/audio-derived signals to the backend coaching endpoints.</p>
  <div class="row">
    <button id="enableMedia">1) Enable Camera & Mic</button>
    <button id="startSession" disabled>2) Start Session</button>
    <button id="startLive" disabled>3) Start Live Coaching</button>
    <button id="stopLive" disabled>Stop Live Coaching</button>
    <button id="stopSession" disabled>Stop Session + Summary</button>
  </div>

  <div class="grid">
    <section class="card">
      <h3>Live Camera</h3>
      <video id="video" autoplay playsinline muted></video>
      <canvas id="canvas" width="640" height="480" style="display:none"></canvas>
      <p>Session ID: <code id="sid">none</code></p>
      <div class="row">
        <span class="metric">Rep count: <strong id="repCount">0</strong></span>
        <span class="metric">Fatigue: <strong id="fatigue">n/a</strong></span>
        <span class="metric">Fatigue signs: <strong id="fatigueSigns">0</strong></span>
        <span class="metric">Voice intent: <strong id="intent">none</strong></span>
      </div>
      <p><strong>Coach:</strong> <span id="coachMsg">Waiting...</span></p>
      <p><strong>Status:</strong> <span id="status">Idle</span></p>
    </section>

    <section class="card">
      <h3>Event Log</h3>
      <pre id="out">Ready.</pre>
      <p>Useful links: <a href="/docs" target="_blank">Swagger</a> · <a href="/redoc" target="_blank">ReDoc</a> · <a href="/health" target="_blank">Health</a></p>
    </section>
  </div>

  <script>
    let mediaStream = null;
    let sessionId = null;
    let loopHandle = null;
    let audioContext = null;
    let analyser = null;
    let previousTick = performance.now();
    let live = false;
    let latestIntent = 'none';
    let speechRecognizer = null;
    let speechSynthesisEnabled = 'speechSynthesis' in window;
    let lastSpokenMessage = '';

    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const out = document.getElementById('out');
    const statusEl = document.getElementById('status');

    function log(msg, obj) {
      const line = obj ? `${msg}
${JSON.stringify(obj, null, 2)}` : msg;
      out.textContent = `${new Date().toLocaleTimeString()} ${line}

` + out.textContent;
    }

    async function api(path, payload) {
      const response = await fetch(path, {
        method: payload ? 'POST' : 'GET',
        headers: payload ? { 'content-type': 'application/json' } : undefined,
        body: payload ? JSON.stringify(payload) : undefined,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(`${response.status} ${JSON.stringify(data)}`);
      }
      return data;
    }

    async function enableMedia() {
      mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      video.srcObject = mediaStream;

      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(mediaStream);
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);

      setupSpeechRecognition();
      if (!speechSynthesisEnabled) {
        log('SpeechSynthesis API unavailable. Audio coaching playback disabled.');
      }
      statusEl.textContent = 'Media enabled';
      document.getElementById('startSession').disabled = false;
      log('Camera/mic enabled.');
    }

    function getStrainScore() {
      if (!analyser) return 0.1;
      const buffer = new Float32Array(analyser.fftSize);
      analyser.getFloatTimeDomainData(buffer);
      let sum = 0;
      for (let i = 0; i < buffer.length; i++) sum += buffer[i] * buffer[i];
      const rms = Math.sqrt(sum / buffer.length);
      return Math.max(0, Math.min(1, rms * 8));
    }

    function estimateVisualFatigue(frameMetrics) {
      let signs = 0;
      if (frameMetrics.torso_lean_deg > 32) signs += 1;
      if (frameMetrics.knee_angle_deg > 138) signs += 1; // repeatedly shallow reps
      if (frameMetrics.knee_inward_offset < -0.10) signs += 1;
      return signs;
    }

    function speakCoachMessage(message) {
      if (!speechSynthesisEnabled || !message || message === lastSpokenMessage) return;
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(message);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;
      window.speechSynthesis.speak(utterance);
      lastSpokenMessage = message;
    }

    function captureBase64Frame() {
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL('image/jpeg', 0.72).split(',')[1];
    }

    function setupSpeechRecognition() {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR) {
        log('SpeechRecognition API unavailable in this browser. Voice intent will stay "none".');
        return;
      }
      speechRecognizer = new SR();
      const recognizer = speechRecognizer;
      recognizer.lang = 'en-US';
      recognizer.continuous = true;
      recognizer.interimResults = true;
      recognizer.onresult = (event) => {
        const text = Array.from(event.results).map(r => r[0].transcript).join(' ').toLowerCase();
        if (text.includes('pause')) latestIntent = 'pause';
        else if (text.includes('resume') || text.includes('continue')) latestIntent = 'resume';
        else if (text.includes('stop')) latestIntent = 'stop';
        else if (text.includes('start')) latestIntent = 'start';
        else if (text.includes('reps left')) latestIntent = 'reps-left';
        else latestIntent = 'none';
        document.getElementById('intent').textContent = latestIntent;
      };
      recognizer.onerror = (e) => log('Speech recognition error', e.error);
      recognizer.onend = () => { if (mediaStream && live) recognizer.start(); };
      recognizer.start();
    }

    async function startSession() {
      const start = await api('/sessions/start', {});
      sessionId = start.session_id;
      document.getElementById('sid').textContent = sessionId;
      document.getElementById('startLive').disabled = false;
      document.getElementById('stopSession').disabled = false;
      statusEl.textContent = 'Session started';
      log('Session started', start);
    }

    async function liveTick() {
      if (!live || !sessionId) return;
      const now = performance.now();
      const dt = (now - previousTick) / 1000;
      previousTick = now;

      const payload = {
        image_base64: captureBase64Frame(),
        audio: {
          command_intent: latestIntent,
          strain_score: getStrainScore(),
          confidence: 0.85,
        },
        tempo_rps: 1.0,
        rest_gap_s: Math.max(0, dt),
      };

      try {
        const event = await api(`/sessions/${sessionId}/ingest`, payload);
        document.getElementById('repCount').textContent = event.rep_count;
        document.getElementById('fatigue').textContent = event.fatigue_level;
        document.getElementById('coachMsg').textContent = event.coaching_message;
        const fatigueSigns = estimateVisualFatigue(event.frame);
        document.getElementById('fatigueSigns').textContent = String(fatigueSigns);
        if (event.fatigue_level === 'HIGH' || fatigueSigns >= 2) {
          speakCoachMessage('Fatigue signs detected. Please pause and recover for thirty seconds.');
        } else {
          speakCoachMessage(event.coaching_message);
        }
        statusEl.textContent = 'Live coaching running';
      } catch (err) {
        statusEl.textContent = 'Ingest error';
        log('Ingest failed. If detail mentions MediaPipe, install vision deps: pip install -e .[vision]', String(err));
        stopLive();
      }
    }

    function startLive() {
      live = true;
      previousTick = performance.now();
      document.getElementById('startLive').disabled = true;
      document.getElementById('stopLive').disabled = false;
      loopHandle = setInterval(liveTick, 1200);
      if (speechRecognizer) { try { speechRecognizer.start(); } catch (e) {} }
      statusEl.textContent = 'Starting live loop...';
    }

    function stopLive() {
      live = false;
      if (loopHandle) clearInterval(loopHandle);
      loopHandle = null;
      document.getElementById('startLive').disabled = false;
      document.getElementById('stopLive').disabled = true;
      if (speechRecognizer) speechRecognizer.stop();
      if (speechSynthesisEnabled) window.speechSynthesis.cancel();
      statusEl.textContent = 'Live loop stopped';
    }

    async function stopSession() {
      if (!sessionId) return;
      stopLive();
      const report = await api(`/sessions/${sessionId}/stop`, {});
      statusEl.textContent = 'Session stopped';
      log('Session summary', report);
    }

    document.getElementById('enableMedia').onclick = () => enableMedia().catch((e) => log('Media permission failed', String(e)));
    document.getElementById('startSession').onclick = () => startSession().catch((e) => log('Start session failed', String(e)));
    document.getElementById('startLive').onclick = () => startLive();
    document.getElementById('stopLive').onclick = () => stopLive();
    document.getElementById('stopSession').onclick = () => stopSession().catch((e) => log('Stop session failed', String(e)));
  </script>
</body>
</html>
"""


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

    try:
        frame = _resolve_frame_metrics(payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"MediaPipe unavailable: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Pose extraction failed: {exc}") from exc
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
