"""Session routes: start, ingest, summary, stop."""

from __future__ import annotations

import logging
import time
import traceback

from fastapi import APIRouter, Body, HTTPException

from ..coaching import AdaptiveCoachAgent, LLMCoachAgent, choose_coaching
from ..dependencies import get_pose_analyzer, get_session_manager
from ..mediapipe_pose import calculate_angles_from_landmarks
from ..models import Event, FatigueLevel, FrameMetrics, PhysioProfile, RepPhase, SessionReport
from ..schemas import IngestPayload, StartSessionRequest, StartSessionResponse
from ..summary import build_summary
from ..vision import assess_form

router = APIRouter(prefix="/sessions")
logger = logging.getLogger(__name__)


def _empty_summary(session_id: str) -> SessionReport:
    return SessionReport(
        session_id=session_id,
        total_reps=0,
        valid_reps=0,
        shallow_rep_count=0,
        knee_warning_count=0,
        torso_warning_count=0,
        fatigue_timeline=[],
        notable_events=[],
        recommendation="No squat frames were recorded for this session.",
    )


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
        analyzer = get_pose_analyzer()
        angles = analyzer.from_image_base64(payload.image_base64)
        return FrameMetrics(
            knee_angle_deg=angles.knee_angle_deg,
            torso_lean_deg=angles.torso_lean_deg,
            knee_inward_offset=angles.knee_inward_offset,
        )

    raise HTTPException(status_code=422, detail="Provide one of: frame, landmarks, or image_base64")


@router.post("/start", response_model=StartSessionResponse)
def start_session(body: StartSessionRequest | None = Body(default=None)) -> StartSessionResponse:
    profile = body.profile.value if body else PhysioProfile.GENERAL_MOBILITY.value
    sm = get_session_manager()
    session_id, _ = sm.create(profile)
    return StartSessionResponse(
        session_id=session_id,
        safety_notice="Assistive wellness support only, not diagnostic advice. Stop if pain occurs.",
        profile=profile,
    )


@router.post("/{session_id}/ingest", response_model=Event)
async def ingest(session_id: str, payload: IngestPayload) -> Event:
    sm = get_session_manager()
    components = sm.get(session_id)
    if components is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")

    now_ts = time.time()
    vision_valid = True
    frame: FrameMetrics
    try:
        frame = _resolve_frame_metrics(payload)
        components.last_good_frame = frame
        components.last_vision_ts = now_ts
    except (ValueError, RuntimeError) as exc:
        vision_valid = False
        logger.info(
            "Frame ingest degraded: session_id=%s reason=%s image_b64=%s landmarks=%s frame=%s",
            session_id,
            str(exc),
            bool(payload.image_base64),
            bool(payload.landmarks),
            bool(payload.frame),
        )
        if components.last_good_frame is None:
            raise HTTPException(status_code=422, detail=f"Frame error: {exc}") from exc
        frame = components.last_good_frame
    except HTTPException:
        raise
    except Exception as exc:
        print(f"ERROR in frame resolution: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Processing error: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        incoming_audio_valid = (
            payload.audio.valid
            if payload.audio.valid is not None
            else payload.audio.confidence >= 0.4
        )
        if incoming_audio_valid:
            components.last_audio_ts = now_ts

        vision_age_ms = (
            max(0, int((now_ts - components.last_vision_ts) * 1000))
            if components.last_vision_ts is not None
            else 0
        )
        if payload.audio.audio_age_ms is not None:
            audio_age_ms = max(0, int(payload.audio.audio_age_ms))
        elif components.last_audio_ts is not None:
            audio_age_ms = max(0, int((now_ts - components.last_audio_ts) * 1000))
        else:
            audio_age_ms = 0

        audio_valid = incoming_audio_valid and audio_age_ms <= 25000
        strain_for_fatigue = payload.audio.strain_score if audio_valid else 0.0

        phase = components.rep_tracker.update(frame.knee_angle_deg)
        form = assess_form(
            knee_angle_deg=frame.knee_angle_deg,
            knee_inward_offset=frame.knee_inward_offset,
            torso_lean_deg=frame.torso_lean_deg,
            thresholds=components.thresholds,
        )

        fatigue, fatigue_reason = components.fatigue_estimator.update_with_reason(
            tempo_rps=payload.tempo_rps,
            depth_angle=frame.knee_angle_deg,
            rest_gap_s=payload.rest_gap_s,
            strain_score=strain_for_fatigue,
        )

        if payload.audio.command_intent == "pause":
            fatigue = FatigueLevel.HIGH
            fatigue_reason = "user requested pause"

        # Rule-based safety net; overridden by adaptive/LLM agent below.
        coaching = choose_coaching(form=form, fatigue=fatigue)
        try:
            agent = components.coach_agent
            normalized_phase = phase if isinstance(phase, RepPhase) else RepPhase(phase)
            if isinstance(agent, LLMCoachAgent):
                coaching = await agent.decide(
                    form=form,
                    fatigue=fatigue,
                    fatigue_reason=fatigue_reason,
                    phase=normalized_phase,
                    rep_count=components.rep_tracker.rep_count,
                )
            else:
                coaching = agent.decide(
                    form=form,
                    fatigue=fatigue,
                    fatigue_reason=fatigue_reason,
                    phase=normalized_phase,
                    rep_count=components.rep_tracker.rep_count,
                )
        except Exception:
            pass

        event = Event(
            ts=time.time(),
            frame=frame,
            phase=phase if isinstance(phase, RepPhase) else RepPhase(phase),
            rep_count=components.rep_tracker.rep_count,
            form=form,
            fatigue_level=fatigue,
            fatigue_reason=fatigue_reason,
            coaching_message=coaching.message,
            vision_valid=vision_valid,
            audio_valid=audio_valid,
            vision_age_ms=vision_age_ms,
            audio_age_ms=audio_age_ms,
        )
        sm.store.add_event(session_id, event)
        return event

    except Exception as exc:
        print(f"ERROR in analysis: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Analysis error: {type(exc).__name__}: {exc}"
        ) from exc


@router.get("/{session_id}/summary", response_model=SessionReport)
def summary(session_id: str) -> SessionReport:
    sm = get_session_manager()
    events = sm.store.get_events(session_id)
    if not events:
        raise HTTPException(status_code=404, detail="No events found for session_id")
    return build_summary(session_id=session_id, events=events)


@router.post("/{session_id}/stop", response_model=SessionReport)
def stop(session_id: str) -> SessionReport:
    sm = get_session_manager()
    events = sm.close(session_id)
    return build_summary(session_id=session_id, events=events) if events else _empty_summary(session_id)
