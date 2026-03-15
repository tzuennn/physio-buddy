from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FatigueLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class PhysioProfile(str, Enum):
    POST_OP_CONSERVATIVE = "post_op_conservative"
    KNEE_REHAB = "knee_rehab"
    GENERAL_MOBILITY = "general_mobility"
    PERFORMANCE = "performance"


class RepPhase(str, Enum):
    STAND = "STAND"
    DESCEND = "DESCEND"
    BOTTOM = "BOTTOM"
    ASCEND = "ASCEND"


class FormAssessment(BaseModel):
    depth_quality: Literal["good", "shallow"]
    knee_tracking_warning: bool
    torso_lean_warning: bool


class FrameMetrics(BaseModel):
    knee_angle_deg: float = Field(..., ge=0, le=180)
    torso_lean_deg: float = Field(..., ge=0, le=90)
    knee_inward_offset: float = Field(..., ge=-1.0, le=1.0)


class LandmarkInput(BaseModel):
    hip: tuple[float, float]
    knee: tuple[float, float]
    ankle: tuple[float, float]
    shoulder: tuple[float, float]
    foot_index: tuple[float, float]


class AudioSignal(BaseModel):
    command_intent: Literal["start", "pause", "resume", "stop", "reps-left", "none"] = "none"
    strain_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    valid: bool | None = None
    audio_age_ms: int | None = Field(None, ge=0)


class Event(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    ts: float = Field(..., ge=0)
    frame: FrameMetrics
    phase: RepPhase
    rep_count: int = Field(..., ge=0)
    form: FormAssessment
    fatigue_level: FatigueLevel
    fatigue_reason: str | None = None
    coaching_message: str
    vision_valid: bool = True
    audio_valid: bool = True
    vision_age_ms: int = Field(0, ge=0)
    audio_age_ms: int = Field(0, ge=0)


class SessionReport(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    session_id: str
    total_reps: int
    valid_reps: int
    shallow_rep_count: int
    knee_warning_count: int
    torso_warning_count: int
    fatigue_timeline: list[tuple[float, FatigueLevel]]
    notable_events: list[str]
    recommendation: str
