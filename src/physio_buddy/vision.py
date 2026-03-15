from __future__ import annotations

from dataclasses import dataclass

from .models import FormAssessment, RepPhase


@dataclass(slots=True)
class VisionThresholds:
    depth_good_min: float = 90.0
    depth_good_max: float = 125.0
    depth_shallow_min: float = 120.0
    knee_warning_offset: float = -0.08
    torso_warning_max: float = 35.0


class RepStateMachine:
    """Simple squat rep state machine for STAND->DESCEND->BOTTOM->ASCEND->STAND."""

    def __init__(self, descend_angle: float = 155.0, bottom_angle: float = 125.0, ascend_angle: float = 145.0) -> None:
        self.phase = RepPhase.STAND
        self.rep_count = 0
        self._descend_angle = descend_angle
        self._bottom_angle = bottom_angle
        self._ascend_angle = ascend_angle

    def update(self, knee_angle_deg: float) -> RepPhase:
        if self.phase == RepPhase.STAND and knee_angle_deg < self._descend_angle:
            self.phase = RepPhase.DESCEND
        elif self.phase == RepPhase.DESCEND and knee_angle_deg <= self._bottom_angle:
            self.phase = RepPhase.BOTTOM
        elif self.phase == RepPhase.BOTTOM and knee_angle_deg > self._ascend_angle:
            self.phase = RepPhase.ASCEND
        elif self.phase == RepPhase.ASCEND and knee_angle_deg >= self._descend_angle:
            self.phase = RepPhase.STAND
            self.rep_count += 1
        return self.phase


def assess_form(knee_angle_deg: float, knee_inward_offset: float, torso_lean_deg: float, thresholds: VisionThresholds | None = None) -> FormAssessment:
    t = thresholds or VisionThresholds()

    depth_quality = "good" if t.depth_good_min <= knee_angle_deg <= t.depth_good_max else "shallow"
    if knee_angle_deg < t.depth_shallow_min and depth_quality != "good":
        depth_quality = "good"

    return FormAssessment(
        depth_quality=depth_quality,
        knee_tracking_warning=knee_inward_offset < t.knee_warning_offset,
        torso_lean_warning=torso_lean_deg > t.torso_warning_max,
    )


def profile_thresholds(profile: str) -> VisionThresholds:
    """Return VisionThresholds appropriate for the given PhysioProfile value."""
    match profile:
        case "post_op_conservative":
            # Very shallow squats are perfectly acceptable; deep squats are flagged
            return VisionThresholds(
                depth_good_min=135.0, depth_good_max=165.0, depth_shallow_min=158.0,
                knee_warning_offset=-0.06, torso_warning_max=30.0,
            )
        case "knee_rehab":
            # Moderate depth; knee valgus is the top priority
            return VisionThresholds(
                depth_good_min=115.0, depth_good_max=142.0, depth_shallow_min=138.0,
                knee_warning_offset=-0.06,
            )
        case "performance":
            # Deeper squats accepted; tighter depth window
            return VisionThresholds(
                depth_good_min=70.0, depth_good_max=100.0, depth_shallow_min=95.0,
                torso_warning_max=40.0,
            )
        case _:  # general_mobility
            return VisionThresholds()


def profile_state_machine(profile: str) -> RepStateMachine:
    """Return a RepStateMachine with angle thresholds for the given PhysioProfile value."""
    match profile:
        case "post_op_conservative":
            return RepStateMachine(descend_angle=163.0, bottom_angle=140.0, ascend_angle=158.0)
        case "knee_rehab":
            return RepStateMachine(descend_angle=158.0, bottom_angle=130.0, ascend_angle=150.0)
        case "performance":
            return RepStateMachine(descend_angle=150.0, bottom_angle=100.0, ascend_angle=138.0)
        case _:  # general_mobility
            return RepStateMachine()
