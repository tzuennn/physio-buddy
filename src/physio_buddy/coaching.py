from __future__ import annotations

from dataclasses import dataclass

from .models import FatigueLevel, FormAssessment


@dataclass(slots=True)
class CoachingDecision:
    message: str
    should_pause: bool = False
    reduce_reps_by: int = 0


def choose_coaching(form: FormAssessment, fatigue: FatigueLevel) -> CoachingDecision:
    if fatigue == FatigueLevel.HIGH:
        return CoachingDecision(
            message="High fatigue detected. Pause for 30 seconds and reduce remaining reps by 2.",
            should_pause=True,
            reduce_reps_by=2,
        )

    if form.knee_tracking_warning:
        return CoachingDecision(message="Keep knees aligned with toes; gently push knees outward.")
    if form.torso_lean_warning:
        return CoachingDecision(message="Chest up and brace core to reduce forward lean.")
    if form.depth_quality == "shallow":
        return CoachingDecision(message="Go deeper slowly until thighs are near parallel.")

    if fatigue == FatigueLevel.MEDIUM:
        return CoachingDecision(message="Good form. Control tempo and take a short breath reset.")

    return CoachingDecision(message="Great rep. Keep a steady rhythm.")
