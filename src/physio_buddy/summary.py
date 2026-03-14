from __future__ import annotations

from collections import Counter

from .models import Event, FatigueLevel, SessionReport


def build_summary(session_id: str, events: list[Event]) -> SessionReport:
    total_reps = max((e.rep_count for e in events), default=0)
    shallow = sum(1 for e in events if e.form.depth_quality == "shallow")
    knee_warn = sum(1 for e in events if e.form.knee_tracking_warning)
    torso_warn = sum(1 for e in events if e.form.torso_lean_warning)
    valid_reps = max(total_reps - max(1, shallow // 8), 0) if total_reps else 0

    fatigue_timeline = [(e.ts, e.fatigue_level) for e in events[:: max(1, len(events) // 10 or 1)]]

    fatigue_counts = Counter(e.fatigue_level for e in events)
    top_fatigue = fatigue_counts.most_common(1)[0][0] if fatigue_counts else FatigueLevel.LOW

    notable_events = []
    if knee_warn:
        notable_events.append("Frequent knee-in warnings; emphasize hip-knee alignment.")
    if torso_warn:
        notable_events.append("Torso lean exceeded safe threshold repeatedly.")
    if top_fatigue == FatigueLevel.HIGH:
        notable_events.append("High fatigue periods detected; introduce longer rest intervals.")

    recommendation = "Continue current plan with gradual progression."
    if shallow > total_reps:
        recommendation = "Prioritize depth mechanics with tempo squats before increasing volume."
    elif top_fatigue in {FatigueLevel.MEDIUM, FatigueLevel.HIGH}:
        recommendation = "Reduce next session target reps by 10-20% and increase rest periods."

    return SessionReport(
        session_id=session_id,
        total_reps=total_reps,
        valid_reps=valid_reps,
        shallow_rep_count=shallow,
        knee_warning_count=knee_warn,
        torso_warning_count=torso_warn,
        fatigue_timeline=fatigue_timeline,
        notable_events=notable_events,
        recommendation=recommendation,
    )
