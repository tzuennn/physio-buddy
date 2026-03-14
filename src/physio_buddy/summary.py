from __future__ import annotations

from collections import Counter

from .models import Event, FatigueLevel, SessionReport


def _count_shallow_reps(events: list[Event], total_reps: int) -> int:
    """Count shallow quality at rep-level (not frame-level)."""
    if total_reps <= 0:
        return 0

    shallow_reps = 0
    current_rep = 0
    rep_had_shallow = False

    for event in sorted(events, key=lambda e: e.ts):
        if event.form.depth_quality == "shallow":
            rep_had_shallow = True

        if event.rep_count > current_rep:
            completed_reps = event.rep_count - current_rep

            if rep_had_shallow:
                shallow_reps += 1

            current_rep = event.rep_count
            rep_had_shallow = False

            if completed_reps > 1:
                # Defensive fallback for skipped counters in sparse streams.
                current_rep = event.rep_count

    return min(shallow_reps, total_reps)


def build_summary(session_id: str, events: list[Event]) -> SessionReport:
    total_reps = max((e.rep_count for e in events), default=0)
    shallow_rep_count = _count_shallow_reps(events, total_reps)
    knee_warn = sum(1 for e in events if e.form.knee_tracking_warning)
    torso_warn = sum(1 for e in events if e.form.torso_lean_warning)
    valid_reps = max(total_reps - shallow_rep_count, 0)

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
    if total_reps > 0 and (shallow_rep_count / total_reps) >= 0.4:
        recommendation = "Prioritize depth mechanics with tempo squats before increasing volume."
    elif top_fatigue in {FatigueLevel.MEDIUM, FatigueLevel.HIGH}:
        recommendation = "Reduce next session target reps by 10-20% and increase rest periods."

    return SessionReport(
        session_id=session_id,
        total_reps=total_reps,
        valid_reps=valid_reps,
        shallow_rep_count=shallow_rep_count,
        knee_warning_count=knee_warn,
        torso_warning_count=torso_warn,
        fatigue_timeline=fatigue_timeline,
        notable_events=notable_events,
        recommendation=recommendation,
    )
