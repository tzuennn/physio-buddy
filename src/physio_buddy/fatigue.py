from __future__ import annotations

from collections import deque

from .models import FatigueLevel


class FatigueEstimator:
    """Rule-based multimodal fatigue estimation over a rolling window."""

    def __init__(self, window_size: int = 6) -> None:
        self._tempo_history: deque[float] = deque(maxlen=window_size)
        self._depth_history: deque[float] = deque(maxlen=window_size)
        self._rest_gap_history: deque[float] = deque(maxlen=window_size)
        self._strain_history: deque[float] = deque(maxlen=window_size)

    def update(self, tempo_rps: float, depth_angle: float, rest_gap_s: float, strain_score: float) -> FatigueLevel:
        level, _ = self.update_with_reason(
            tempo_rps=tempo_rps,
            depth_angle=depth_angle,
            rest_gap_s=rest_gap_s,
            strain_score=strain_score,
        )
        return level

    def update_with_reason(
        self, tempo_rps: float, depth_angle: float, rest_gap_s: float, strain_score: float
    ) -> tuple[FatigueLevel, str]:
        self._tempo_history.append(tempo_rps)
        self._depth_history.append(depth_angle)
        self._rest_gap_history.append(rest_gap_s)
        self._strain_history.append(strain_score)

        if len(self._tempo_history) < 3:
            return FatigueLevel.LOW, "warming up; not enough data yet"

        signs = 0
        reasons: list[str] = []
        tempo_drop = (max(self._tempo_history) - self._tempo_history[-1]) / max(self._tempo_history)
        if tempo_drop > 0.20:
            signs += 1
            reasons.append("tempo dropped")

        depth_loss = self._depth_history[-1] - min(self._depth_history)
        if depth_loss > 10:
            signs += 1
            reasons.append("range became shallower")

        if self._rest_gap_history[-1] > (sum(self._rest_gap_history) / len(self._rest_gap_history)) * 1.3:
            signs += 1
            reasons.append("rest gap increased")

        if self._strain_history[-1] > 0.65:
            signs += 1
            reasons.append("audio strain is high")

        if signs >= 3:
            return FatigueLevel.HIGH, ", ".join(reasons) or "multiple fatigue signs"
        if signs >= 2:
            return FatigueLevel.MEDIUM, ", ".join(reasons) or "some fatigue signs"
        return FatigueLevel.LOW, "steady tempo and movement"
