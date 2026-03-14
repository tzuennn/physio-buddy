from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class PoseAngles:
    knee_angle_deg: float
    torso_lean_deg: float
    knee_inward_offset: float


class MediaPipePoseAnalyzer:
    """google-ai-edge/mediapipe-backed pose analyzer for squat metrics."""

    def __init__(self) -> None:
        try:
            import mediapipe as mp  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "mediapipe is required for pose analysis. Install with `pip install -e .[vision]` or `pip install mediapipe opencv-python`."
            ) from exc

        solutions = getattr(mp, "solutions", None)
        pose_module = getattr(solutions, "pose", None) if solutions else None
        pose_cls = getattr(pose_module, "Pose", None) if pose_module else None
        if pose_cls is None:
            raise RuntimeError(
                "Installed mediapipe package does not expose the Solutions Pose API. "
                "Reinstall with a standard wheel, for example `pip install --upgrade mediapipe`."
            )

        self._mp = mp
        self._pose = pose_cls(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def _decode_image(image_b64: str) -> np.ndarray:
        import cv2  # lazy import to avoid hard failure for non-vision flows

        raw = base64.b64decode(image_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode base64 image")
        return image

    @staticmethod
    def _angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        denom = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denom == 0:
            return 180.0
        cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    def from_image_base64(self, image_b64: str) -> PoseAngles:
        image = self._decode_image(image_b64)
        rgb = image[:, :, ::-1]
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            raise ValueError("No pose detected in frame")

        lm = result.pose_landmarks.landmark
        p = self._mp.solutions.pose.PoseLandmark

        hip = (lm[p.LEFT_HIP.value].x, lm[p.LEFT_HIP.value].y)
        knee = (lm[p.LEFT_KNEE.value].x, lm[p.LEFT_KNEE.value].y)
        ankle = (lm[p.LEFT_ANKLE.value].x, lm[p.LEFT_ANKLE.value].y)
        shoulder = (lm[p.LEFT_SHOULDER.value].x, lm[p.LEFT_SHOULDER.value].y)
        foot_index = (lm[p.LEFT_FOOT_INDEX.value].x, lm[p.LEFT_FOOT_INDEX.value].y)

        knee_angle = self._angle(hip, knee, ankle)

        torso_vec = (shoulder[0] - hip[0], shoulder[1] - hip[1])
        torso_lean = abs(math.degrees(math.atan2(torso_vec[0], -torso_vec[1])))

        knee_inward_offset = knee[0] - foot_index[0]

        return PoseAngles(
            knee_angle_deg=max(0.0, min(180.0, knee_angle)),
            torso_lean_deg=max(0.0, min(90.0, torso_lean)),
            knee_inward_offset=max(-1.0, min(1.0, knee_inward_offset)),
        )


def calculate_angles_from_landmarks(landmarks: dict[str, tuple[float, float]]) -> PoseAngles:
    required = {"hip", "knee", "ankle", "shoulder", "foot_index"}
    missing = required.difference(landmarks)
    if missing:
        raise ValueError(f"Missing landmarks: {sorted(missing)}")

    analyzer: Any = MediaPipePoseAnalyzer
    knee_angle = analyzer._angle(landmarks["hip"], landmarks["knee"], landmarks["ankle"])
    torso_vec = (
        landmarks["shoulder"][0] - landmarks["hip"][0],
        landmarks["shoulder"][1] - landmarks["hip"][1],
    )
    torso_lean = abs(math.degrees(math.atan2(torso_vec[0], -torso_vec[1])))
    knee_inward_offset = landmarks["knee"][0] - landmarks["foot_index"][0]

    return PoseAngles(
        knee_angle_deg=max(0.0, min(180.0, knee_angle)),
        torso_lean_deg=max(0.0, min(90.0, torso_lean)),
        knee_inward_offset=max(-1.0, min(1.0, knee_inward_offset)),
    )
