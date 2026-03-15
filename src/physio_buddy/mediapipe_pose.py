from __future__ import annotations

import base64
import math
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from .config import settings


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

        self._mp = mp
        self._backend: str | None = None
        self._pose_module: Any | None = None
        self._pose: Any | None = None
        self._pose_landmarker: Any | None = None
        self._pose_landmark_enum: Any | None = None

        self._init_tasks_backend()

    def _init_tasks_backend(self) -> None:
        try:
            from mediapipe.tasks.python.core import base_options as base_options_module  # type: ignore
            from mediapipe.tasks.python.vision import pose_landmarker as pose_landmarker_module  # type: ignore
            from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module  # type: ignore
        except Exception:
            self._init_solutions_backend()
            return

        model_path = self._resolve_task_model_path()

        options = pose_landmarker_module.PoseLandmarkerOptions(
            base_options=base_options_module.BaseOptions(model_asset_path=model_path),
            running_mode=running_mode_module.VisionTaskRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )

        try:
            self._pose_landmarker = pose_landmarker_module.PoseLandmarker.create_from_options(options)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to initialize MediaPipe Tasks PoseLandmarker with model at '{model_path}'. "
                "Set MEDIAPIPE_POSE_TASK_MODEL_PATH to a valid .task file."
            ) from exc

        self._pose_landmark_enum = pose_landmarker_module.PoseLandmark
        self._backend = "tasks"

    def _init_solutions_backend(self) -> None:
        pose_module: Any | None = None
        try:
            solutions = getattr(self._mp, "solutions", None)
            if solutions is not None and hasattr(solutions, "pose"):
                pose_module = solutions.pose
        except Exception:
            pose_module = None

        if pose_module is None:
            try:
                from mediapipe.python.solutions import pose as pose_module  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "Installed mediapipe package does not expose Solutions Pose or Tasks PoseLandmarker. "
                    "Install a compatible mediapipe wheel and provide MEDIAPIPE_POSE_TASK_MODEL_PATH for Tasks."
                ) from exc

        assert pose_module is not None  # guaranteed – RuntimeError raised above if import failed
        self._pose_module = pose_module
        self._pose = pose_module.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._backend = "solutions"

    @staticmethod
    def _download_to_file(url: str, target: Path) -> None:
        with urllib.request.urlopen(url, timeout=30) as response:
            target.write_bytes(response.read())

    @staticmethod
    def _resolve_cache_dir(configured_cache_dir: str) -> Path:
        """Return a writable cache dir, falling back if configured path is read-only."""
        candidates = [
            Path(os.path.expanduser(configured_cache_dir)),
            Path.home() / ".cache" / "physio-buddy" / "models",
            Path(tempfile.gettempdir()) / "physio-buddy" / "models",
        ]

        for candidate in candidates:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                test_file = candidate / ".write_test"
                test_file.write_text("ok", encoding="utf-8")
                test_file.unlink(missing_ok=True)
                return candidate
            except OSError:
                continue

        raise RuntimeError(
            "No writable MediaPipe model cache directory found. "
            "Set MEDIAPIPE_POSE_TASK_CACHE_DIR to a writable path."
        )

    def _resolve_task_model_path(self) -> str:
        configured_path = settings.mediapipe_pose_task_model_path
        if configured_path:
            model_path = Path(configured_path).expanduser()
            if model_path.exists():
                return str(model_path)
            raise RuntimeError(
                f"MEDIAPIPE_POSE_TASK_MODEL_PATH points to a missing file: {model_path}"
            )

        cache_dir = self._resolve_cache_dir(settings.mediapipe_pose_task_cache_dir)
        auto_model_path = cache_dir / "pose_landmarker_full.task"

        if auto_model_path.exists():
            return str(auto_model_path)

        if not settings.mediapipe_pose_task_auto_download:
            raise RuntimeError(
                "No PoseLandmarker model found. Set MEDIAPIPE_POSE_TASK_MODEL_PATH to a local .task model file."
            )

        try:
            self._download_to_file(settings.mediapipe_pose_task_model_url, auto_model_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to auto-download PoseLandmarker model. "
                "Set MEDIAPIPE_POSE_TASK_MODEL_PATH to a local .task model file."
            ) from exc

        return str(auto_model_path)

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
        """Extract pose angles from base64-encoded image using MediaPipe Pose."""
        import cv2  # lazy import

        image = self._decode_image(image_b64)

        # MediaPipe expects RGB format
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self._backend == "tasks":
            if self._pose_landmarker is None or self._pose_landmark_enum is None:
                raise RuntimeError("Tasks backend not initialized")
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            result = self._pose_landmarker.detect(mp_image)
            if not result.pose_landmarks:
                raise ValueError("No pose detected in frame")
            lm = result.pose_landmarks[0]
            p = self._pose_landmark_enum
        else:
            if self._pose is None or self._pose_module is None:
                raise RuntimeError("Solutions backend not initialized")
            result = self._pose.process(rgb)
            if not result.pose_landmarks:
                raise ValueError("No pose detected in frame")
            lm = result.pose_landmarks.landmark
            p = self._pose_module.PoseLandmark

        # Extract key landmarks with visibility check
        def get_landmark(landmark_enum: Any) -> tuple[float, float] | None:
            lm_obj = lm[landmark_enum.value]
            visibility = getattr(lm_obj, "visibility", 1.0)
            if visibility is not None and visibility < 0.3:
                return None
            return (lm_obj.x, lm_obj.y)

        hip = get_landmark(p.LEFT_HIP)
        knee = get_landmark(p.LEFT_KNEE)
        ankle = get_landmark(p.LEFT_ANKLE)
        shoulder = get_landmark(p.LEFT_SHOULDER)
        foot_index = get_landmark(p.LEFT_FOOT_INDEX)

        if any(v is None for v in [hip, knee, ankle, shoulder, foot_index]):
            raise ValueError("Key landmarks not visible or not detected")

        hip = cast(tuple[float, float], hip)
        knee = cast(tuple[float, float], knee)
        ankle = cast(tuple[float, float], ankle)
        shoulder = cast(tuple[float, float], shoulder)
        foot_index = cast(tuple[float, float], foot_index)

        # Calculate knee angle (angle at knee joint: hip-knee-ankle)
        knee_angle = self._angle(hip, knee, ankle)

        # Calculate torso lean (angle from vertical)
        torso_vec = (shoulder[0] - hip[0], shoulder[1] - hip[1])
        torso_lean = abs(math.degrees(math.atan2(torso_vec[0], -torso_vec[1])))

        # Calculate knee inward offset (knee x - foot x)
        knee_inward_offset = knee[0] - foot_index[0]

        return PoseAngles(
            knee_angle_deg=max(0.0, min(180.0, knee_angle)),
            torso_lean_deg=max(0.0, min(90.0, torso_lean)),
            knee_inward_offset=max(-1.0, min(1.0, knee_inward_offset)),
        )

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
            self._pose = None
        if self._pose_landmarker is not None:
            self._pose_landmarker.close()
            self._pose_landmarker = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


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
