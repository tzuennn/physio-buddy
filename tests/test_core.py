from fastapi.testclient import TestClient

from physio_buddy.api import app
from physio_buddy.coaching import choose_coaching
from physio_buddy.fatigue import FatigueEstimator
from physio_buddy.mediapipe_pose import calculate_angles_from_landmarks
from physio_buddy.models import FatigueLevel
from physio_buddy.summary import build_summary
from physio_buddy.vision import RepStateMachine, assess_form


def test_rep_state_machine_counts_full_cycle() -> None:
    sm = RepStateMachine()
    for angle in [170, 145, 108, 120, 140, 155, 165]:
        sm.update(angle)
    assert sm.rep_count == 1


def test_form_assessment_flags() -> None:
    form = assess_form(knee_angle_deg=130, knee_inward_offset=-0.2, torso_lean_deg=40)
    assert form.depth_quality == "shallow"
    assert form.knee_tracking_warning
    assert form.torso_lean_warning


def test_fatigue_estimator_high() -> None:
    est = FatigueEstimator(window_size=6)
    seq = [
        (1.0, 100, 1.0, 0.2),
        (1.0, 100, 1.0, 0.3),
        (0.75, 110, 1.0, 0.4),
        (0.65, 115, 1.4, 0.7),
    ]
    level = FatigueLevel.LOW
    for row in seq:
        level = est.update(*row)
    assert level == FatigueLevel.HIGH


def test_summary_recommendation() -> None:
    sm = RepStateMachine()
    events = []
    for i, angle in enumerate([170, 145, 108, 140, 160]):
        phase = sm.update(angle)
        form = assess_form(knee_angle_deg=130, knee_inward_offset=-0.12, torso_lean_deg=10)
        fatigue = FatigueLevel.MEDIUM
        msg = choose_coaching(form, fatigue).message
        from physio_buddy.models import Event, FrameMetrics

        events.append(
            Event(
                ts=i,
                frame=FrameMetrics(knee_angle_deg=angle, torso_lean_deg=10, knee_inward_offset=-0.12),
                phase=phase,
                rep_count=sm.rep_count,
                form=form,
                fatigue_level=fatigue,
                coaching_message=msg,
            )
        )

    report = build_summary("s1", events)
    assert report.session_id == "s1"
    assert report.knee_warning_count > 0


def test_landmark_angle_conversion() -> None:
    angles = calculate_angles_from_landmarks(
        {
            "hip": (0.5, 0.4),
            "knee": (0.5, 0.6),
            "ankle": (0.55, 0.8),
            "shoulder": (0.48, 0.2),
            "foot_index": (0.6, 0.85),
        }
    )
    assert 0 <= angles.knee_angle_deg <= 180
    assert 0 <= angles.torso_lean_deg <= 90
    assert -1 <= angles.knee_inward_offset <= 1


def test_ingest_with_landmarks() -> None:
    client = TestClient(app)
    start = client.post("/sessions/start")
    session_id = start.json()["session_id"]

    payload = {
        "landmarks": {
            "hip": [0.5, 0.4],
            "knee": [0.5, 0.6],
            "ankle": [0.55, 0.8],
            "shoulder": [0.48, 0.2],
            "foot_index": [0.6, 0.85],
        },
        "audio": {"command_intent": "none", "strain_score": 0.2, "confidence": 0.9},
        "tempo_rps": 1.0,
        "rest_gap_s": 1.0,
    }

    response = client.post(f"/sessions/{session_id}/ingest", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["frame"]["knee_angle_deg"] >= 0


def test_meralion_requires_key() -> None:
    client = TestClient(app)
    response = client.post("/audio/upload-url", json={"filename": "clip.wav", "content_type": "audio/wav"})
    assert response.status_code == 503


def test_meralion_upload_passthrough(monkeypatch):
    from physio_buddy import api as api_module

    class FakeMeralion:
        enabled = True

        def upload_url(self, filename: str, content_type: str):
            return {"file_id": "f1", "url": "https://s3.example/upload", "filename": filename, "content_type": content_type}

    monkeypatch.setattr(api_module, "_meralion", FakeMeralion())
    client = TestClient(app)
    response = client.post("/audio/upload-url", json={"filename": "clip.wav", "content_type": "audio/wav"})
    assert response.status_code == 200
    assert response.json()["file_id"] == "f1"
