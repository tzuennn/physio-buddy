# Physio Buddy MVP Core

FastAPI backend implementing the hackathon plan modules:
- Vision form checks + rep state machine
- MediaPipe pose processing from landmarks or base64 image frames
- MERaLiON audio pipeline integration (`/upload-url`, `/upload-status`, `/transcribe`, `/analyze`, `/process`)
- Multimodal fatigue estimation and rule-based coaching
- Session summary report

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
# For image-based MediaPipe inference:
pip install -e .[vision]
uvicorn physio_buddy.api:app --reload
```

## Configuration
Set environment variables:
- `MERALION_API_KEY` (required for MERaLiON endpoints)
- `MERALION_BASE_URL` (default: `https://api.cr8lab.com`)
- `MERALION_TIMEOUT_S` (default: `20`)

## Endpoints
### Session
- `POST /sessions/start`
- `POST /sessions/{session_id}/ingest`
  - Supports one of:
    - `frame` (precomputed metrics)
    - `landmarks` (hip/knee/ankle/shoulder/foot_index)
    - `image_base64` (decoded + processed via MediaPipe Pose)
- `GET /sessions/{session_id}/summary`
- `POST /sessions/{session_id}/stop`

### MERaLiON pass-through
- `GET /integrations/meralion/status`
- `POST /audio/upload-url`
- `POST /audio/upload-status`
- `POST /audio/transcribe`
- `POST /audio/analyze`
- `POST /audio/process`

## Safety
The system is assistive and non-diagnostic.


## Troubleshooting
### MediaPipe Pose runtime errors
If `/sessions/{id}/ingest` returns a MediaPipe error like:

- `Installed mediapipe package does not expose the Solutions Pose API`

then your Python environment likely has an incompatible package variant. This backend uses the **Python Solutions Pose API** (`mediapipe.solutions.pose`), not the JS Tasks packages.

Fix with:

```bash
pip uninstall -y mediapipe mediapipe-silicon
pip install --upgrade mediapipe opencv-python
# or install this project with vision extras
pip install -e .[vision]
```

For browser-only projects, MediaPipe Tasks JS packages such as `@mediapipe/tasks-vision` are valid, but they do not replace Python `mediapipe` for this FastAPI backend.
