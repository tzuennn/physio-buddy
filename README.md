# Physio Buddy MVP Core

FastAPI backend implementing the hackathon plan modules:

- Vision form checks + rep state machine (MediaPipe)
- Real-time squat tracking with elderly-friendly UI
- Automatic audio capture for strain analysis (MERaLiON API)
- Text-to-speech coaching feedback
- Voice command recognition (Web Speech API)
- Multimodal fatigue estimation and rule-based coaching
- Session summary report

---

## Deploying for End Users (Elderly / Non-Technical)

**Elderly users should never touch a terminal, API key, or config file.**
The right model is: **you (or a caregiver) deploy once → user opens a browser link.**

### Why you need a hosted URL (not localhost)

Browsers only allow camera access on **HTTPS** pages. `http://localhost` is the one
exception (for dev only). Any real user on another device needs a proper HTTPS URL —
which a hosting platform provides automatically for free.

---

### Option A — Deploy to Render (recommended, free tier)

Render gives you a free HTTPS URL in about 5 minutes.

**Step 1 — Push your code to GitHub** (if not already)

**Step 2 — One-click deploy**

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

Or manually:

1. Go to [render.com](https://render.com) → New → Web Service
2. Connect your GitHub repo
3. Render detects `render.yaml` automatically — click **Deploy**

**Step 3 — Set secret keys in the Render dashboard** (not in files)

Go to your service → **Environment** tab → add:

| Key                | Where to get it                                                   | Required?                                 |
| ------------------ | ----------------------------------------------------------------- | ----------------------------------------- |
| `LLM_API_KEY`      | [console.groq.com](https://console.groq.com) — free, 2-min signup | Optional (falls back to rule-based coach) |
| `MERALION_API_KEY` | CR8Lab dashboard                                                  | Optional (disables audio strain analysis) |

Everything else in `render.yaml` is pre-configured (Groq URL, model, MediaPipe auto-download).

**Step 4 — Share the URL**

Render gives you `https://physio-buddy-xxxx.onrender.com`.
Send that to your users — **that's it, they are done**.

> **Free tier note:** Render free services spin down after 15 min of inactivity.
> First visit after idle takes ~30 seconds to wake up. Upgrade to a paid plan ($7/mo)
> to keep it always-on. For a hackathon or pilot, free is fine.

---

### Option B — Docker (for a VPS or cloud VM you control)

```bash
# On any Linux server with Docker installed:
git clone <your-repo>
cd physio-buddy
cp .env.example .env
# Fill in LLM_API_KEY and MERALION_API_KEY in .env
docker compose up -d --build
```

Point your domain / Nginx at port 8000. Add SSL via Let's Encrypt (Certbot).
Your HTTPS URL is then `https://your-domain.com`.

---

### What the elderly user actually does

1. Caregiver bookmarks the URL on their device (phone, tablet, or laptop)
2. User opens the browser and taps the bookmark
3. Browser asks to allow camera — user taps **Allow**
4. App opens, user taps **Start Session** — done

No installation. No terminal. No API keys. No configuration.

---

## Quick Start (For Test Users)

```bash
# Setup environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (IMPORTANT: Install vision extras for MediaPipe)
pip install -e .[vision]

# Create local environment file (one-time)
cp .env.example .env

# Run server (no keys needed for basic testing)
uvicorn physio_buddy.app:app --reload

# Visit http://127.0.0.1:8000
```

Put your keys in `.env` (recommended) or use `export` in your shell.

### Choose your test mode

1. Basic mode (no API keys):

- Works immediately.
- Squat tracking, rep counting, and rule-based coaching work.
- Audio strain analysis is disabled.
- LLM coaching falls back to adaptive rule-based mode.

2. LLM coach mode (optional):

```bash
# Groq — free tier, no credit card, fastest option
# Sign up at https://console.groq.com → create API key → paste below
export LLM_API_URL=https://api.groq.com/openai/v1
export LLM_API_KEY=gsk_...
export LLM_MODEL=llama-3.3-70b-versatile

# or local Ollama (free, runs on your machine, no API key needed)
# export LLM_API_URL=http://localhost:11434/v1
# export LLM_MODEL=llama3

# or OpenAI (paid)
# export LLM_API_URL=https://api.openai.com/v1
# export LLM_API_KEY=sk-...
# export LLM_MODEL=gpt-4o-mini
```

3. MERaLiON audio mode (optional):

```bash
export MERALION_API_KEY=your_api_key_here
```

If you do not set keys, the app still runs and degrades gracefully.

**⚠️ Important:** You MUST install the `[vision]` extras which includes:

- **MediaPipe** (pose detection)
- **OpenCV** (image processing)

If you get `500 Internal Server Error` on frame ingest, see [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)

## Features

### User Interface

- **Elderly-friendly design**: Large text, minimal buttons, clear instructions
- **Consent & Safety**: Explicit consent for camera/microphone access with safety notice
- **Live Metrics**: Big, easy-to-read display of squats, form quality, current position, fatigue level
- **Real-time Coaching**: Large interactive message box with AI coaching feedback

### Audio Integration

- **Auto Audio Capture**: Microphone automatically captures audio during session (no manual upload)
- **MERaLiON Analysis**: Audio analyzed every 10 seconds for strain/emotion detection
- **Text-to-Speech**: Coaching messages spoken aloud for better accessibility
- **Voice Commands**: Say "stop", "slower", "faster", or "help" to control the session

### Computer Vision

- **Form Detection**: MediaPipe Pose detects squat depth, knee alignment, and torso lean
- **Rep Counting**: Accurate rep counter with state machine tracking
- **Form Assessment**: Real-time warnings for form issues (shallow depth, knee inward, excessive lean)

## Configuration

Set these environment variables:

```bash
# MERaLiON Audio AI (optional, but enables audio strain detection)
MERALION_API_KEY=your_api_key_here      # Your API key from CR8Lab
MERALION_BASE_URL=https://api.cr8lab.com # API endpoint (default)
MERALION_TIMEOUT_S=20                    # Request timeout (default)

# LLM coach (optional; OpenAI-compatible endpoint)
# OpenAI example:
LLM_API_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
LLM_TIMEOUT_S=3.0

# Ollama example (no API key):
# LLM_API_URL=http://localhost:11434/v1
# LLM_MODEL=llama3

# MediaPipe Tasks model (for image_base64 pose extraction)
# Optional: if unset, the backend auto-downloads into ~/.cache/physio-buddy/models
MEDIAPIPE_POSE_TASK_MODEL_PATH=/absolute/path/to/pose_landmarker_full.task
MEDIAPIPE_POSE_TASK_AUTO_DOWNLOAD=true
MEDIAPIPE_POSE_TASK_MODEL_URL=https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
MEDIAPIPE_POSE_TASK_CACHE_DIR=~/.cache/physio-buddy/models
```

For detailed MERaLiON integration instructions, see [MERALION_INTEGRATION.md](./MERALION_INTEGRATION.md)

## API Endpoints

### Session Management

| Endpoint                 | Method | Purpose                             |
| ------------------------ | ------ | ----------------------------------- |
| `/sessions/start`        | POST   | Start a new squat session           |
| `/sessions/{id}/ingest`  | POST   | Submit frame + metrics for analysis |
| `/sessions/{id}/summary` | GET    | Get live session summary            |
| `/sessions/{id}/stop`    | POST   | End session and get final report    |

### Health & Status

| Endpoint                        | Method | Purpose                     |
| ------------------------------- | ------ | --------------------------- |
| `/health`                       | GET    | API health check            |
| `/integrations/meralion/status` | GET    | Check MERaLiON connectivity |

### Audio Processing (via MERaLiON)

| Endpoint             | Method | Purpose                        |
| -------------------- | ------ | ------------------------------ |
| `/audio/upload-file` | POST   | Upload audio file              |
| `/audio/analyze`     | POST   | Analyze audio (emotion/strain) |
| `/audio/transcribe`  | POST   | Convert audio to text          |
| `/audio/summarize`   | POST   | Generate audio summary         |
| `/audio/translate`   | POST   | Translate audio to language    |
| `/audio/process`     | POST   | Custom LLM processing          |

## Usage Flow

1. **User opens app** → Page asks for consent
2. **Clicks "Start Camera"** → Camera streams video
3. **Clicks "Start Session"** → Session begins, audio capture starts
4. **System tracks squats** via vision:
   - Detects squat depth
   - Checks form issues
   - Counts reps
5. **Every 10 seconds**: Audio is analyzed via MERaLiON for strain detection
6. **Coaching messages** are:
   - Displayed in large text box
   - Spoken aloud via text-to-speech
   - Based on form quality and fatigue level
7. **Voice commands** available:
   - "stop" / "pause" → End session
   - "slower" / "faster" → Tempo feedback
   - "help" → Form tips
8. **Clicks "Stop Session"** → Summary shown with total reps and stats

## Session Response Format

```json
{
  "ts": 1234.567,
  "frame": {
    "knee_angle_deg": 95.0,
    "torso_lean_deg": 15.0,
    "knee_inward_offset": -0.05
  },
  "phase": "BOTTOM",
  "rep_count": 5,
  "form": {
    "depth_quality": "good",
    "knee_tracking_warning": false,
    "torso_lean_warning": false
  },
  "fatigue_level": "MEDIUM",
  "coaching_message": "Good form. Control tempo and take a short breath reset."
}
```

## Testing

```bash
# Run tests
python -m pytest -q

# Coverage
pytest --cov=physio_buddy tests/
```

### Smoke test for real users (2 minutes)

1. Start server: `uvicorn physio_buddy.app:app --reload`
2. Open browser at `http://127.0.0.1:8000`
3. Accept consent, start camera, start session, do 3 squats, stop session
4. Confirm:

- rep count increments
- form/fatigue/coaching fields update
- summary appears at stop

### API sanity checks

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/sessions/start -H "Content-Type: application/json" -d '{"profile":"general_mobility"}'
```

## Safety Notice

This is an assistive wellness tool, not medical advice. Users should:

- Stop immediately if experiencing pain
- Consult medical professionals for injuries
- Use only as supplementary coaching

## Documentation

- **[TROUBLESHOOTING.md](./TROUBLESHOOTING.md)** - Fix common issues like `500 Internal Server Error`, missing dependencies, or camera setup problems
- **[MERALION_INTEGRATION.md](./MERALION_INTEGRATION.md)** - Complete guide to setting up and integrating MERaLiON audio AI
- **[DEBUGGING.md](./DEBUGGING.md)** - Advanced debugging guide for developers
