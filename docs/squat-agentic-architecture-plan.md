# Physio Buddy — 1-Day Hackathon Plan (Squat Monitoring, Multimodal)

## 0) Reframe: not "agents", but **buildable modules**
For this hackathon, call these **modules/services** unless they are truly autonomous decision-makers.

A practical 1-day build should have 4 modules:
1. **Vision Module** (pose + squat metrics)
2. **Audio Module** (speech command + vocal fatigue cues)
3. **Coaching Logic Module** (rules + optional LLM wording)
4. **Session Summary Module** (clinician-facing output)

This avoids overengineering while still showing multimodal intelligence.

---

## 1) Problem statement fit (Track 2)

### Target condition/metric monitored
We monitor **at-home squat quality and exertion risk** using:
- **Movement quality metrics**: knee angle depth, knee tracking, torso lean, rep consistency.
- **Fatigue/strain indicators**: slowdown in rep tempo + reduced depth + audio cues (breathing/vocal strain).

This directly matches the track requirements for remote posture/activity monitoring with multimodal inputs.

### Why multimodal AI is necessary
Video-only can detect posture but misses user state (e.g., strain, frustration, pain cues).
Audio-only can detect stress/vocal effort but misses biomechanical correctness.

**Combining video + audio** improves safety and coaching quality:
- Video detects *how* movement is performed.
- Audio detects *how hard* it feels for the user.
- Together enables adaptation (pause/reduce reps) instead of generic feedback.

---

## 2) Scope for a true 1-day MVP

### In scope (must-have)
- Webcam squat tracking (MediaPipe Pose).
- Rep counting + 3 form checks:
  - squat depth,
  - knee alignment,
  - torso lean.
- Voice commands: start/pause/resume/stop/reps-left.
- Fatigue level (LOW/MEDIUM/HIGH) from simple multimodal heuristics.
- End-session summary report.

### Out of scope (for later)
- Clinical diagnosis.
- Personalized rehab protocol generation.
- Multi-exercise support.
- Wearable integration.

---

## 3) System architecture (simple and feasible)

```text
Webcam ──> Vision Module ──┐
                           ├──> Coaching Logic ──> UI + TTS feedback
Mic ─────> Audio Module ───┘

All events ────────────────> Session Store ──> Summary Module (clinician report)
```

### Module A — Vision Module
**Input:** webcam frames  
**Tool:** MediaPipe Pose  
**Output per frame/window:**
- landmarks,
- knee angle,
- torso angle,
- knee alignment flag,
- rep phase + rep count.

### Module B — Audio Module (MERaLiON-backed)
**Input:** 5–10s mic chunks  
**Endpoints:** `/upload-url`, `/upload-status`, `/transcribe`, `/analyze`  
**Output:**
- command intent (e.g., pause/resume),
- vocal strain/emotion signal,
- confidence.

### Module C — Coaching Logic Module
Rule-based first (fast + reliable), LLM only for wording polish.

**Examples:**
- If knee angle bottom >120° => "Go deeper slowly."
- If knees cave inward => "Push knees slightly outward."
- If fatigue HIGH => "Pause for 30 seconds; reducing remaining reps."

### Module D — Session Summary Module
Creates a concise report:
- total reps / valid reps,
- error distribution,
- fatigue timeline,
- notable events,
- recommendations for next session.

---

## 4) Exact metrics and thresholds (deliverable-ready)

Use fixed initial thresholds; tune during demo testing.

### Form metrics
1. **Depth** (knee angle at bottom)
   - Good: 85°–110°
   - Shallow: >120°
2. **Knee tracking** (knee vs toe/ankle horizontal alignment)
   - Warning if persistent inward deviation over N frames.
3. **Torso lean** (torso from vertical)
   - Warning if >35° for sustained period.

### Rep counting state machine
`STAND -> DESCEND -> BOTTOM -> ASCEND -> STAND`
- Count +1 only on valid full cycle.

### Fatigue heuristics (multimodal)
Compute every 20–30 seconds:
- visual tempo drop >20%,
- depth reduction trend >10°,
- longer rest gaps,
- audio strain/emotion increase.

Map to:
- LOW: no strong signs,
- MEDIUM: 2 signs,
- HIGH: 3+ signs.

---

## 5) NMLP model usage (explicit)

### MERaLiON (primary in MVP)
Use for:
- command transcription,
- paralinguistic analysis (emotion/strain),
- multilingual/code-switch speech robustness.

### SEA-LION v4 (optional if time permits)
Use only for:
- generating clinician-friendly summary language,
- translating summary/instructions for multilingual users,
- optional document-to-exercise note interpretation.

If constrained by time, keep SEA-LION as a **post-processing step**, not in the real-time loop.

---

## 6) Data handling, privacy, and safety plan

### Consent
- Explicit pre-session consent screen:
  - camera + mic usage,
  - purpose of monitoring,
  - retention duration.

### Minimization
- Process live streams in-memory where possible.
- Store only derived metrics/events by default.
- Audio/video raw storage is optional and off by default.

### Anonymization
- Session IDs without personal identifiers.
- Redact names from transcripts before export.

### Security
- HTTPS for APIs.
- Access-controlled summary endpoint.
- Time-limited presigned URLs for uploads.

### Safety/ethics language
- "Assistive wellness/rehab support, not a diagnostic system."
- "Stop exercise if pain occurs; consult a clinician."
- Avoid overconfident medical claims in UI/report.

---

## 7) User flow (what demo judges will see)

1. User says: **"Start workout"**.
2. System tracks squats, overlays skeleton + live cues.
3. Real-time feedback appears for form correction.
4. If strain increases, system suggests short rest / reduced reps.
5. User says: **"Stop"**.
6. Clinician-style summary appears with metrics + recommendations.

---

## 8) One-day feasibility plan (hour-by-hour)

### Hours 0–2: Foundation
- Setup app shell + webcam stream + overlay canvas.
- Integrate MediaPipe Pose.

### Hours 2–5: Core squat analytics
- Joint angle calculations.
- Rep state machine.
- 3 form checks + UI indicators.

### Hours 5–7: Audio integration
- MERaLiON upload/transcribe/analyze pipeline.
- Voice command intent mapping.

### Hours 7–9: Multimodal fatigue + adaptation
- Implement LOW/MEDIUM/HIGH fatigue logic.
- Trigger adaptive coaching (pause/reduce reps).

### Hours 9–11: Summary + polish
- Session JSON log + summary panel/export.
- Safety text + consent screen.

### Hour 11–12: Demo hardening
- Record fallback demo video.
- Dry-run script + latency checks.

---

## 9) Recommended tech stack (for 1 day)

### Option A (fastest web demo)
- **Frontend:** Next.js (React + TypeScript)
- **Vision:** MediaPipe Pose JS
- **Backend:** FastAPI (Python) or Node/Express
- **Audio AI:** MERaLiON API
- **Storage:** SQLite or JSONL session logs
- **TTS:** browser SpeechSynthesis API

### Option B (Python-heavy)
- **UI:** Streamlit or Gradio
- **Vision:** MediaPipe Python + OpenCV
- **Audio:** MERaLiON API via Python requests
- **Pros:** very fast prototyping, less UI polish

**Recommendation for judges:** Option A if you have front-end skills; Option B if team is mostly ML/Python.

---

## 10) Concrete work breakdown (who does what)

### If 3 people
- **Dev 1 (Vision):** landmarks, angles, rep logic, form flags.
- **Dev 2 (Audio/AI):** MERaLiON integration, command parser, fatigue fusion.
- **Dev 3 (App):** UI panels, orchestrator, session storage, summary view.

### If 2 people
- **Dev A:** vision + form + fatigue heuristics.
- **Dev B:** UI + MERaLiON + summary/reporting.

---

## 11) Demo deliverables checklist (track-aligned)

- [ ] **Clear metric definition:** squat quality + exertion/fatigue indicators.
- [ ] **Why multimodal:** explicit video/audio complementarity.
- [ ] **Functional demo:** live prototype with real-time feedback.
- [ ] **Data/privacy plan:** consent, minimization, anonymization.
- [ ] **Ethics/safety:** non-diagnostic statement and safe-use guidance.
- [ ] **Feasibility:** architecture and 1-day implementation evidence.

---

## 12) MVP definition of done

You are done when these all work in a live run:
1. Rep counting for squats with <10% counting error on a short test.
2. At least 3 form checks trigger correctly.
3. Voice start/pause/resume/stop commands function.
4. Fatigue level updates from multimodal cues at least every 30s.
5. End-session report generated in <5s after stopping.

