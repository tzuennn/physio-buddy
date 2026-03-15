// =============== DOM Elements ===============
const apiStatus = document.getElementById("apiStatus");
const consent = document.getElementById("consent");
const startCamera = document.getElementById("startCamera");
const stopCamera = document.getElementById("stopCamera");
const cameraStatus = document.getElementById("cameraStatus");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const startSession = document.getElementById("startSession");
const stopSession = document.getElementById("stopSession");
const phaseLabel = document.getElementById("phase");
const repCountLabel = document.getElementById("repCount");
const overlayRepCount = document.getElementById("overlayRepCount");
const formLabel = document.getElementById("form");
const fatigueLabel = document.getElementById("fatigue");
const overlayFatigue = document.getElementById("overlayFatigue");
const coachMessage = document.getElementById("coachMessage");
const fetchSummary = document.getElementById("fetchSummary");
const summaryOutput = document.getElementById("summaryOutput");
const voiceStatus = document.getElementById("voiceStatus");
const profileGrid = document.getElementById("profileGrid");
const profileStatus = document.getElementById("profileStatus");

// Enable with `?debug=1` or `localStorage.setItem("physioDebug", "1")`.
const DEBUG_MODE =
  new URLSearchParams(window.location.search).get("debug") === "1" ||
  localStorage.getItem("physioDebug") === "1";
const debugEvents = [];

function debugLog(event, details = {}) {
  if (!DEBUG_MODE) return;
  const entry = {
    ts: new Date().toISOString(),
    event,
    details,
  };
  debugEvents.push(entry);
  if (debugEvents.length > 200) debugEvents.shift();
  window.physioDebugEvents = debugEvents;
  console.debug("[PhysioBuddyDebug]", entry);
}

async function parseJsonSafe(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text };
  }
}

// =============== State ===============
let stream = null;
let audioStream = null;
let mediaRecorder = null;
let audioChunks = [];
let sessionId = null;
let tracking = false;
let lastFrameSent = 0;
let isSendingFrame = false;
const sendIntervalMs = 220;
let ingest422Count = 0;
let frameBackoffUntil = 0;
const speakMinIntervalMs = 3500;
const coachMessageHoldMs = 3800;
let lastSpokenMessage = "";
let lastSpokenAt = 0;
let lastCoachMessage = "";
let lastCoachMessageAt = 0;

function updateCoachMessageStable(message, { force = false } = {}) {
  const next = (message || "").trim();
  if (!next) return;

  const now = Date.now();
  const holdActive = now - lastCoachMessageAt < coachMessageHoldMs;
  const isDifferent = next.toLowerCase() !== lastCoachMessage.toLowerCase();
  const speakingNow = "speechSynthesis" in window && speechSynthesis.speaking;

  if (!force && isDifferent && (holdActive || speakingNow)) {
    return;
  }

  coachMessage.textContent = next;
  lastCoachMessage = next;
  lastCoachMessageAt = now;
}

// Speech Recognition API
const SpeechRecognition =
  window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = new SpeechRecognition();
recognition.continuous = true;
recognition.interimResults = true;
recognition.lang = "en-US";

// =============== API Health Check ===============
async function checkApi() {
  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error("API offline");
    apiStatus.textContent = "API: online";
    debugLog("health-ok", { status: res.status });
  } catch (err) {
    apiStatus.textContent = "API: offline";
    debugLog("health-fail", { error: String(err) });
  }
}

// =============== Camera Management ===============
async function startCameraStream() {
  if (!consent.checked) {
    cameraStatus.textContent = "Consent required";
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    video.srcObject = stream;
    cameraStatus.textContent = "✓ Camera active";
    stopCamera.disabled = false;
    startCamera.disabled = true;
    startSession.disabled = false;
  } catch (err) {
    cameraStatus.textContent = "Camera denied";
  }
}

function stopCameraStream() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }
  stream = null;
  video.srcObject = null;
  cameraStatus.textContent = "Camera: off";
  stopCamera.disabled = true;
  startCamera.disabled = false;
  startSession.disabled = true;
}

// =============== Audio Capture Setup ===============
let audioAnalysisInterval = null;
let currentStrainScore = 0;
let audioFailureCount = 0;
let audioDisabledUntil = 0;
let recorderMimeType = "";
let lastAudioAnalysisAt = 0;
let audioSignalValid = false;

function chooseRecorderMimeType() {
  if (typeof MediaRecorder === "undefined") return "";
  const candidates = [
    // MERaLiON supported: .m4a/.mp3/.wav/.flac
    "audio/mp4",
    "audio/mpeg",
    "audio/wav",
    "audio/flac",
  ];
  for (const mime of candidates) {
    if (MediaRecorder.isTypeSupported?.(mime)) return mime;
  }
  return "";
}

function clampStrainScore(value, fallback = 0) {
  const num = Number(value);
  if (!Number.isFinite(num)) return fallback;
  // Some providers return 0-100 instead of 0-1.
  const normalized = num > 1 && num <= 100 ? num / 100 : num;
  return Math.max(0, Math.min(1, normalized));
}

async function startAudioCapture() {
  try {
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorderMimeType = chooseRecorderMimeType();
    if (!recorderMimeType) {
      voiceStatus.textContent = "🎤 Audio analysis unavailable on this browser";
      console.warn(
        "No MERaLiON-supported recording format available (need mp4/mp3/wav/flac).",
      );
      audioStream.getTracks().forEach((track) => track.stop());
      audioStream = null;
      return;
    }
    mediaRecorder = new MediaRecorder(audioStream, {
      mimeType: recorderMimeType,
    });
    audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
      audioChunks.push(event.data);
    };

    mediaRecorder.start(1000); // Capture in 1-second chunks
    voiceStatus.textContent = "🎤 Listening...";
    debugLog("audio-recorder-started", { recorderMimeType });

    // Start audio analysis every 10 seconds
    // If MERALION_API_KEY is not set, analysis will fail gracefully with 503 error
    audioAnalysisInterval = setInterval(() => {
      analyzeAudioWithMeralion();
    }, 10000);
    console.log("Audio capture started with MERaLiON analysis enabled");
  } catch (err) {
    console.error("Audio capture error:", err);
    debugLog("audio-capture-error", { error: String(err) });
    voiceStatus.textContent = "🎤 Mic error";
  }
}

function stopAudioCapture() {
  if (audioAnalysisInterval) {
    clearInterval(audioAnalysisInterval);
  }
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  if (audioStream) {
    audioStream.getTracks().forEach((track) => track.stop());
  }
  voiceStatus.textContent = "🎤 Voice: off";
}

// =============== Audio Analysis with MERaLiON ===============
async function analyzeAudioWithMeralion() {
  if (Date.now() < audioDisabledUntil) return;
  if (audioChunks.length === 0) return;

  try {
    const rawMimeType = mediaRecorder?.mimeType || recorderMimeType || "";
    if (!rawMimeType) return;
    const mimeType = rawMimeType.includes("mp4") ? "audio/m4a" : rawMimeType;
    const ext = mimeType.includes("wav")
      ? "wav"
      : mimeType.includes("mpeg")
        ? "mp3"
        : mimeType.includes("m4a") || mimeType.includes("mp4")
          ? "m4a"
          : mimeType.includes("flac")
            ? "flac"
            : "bin";

    // Create audio blob from chunks
    const audioBlob = new Blob(audioChunks, { type: mimeType });

    // Upload to MERaLiON
    const form = new FormData();
    form.append("file", audioBlob, `session_audio.${ext}`);

    const uploadRes = await fetch("/audio/upload-file", {
      method: "POST",
      body: form,
    });

    if (!uploadRes.ok) {
      const errorData = await parseJsonSafe(uploadRes);
      const errorMsg =
        errorData.detail || `Upload failed (${uploadRes.status})`;
      debugLog("audio-upload-fail", {
        status: uploadRes.status,
        detail: errorMsg,
        mimeType,
        chunkCount: audioChunks.length,
        recorderMimeType,
      });

      // Gracefully handle missing API key
      if (uploadRes.status === 503) {
        console.warn(
          "MERaLiON API key not configured. Audio analysis disabled.",
        );
        voiceStatus.textContent = "🎤 (API key not set)";
      } else {
        console.error(`Audio upload error: ${errorMsg}`);
        if (uploadRes.status === 502) {
          voiceStatus.textContent = "🎤 Audio format not accepted";
        }
        audioFailureCount += 1;
        if (audioFailureCount >= 3) {
          // Back off for 2 minutes when upstream repeatedly fails.
          audioDisabledUntil = Date.now() + 120000;
          voiceStatus.textContent = "🎤 Audio analysis unavailable";
        }
      }
      return;
    }

    const uploadData = await uploadRes.json();
    const fileKey = uploadData.fileKey;

    // Analyze the uploaded audio
    const analyzeRes = await fetch("/audio/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_key: fileKey }),
    });

    if (!analyzeRes.ok) {
      const errorData = await parseJsonSafe(analyzeRes);
      const errorMsg =
        errorData.detail || `Analysis failed (${analyzeRes.status})`;
      debugLog("audio-analyze-fail", {
        status: analyzeRes.status,
        detail: errorMsg,
        fileKey,
      });

      // Only log errors other than 503 (which is API key not set)
      if (analyzeRes.status !== 503) {
        console.error(`Audio analysis error: ${errorMsg}`);
        audioFailureCount += 1;
        if (audioFailureCount >= 3) {
          audioDisabledUntil = Date.now() + 120000;
          voiceStatus.textContent = "🎤 Audio analysis unavailable";
        }
      }
      return;
    }

    const analysisData = await analyzeRes.json();
    debugLog("audio-analyze-ok", {
      keys: Object.keys(analysisData || {}),
      fileKey,
    });
    audioFailureCount = 0;
    audioSignalValid = true;
    lastAudioAnalysisAt = Date.now();

    // Extract strain score from analysis
    // MERaLiON returns emotion/strain data - extract relevant metrics
    if (analysisData.emotion_intensity != null) {
      currentStrainScore = clampStrainScore(analysisData.emotion_intensity, 0);
    } else if (analysisData.strain_level != null) {
      currentStrainScore = clampStrainScore(analysisData.strain_level, 0);
    } else {
      // Fallback: estimate based on audio characteristics
      currentStrainScore = clampStrainScore(audioChunks.length * 0.05, 0);
    }

    voiceStatus.textContent = `🎤 Strain: ${(currentStrainScore * 100).toFixed(0)}%`;

    // Clear chunks for next analysis window
    audioChunks = [];
  } catch (err) {
    console.error("Audio analysis error:", err);
    debugLog("audio-analysis-error", { error: String(err) });
    audioSignalValid = false;
  }
}

// =============== Session Management ===============
async function createSession() {
  if (!consent.checked) {
    coachMessage.textContent = "Consent required to start.";
    return;
  }

  const selectedProfile =
    document.querySelector('input[name="profile"]:checked')?.value ||
    "general_mobility";

  try {
    const res = await fetch("/sessions/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile: selectedProfile }),
    });
    if (!res.ok) throw new Error("Failed to start session");
    const data = await res.json();
    sessionId = data.session_id;
    stopSession.disabled = false;
    fetchSummary.disabled = false;
    tracking = true;

    // Lock profile selector during active session
    if (profileGrid) profileGrid.style.pointerEvents = "none";
    if (profileStatus) {
      const labels = {
        post_op_conservative: "🏥 Post-Op Recovery",
        knee_rehab: "🦵 Knee Rehab",
        general_mobility: "🧘 General Mobility",
        performance: "🏋️ Performance",
      };
      profileStatus.textContent = `Active profile: ${labels[selectedProfile] || selectedProfile}`;
    }

    const isLLM = !!data.llm_active; // informational only
    updateCoachMessageStable("Session started! Begin your squats.", {
      force: true,
    });
    speakMessage("Session started. Begin your squats.");

    // Start audio capture for strain analysis
    await startAudioCapture();

    // Start voice command recognition
    setupVoiceCommands();
  } catch (err) {
    coachMessage.textContent = "Unable to start session.";
  }
}

async function endSession() {
  tracking = false;
  stopAudioCapture();

  if (recognition.continuous) {
    try {
      recognition.stop();
    } catch (e) {
      console.log("Voice recognition stop error:", e);
    }
  }

  if (!sessionId) {
    coachMessage.textContent = "No active session to stop.";
    return;
  }

  try {
    console.log("Stopping session:", sessionId);
    const res = await fetch(`/sessions/${sessionId}/stop`, { method: "POST" });

    if (!res.ok) {
      const err = await res.json();
      console.error("Stop session error:", err);
      coachMessage.textContent = `Error stopping session: ${err.detail || "Unknown error"}`;
      return;
    }

    const data = await res.json();

    // Display summary in a simple format
    const summary = `
Session Summary:
- Total Squats: ${data.total_reps || data.rep_count || 0}
- Valid Reps: ${data.valid_reps || data.valid_rep_count || data.rep_count || 0}
- Shallow Reps: ${data.shallow_rep_count || 0}
- Recommendation: ${data.recommendation || "Great effort!"}
    `.trim();

    summaryOutput.textContent = summary;
    updateCoachMessageStable("Great job! Session complete.", { force: true });
    speakMessage("Session complete. Great job!");

    console.log("Session stopped:", data);
  } catch (err) {
    console.error("Stop session network error:", err);
    coachMessage.textContent = `Connection error: ${err.message}`;
  }

  sessionId = null;
  stopSession.disabled = true;
  fetchSummary.disabled = true;
  repCountLabel.textContent = "0";
  if (overlayRepCount) overlayRepCount.textContent = "0";
  fatigueLabel.textContent = "-";
  if (overlayFatigue) overlayFatigue.textContent = "-";
  // Unlock profile selector for next session
  if (profileGrid) profileGrid.style.pointerEvents = "";
  if (profileStatus) profileStatus.textContent = "";
}

// =============== Text-to-Speech ===============
function speakMessage(message) {
  if (!("speechSynthesis" in window) || !message) return;
  const now = Date.now();
  const normalized = message.trim().toLowerCase();
  const isRepeat = normalized === lastSpokenMessage;
  if (isRepeat && now - lastSpokenAt < speakMinIntervalMs * 2) return;
  if (now - lastSpokenAt < speakMinIntervalMs) return;
  if (speechSynthesis.speaking || speechSynthesis.pending) return;

  const utterance = new SpeechSynthesisUtterance(message);
  utterance.rate = 0.93;
  utterance.pitch = 1;
  speechSynthesis.speak(utterance);
  lastSpokenMessage = normalized;
  lastSpokenAt = now;
}

// =============== Voice Commands ===============
function setupVoiceCommands() {
  recognition.onstart = () => {
    voiceStatus.textContent = "🎤 Listening...";
  };

  recognition.onresult = (event) => {
    let transcript = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript.toLowerCase();
    }

    // Process voice commands
    if (transcript.includes("stop") || transcript.includes("pause")) {
      coachMessage.textContent = "Pausing session...";
      speakMessage("Pausing");
      stopSession.click();
    } else if (transcript.includes("slower") || transcript.includes("speed")) {
      coachMessage.textContent = "Slow down your pace.";
      speakMessage("Slow down your pace");
    } else if (transcript.includes("faster")) {
      coachMessage.textContent = "Nice! Keep up the pace.";
      speakMessage("Nice, keep up the pace");
    } else if (transcript.includes("help") || transcript.includes("how")) {
      coachMessage.textContent =
        "Move in a smooth, pain-free range and keep your chest lifted.";
      speakMessage(
        "Move in a smooth, pain free range and keep your chest lifted",
      );
    }
  };

  recognition.onerror = () => {
    console.log("Voice recognition error");
  };

  recognition.onend = () => {
    // Restart recognition if still tracking
    if (tracking) {
      recognition.start();
    }
  };

  try {
    recognition.start();
  } catch (err) {
    console.log("Voice recognition already running or unavailable");
  }
}

// =============== Frame Capture & Sending ===============
async function sendFrame() {
  if (!sessionId || !tracking || !stream || isSendingFrame) return;

  const now = Date.now();
  if (now < frameBackoffUntil) return;
  if (now - lastFrameSent < sendIntervalMs) return;
  lastFrameSent = now;

  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) return;

  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, width, height);

  const dataUrl = canvas.toDataURL("image/jpeg", 0.6);
  const base64 = dataUrl.split(",")[1];
  if (!base64 || base64.length < 100) return;

  // Use MERaLiON strain score if available, otherwise use default
  const strainScore = clampStrainScore(currentStrainScore, 0);
  const audioAgeMs = lastAudioAnalysisAt
    ? Math.max(0, Date.now() - lastAudioAnalysisAt)
    : null;
  const effectiveAudioValid =
    !!audioSignalValid && audioAgeMs !== null && audioAgeMs <= 25000;

  const payload = {
    image_base64: base64,
    audio: {
      command_intent: "none",
      strain_score: strainScore,
      confidence: effectiveAudioValid ? 0.9 : 0.2,
      valid: effectiveAudioValid,
      audio_age_ms: audioAgeMs,
    },
    tempo_rps: 0.6, // Default tempo
    rest_gap_s: 0,
  };

  try {
    isSendingFrame = true;
    const res = await fetch(`/sessions/${sessionId}/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await parseJsonSafe(res);
      const errorDetail = err.detail || "Frame error";

      // Log frame details for debugging
      console.warn(`Frame error: ${errorDetail}`, {
        status: res.status,
        imageSize: { width, height },
        base64Length: base64.length,
      });
      debugLog("frame-ingest-fail", {
        status: res.status,
        detail: errorDetail,
        imageSize: { width, height },
        base64Length: base64.length,
      });

      if (res.status === 422) {
        ingest422Count += 1;
        const backoffMs =
          ingest422Count >= 10 ? 2200 : ingest422Count >= 4 ? 1200 : 500;
        frameBackoffUntil = Date.now() + backoffMs;
      }

      // Show helpful error message
      if (errorDetail.includes("No pose detected")) {
        updateCoachMessageStable(
          "Posture not detected. Make sure full body is visible and well-lit.",
        );
      } else if (errorDetail.includes("Key landmarks")) {
        updateCoachMessageStable(
          "Can't see your form clearly. Adjust the camera angle.",
        );
      } else if (errorDetail.includes("Frame error")) {
        updateCoachMessageStable("Checking form... Try different lighting.");
      } else {
        updateCoachMessageStable(`Form check: ${errorDetail}`);
      }
      return;
    }

    const data = await res.json();
    ingest422Count = 0;
    frameBackoffUntil = 0;
    debugLog("frame-ingest-ok", {
      phase: data.phase,
      rep_count: data.rep_count,
      fatigue_level: data.fatigue_level,
      fusion: {
        vision_valid: data.vision_valid,
        audio_valid: data.audio_valid,
        vision_age_ms: data.vision_age_ms,
        audio_age_ms: data.audio_age_ms,
      },
    });

    // Update UI
    phaseLabel.textContent = data.phase || "-";
    repCountLabel.textContent = data.rep_count || 0;
    if (overlayRepCount) overlayRepCount.textContent = data.rep_count || 0;

    // Form feedback
    let formText = data.form?.depth_quality || "good";
    if (data.form?.knee_tracking_warning) formText += " ⚠ knees";
    if (data.form?.torso_lean_warning) formText += " ⚠ back";
    formLabel.textContent = formText;

    const fatigueLevel = data.fatigue_level || "-";
    const fatigueReason = data.fatigue_reason || "";
    fatigueLabel.textContent = fatigueReason
      ? `${fatigueLevel} (${fatigueReason})`
      : fatigueLevel;
    if (overlayFatigue) overlayFatigue.textContent = fatigueLevel;
    if (!data.vision_valid) {
      cameraStatus.textContent = "⚠ Camera unclear - adjust angle/lighting";
    }
    const incomingMessage = data.coaching_message || "Good form!";
    const urgentMessage =
      data.fatigue_level === "HIGH" ||
      data.form?.knee_tracking_warning ||
      data.form?.torso_lean_warning;
    updateCoachMessageStable(incomingMessage, { force: urgentMessage });

    // Debug logging
    console.log("Frame processed:", {
      phase: data.phase,
      reps: data.rep_count,
      angles: {
        knee: data.frame.knee_angle_deg.toFixed(1),
        torso: data.frame.torso_lean_deg.toFixed(1),
        kneeOffset: data.frame.knee_inward_offset.toFixed(3),
      },
      form: data.form,
      fatigue: data.fatigue_level,
    });

    // Speak only for meaningful updates to avoid overlapping chatter.
    if (
      data.fatigue_level === "HIGH" ||
      data.form?.knee_tracking_warning ||
      data.form?.torso_lean_warning ||
      (data.coaching_message || "").toLowerCase() !== lastSpokenMessage
    ) {
      speakMessage(incomingMessage);
    }
  } catch (err) {
    console.error("Frame send error:", err);
    debugLog("frame-send-error", { error: String(err) });
    updateCoachMessageStable("Connection error. Check console.");
  } finally {
    isSendingFrame = false;
  }
}

// =============== Session Summary ===============
async function fetchSessionSummary() {
  if (!sessionId) return;
  try {
    const res = await fetch(`/sessions/${sessionId}/summary`);
    if (!res.ok) throw new Error("Summary not ready");
    const data = await res.json();

    const summary = `
Session Summary:
- Total Squats: ${data.rep_count || 0}
- Valid Reps: ${data.valid_rep_count || data.rep_count || 0}
- Shallow Reps: ${data.shallow_rep_count || 0}
- Fatigue Timeline: ${data.fatigue_levels?.join(", ") || "N/A"}
    `.trim();

    summaryOutput.textContent = summary;
  } catch (err) {
    summaryOutput.textContent = "Summary not available yet.";
  }
}

// =============== Animation Loop ===============
function loop() {
  sendFrame();
  requestAnimationFrame(loop);
}

// =============== Event Listeners ===============
startCamera.addEventListener("click", startCameraStream);
stopCamera.addEventListener("click", stopCameraStream);
startSession.addEventListener("click", createSession);
stopSession.addEventListener("click", endSession);
fetchSummary.addEventListener("click", fetchSessionSummary);

// Initial setup
checkApi();
loop();
