# ── base image ────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

WORKDIR /app

# System libraries required by OpenCV and MediaPipe on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── dependency install ─────────────────────────────────────────────────────────
COPY pyproject.toml ./
COPY src/ ./src/

# Install with vision extras (MediaPipe + OpenCV)
RUN pip install --no-cache-dir -e ".[vision]"

# Pre-download the MediaPipe pose model at build time so the first request is fast.
# The model is cached into /app/.cache/models — mount a Docker volume there to
# avoid re-downloading on every container restart.
ENV MEDIAPIPE_POSE_TASK_AUTO_DOWNLOAD=true
ENV MEDIAPIPE_POSE_TASK_CACHE_DIR=/app/.cache/models

RUN python -c "\
from physio_buddy.mediapipe_pose import MediaPipePoseAnalyzer; \
a = MediaPipePoseAnalyzer(); a.close() \
" || echo "Model pre-download skipped (network unavailable during build)"

# ── runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8000

# Single worker — MediaPipe's pose analyzer is not fork-safe.
# Use --workers 1 and scale horizontally with multiple containers if needed.
CMD ["uvicorn", "physio_buddy.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
