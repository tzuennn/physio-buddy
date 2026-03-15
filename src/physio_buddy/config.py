from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Auto-load values from a local `.env` file for easier setup.
load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    meralion_base_url: str = os.getenv("MERALION_BASE_URL", "https://api.cr8lab.com")
    meralion_api_key: str | None = os.getenv("MERALION_API_KEY")
    meralion_timeout_s: float = float(os.getenv("MERALION_TIMEOUT_S", "20"))
    mediapipe_pose_task_model_path: str | None = os.getenv("MEDIAPIPE_POSE_TASK_MODEL_PATH")
    mediapipe_pose_task_auto_download: bool = os.getenv("MEDIAPIPE_POSE_TASK_AUTO_DOWNLOAD", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    mediapipe_pose_task_model_url: str = os.getenv(
        "MEDIAPIPE_POSE_TASK_MODEL_URL",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    )
    mediapipe_pose_task_cache_dir: str = os.getenv(
        "MEDIAPIPE_POSE_TASK_CACHE_DIR",
        "~/.cache/physio-buddy/models",
    )

    # LLM coach settings (OpenAI-compatible – also works with local Ollama):
    # LLM_API_URL=https://api.openai.com/v1  or  http://localhost:11434/v1
    # LLM_API_KEY=sk-...  (leave unset for Ollama which needs no key)
    # LLM_MODEL=gpt-4o-mini  or  llama3  or  any model your endpoint supports
    llm_api_url: str | None = os.getenv("LLM_API_URL")
    llm_api_key: str | None = os.getenv("LLM_API_KEY")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_timeout_s: float = float(os.getenv("LLM_TIMEOUT_S", "3.0"))


settings = Settings()
