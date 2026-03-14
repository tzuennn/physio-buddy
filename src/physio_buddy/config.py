from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    meralion_base_url: str = os.getenv("MERALION_BASE_URL", "https://api.cr8lab.com")
    meralion_api_key: str | None = os.getenv("MERALION_API_KEY")
    meralion_timeout_s: float = float(os.getenv("MERALION_TIMEOUT_S", "20"))


settings = Settings()
