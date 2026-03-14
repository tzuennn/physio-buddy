from __future__ import annotations

from typing import Any

import httpx


class MeralionClient:
    """Thin client for MERaLiON Audio LLM API."""

    def __init__(self, base_url: str, api_key: str, timeout_s: float = 20.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_s

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(
            f"{self._base_url}{path}",
            json=payload,
            headers=self._headers(),
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("MERaLiON response must be a JSON object")
        return data

    def status(self) -> dict[str, Any]:
        response = httpx.get(f"{self._base_url}/status", timeout=self._timeout)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("MERaLiON status response must be a JSON object")
        return data

    def upload_url(self, filename: str, content_type: str = "audio/wav") -> dict[str, Any]:
        return self._post("/upload-url", {"filename": filename, "content_type": content_type})

    def upload_status(self, file_id: str) -> dict[str, Any]:
        return self._post("/upload-status", {"file_id": file_id})

    def transcribe(self, file_id: str, language: str = "en") -> dict[str, Any]:
        return self._post("/transcribe", {"file_id": file_id, "language": language})

    def analyze(self, file_id: str) -> dict[str, Any]:
        return self._post("/analyze", {"file_id": file_id})

    def process(self, file_id: str, instruction: str) -> dict[str, Any]:
        return self._post("/process", {"file_id": file_id, "instruction": instruction})
