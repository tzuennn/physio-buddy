from __future__ import annotations

from typing import Any, BinaryIO

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
        return {"x-api-key": self._api_key}

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
        response = httpx.get(
            f"{self._base_url}/status",
            headers=self._headers(),
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("MERaLiON status response must be a JSON object")
        return data

    def upload_url(self, filename: str, content_type: str, file_size: int) -> dict[str, Any]:
        return self._post(
            "/upload-url",
            {"fileName": filename, "contentType": content_type, "fileSize": file_size},
        )

    def upload_file(self, url: str, content: bytes | BinaryIO, content_type: str) -> None:
        response = httpx.put(
            url,
            content=content,
            headers={"Content-Type": content_type},
            timeout=self._timeout,
        )
        response.raise_for_status()

    def upload_status(self, file_key: str) -> dict[str, Any]:
        return self._post("/upload-status", {"fileKey": file_key})

    def transcribe(self, file_key: str) -> dict[str, Any]:
        return self._post("/transcribe", {"fileKey": file_key})

    def summarize(self, file_key: str) -> dict[str, Any]:
        return self._post("/summarize", {"fileKey": file_key})

    def translate(self, file_key: str, language: str) -> dict[str, Any]:
        return self._post("/translate", {"fileKey": file_key, "language": language})

    def analyze(self, file_key: str) -> dict[str, Any]:
        return self._post("/analyze", {"fileKey": file_key})

    def process(self, file_key: str, instruction: str) -> dict[str, Any]:
        return self._post("/process", {"fileKey": file_key, "instruction": instruction})
