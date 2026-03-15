from __future__ import annotations

import time
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
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body_preview = response.text[:300] if response.text else ""
            raise httpx.HTTPStatusError(
                f"{exc}. Response: {body_preview}",
                request=exc.request,
                response=exc.response,
            ) from exc
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("MERaLiON response must be a JSON object")
        return data

    @staticmethod
    def _unwrap_response(data: dict[str, Any]) -> dict[str, Any]:
        """MERaLiON often nests actual data under a top-level `response` field."""
        nested = data.get("response")
        if isinstance(nested, dict):
            return nested
        return data

    def _post_with_key(self, path: str, file_key: str, **extra: Any) -> dict[str, Any]:
        """Try both `fileKey` and `key` payload conventions for compatibility."""
        last_exc: Exception | None = None
        for key_field in ("fileKey", "key"):
            payload: dict[str, Any] = {key_field: file_key}
            payload.update(extra)
            try:
                data = self._post(path, payload)
                return self._unwrap_response(data)
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                # For validation-style client errors, try the alternate key field.
                if exc.response is not None and exc.response.status_code in {400, 422}:
                    continue
                raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Unable to call {path} with key payload")

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
        # Send both camelCase and snake_case/common aliases for robust compatibility.
        data = self._post(
            "/upload-url",
            {
                "filename": filename,
                "fileName": filename,
                "contentType": content_type,
                "fileSize": file_size,
                "file_size": file_size,
            },
        )
        response = self._unwrap_response(data)
        # Normalize key names so callers can depend on `fileKey`.
        if "fileKey" not in response and "key" in response:
            response["fileKey"] = response["key"]
        return response

    def upload_file(self, url: str, content: bytes | BinaryIO, content_type: str) -> None:
        response = httpx.put(
            url,
            content=content,
            headers={"Content-Type": content_type},
            timeout=self._timeout,
        )
        response.raise_for_status()

    def upload_status(self, file_key: str) -> dict[str, Any]:
        return self._post_with_key("/upload-status", file_key)

    @staticmethod
    def _is_ready_status(status_payload: dict[str, Any]) -> bool:
        # Handle a wide range of status response shapes conservatively.
        merged = {**status_payload}
        nested = merged.get("status")
        if isinstance(nested, dict):
            merged.update(nested)

        for k in ("uploaded", "ready", "isReady", "completed", "done", "success"):
            v = merged.get(k)
            if isinstance(v, bool) and v:
                return True

        for k in ("state", "status", "uploadStatus", "processingStatus"):
            v = merged.get(k)
            if isinstance(v, str) and v.strip().lower() in {
                "uploaded",
                "ready",
                "complete",
                "completed",
                "done",
                "success",
                "ok",
            }:
                return True

        # If status payload is ambiguous, let downstream API call determine readiness.
        return False

    def wait_until_ready(self, file_key: str, timeout_s: float = 20.0) -> bool:
        deadline = time.time() + timeout_s
        delay = 0.3
        while time.time() < deadline:
            try:
                status = self.upload_status(file_key)
                if self._is_ready_status(status):
                    return True
            except httpx.HTTPStatusError as exc:
                if exc.response is None or exc.response.status_code not in {400, 404, 429, 500, 503, 504}:
                    raise
            time.sleep(delay)
            delay = min(delay * 1.6, 1.2)
        return False

    def _post_with_retry(self, path: str, file_key: str, **extra: Any) -> dict[str, Any]:
        delay = 0.45
        last_exc: Exception | None = None
        for _ in range(8):
            try:
                return self._post_with_key(path, file_key, **extra)
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response is None or exc.response.status_code not in {404, 429, 500, 503, 504}:
                    raise
                time.sleep(delay)
                delay = min(delay * 1.7, 3.0)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Unable to call {path}")

    def transcribe(self, file_key: str) -> dict[str, Any]:
        self.wait_until_ready(file_key)
        return self._post_with_retry("/transcribe", file_key)

    def summarize(self, file_key: str) -> dict[str, Any]:
        self.wait_until_ready(file_key)
        return self._post_with_retry("/summarize", file_key)

    def translate(self, file_key: str, language: str) -> dict[str, Any]:
        self.wait_until_ready(file_key)
        return self._post_with_retry("/translate", file_key, language=language)

    def analyze(self, file_key: str) -> dict[str, Any]:
        self.wait_until_ready(file_key)
        return self._post_with_retry("/analyze", file_key, segment_length=10)

    def process(self, file_key: str, instruction: str) -> dict[str, Any]:
        self.wait_until_ready(file_key)
        return self._post_with_retry("/process", file_key, instruction=instruction)
