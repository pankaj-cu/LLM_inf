from __future__ import annotations

from typing import Any, Protocol

import httpx


class BackendError(Exception):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class Backend(Protocol):
    async def list_models(self) -> list[dict[str, Any]]: ...

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def close(self) -> None: ...


class VLLMBackend:
    """Adapter for vLLM's OpenAI-compatible API."""

    def __init__(self, base_url: str, api_key: str | None, timeout: float) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout
        )

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            response = await self.client.get("/models")
            response.raise_for_status()
            data = response.json().get("data", [])
            return data if isinstance(data, list) else []
        except httpx.TimeoutException as exc:
            raise BackendError("Inference backend timed out", 504) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise BackendError("Inference backend is unavailable", 503) from exc

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self.client.post("/chat/completions", json=payload)
            if response.is_error:
                raise BackendError(
                    f"Inference backend rejected the request ({response.status_code})",
                    503 if response.status_code >= 500 else 400,
                )
            result = response.json()
            if not isinstance(result, dict):
                raise BackendError("Inference backend returned an invalid response", 502)
            return result
        except httpx.TimeoutException as exc:
            raise BackendError("Inference backend timed out", 504) from exc
        except httpx.HTTPError as exc:
            raise BackendError("Inference backend is unavailable", 503) from exc
        except ValueError as exc:
            raise BackendError("Inference backend returned invalid JSON", 502) from exc

    async def close(self) -> None:
        await self.client.aclose()


class MockBackend:
    """Deterministic local backend for development; not the inference engine."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    async def list_models(self) -> list[dict[str, Any]]:
        return [{"id": self.model_id, "object": "model", "owned_by": "openinfer-mock"}]

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        content = payload["messages"][-1].get("content", "")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": self.model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": f"Mock backend received: {content}"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def close(self) -> None:
        return None
