from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol

import httpx


class BackendError(Exception):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class Backend(Protocol):
    async def list_models(self) -> list[dict[str, Any]]: ...

    async def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def open_chat_stream(self, payload: dict[str, Any]) -> ChatStream: ...

    async def close(self) -> None: ...


class ChatStream:
    def __init__(
        self,
        events: AsyncIterator[dict[str, Any] | str],
        close: Callable[[], Awaitable[None]],
    ) -> None:
        self.events = events
        self._close = close

    async def aclose(self) -> None:
        await self._close()


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

    async def open_chat_stream(self, payload: dict[str, Any]) -> ChatStream:
        try:
            request = self.client.build_request("POST", "/chat/completions", json=payload)
            response = await self.client.send(request, stream=True)
        except httpx.TimeoutException as exc:
            raise BackendError("Inference backend timed out", 504) from exc
        except httpx.HTTPError as exc:
            raise BackendError("Inference backend is unavailable", 503) from exc

        if response.is_error:
            status_code = response.status_code
            await response.aclose()
            raise BackendError(
                f"Inference backend rejected the request ({status_code})",
                503 if status_code >= 500 else 400,
            )
        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
        if content_type != "text/event-stream":
            await response.aclose()
            raise BackendError("Inference backend returned a non-streaming response", 502)

        async def events() -> AsyncIterator[dict[str, Any] | str]:
            data_lines: list[str] = []
            done = False

            def parse_event() -> dict[str, Any] | str:
                data = "\n".join(data_lines)
                if data == "[DONE]":
                    return data
                try:
                    event = json.loads(data)
                except ValueError as exc:
                    raise BackendError("Inference backend returned invalid stream data", 502) from exc
                if not isinstance(event, dict):
                    raise BackendError("Inference backend returned invalid stream data", 502)
                if "error" in event:
                    raise BackendError("Inference backend failed during streaming", 502)
                return event

            try:
                async for line in response.aiter_lines():
                    if line == "":
                        if not data_lines:
                            continue
                        event = parse_event()
                        data_lines.clear()
                        yield event
                        if event == "[DONE]":
                            done = True
                            break
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].lstrip(" "))
                if not done and data_lines:
                    event = parse_event()
                    yield event
                    done = event == "[DONE]"
                if not done:
                    raise BackendError("Inference backend ended the stream early", 502)
            except httpx.TimeoutException as exc:
                raise BackendError("Inference backend timed out", 504) from exc
            except httpx.HTTPError as exc:
                raise BackendError("Inference backend is unavailable", 503) from exc
            finally:
                await response.aclose()

        iterator = events()

        async def close() -> None:
            try:
                await iterator.aclose()
            finally:
                await response.aclose()

        return ChatStream(iterator, close)

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

    async def open_chat_stream(self, payload: dict[str, Any]) -> ChatStream:
        result = await self.chat_completion(payload)
        common = {
            "id": result["id"],
            "object": "chat.completion.chunk",
            "created": result["created"],
            "model": result["model"],
        }

        async def events() -> AsyncIterator[dict[str, Any] | str]:
            yield {**common, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                            "finish_reason": None}]}
            yield {**common, "choices": [{"index": 0, "delta": {
                "content": result["choices"][0]["message"]["content"]
            }, "finish_reason": None}]}
            yield {**common, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield "[DONE]"

        iterator = events()
        return ChatStream(iterator, iterator.aclose)

    async def close(self) -> None:
        return None
