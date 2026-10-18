"""Check local gateway streaming and cleanup without a GPU or network access."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gateway.app.backends import ChatStream, MockBackend, VLLMBackend
from gateway.app.main import create_app


MODEL = "openinfer-qwen"
KEY = "dev-key-change-me"
PATH = "/v1/chat/completions"
PAYLOAD = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "hello"}],
    "stream": True,
}


async def check_mock_sdk() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = os.environ.copy()
    env.update({"BACKEND_MODE": "mock", "PUBLIC_MODEL_ID": MODEL,
                "PUBLIC_API_KEYS": KEY, "RATE_LIMIT_PER_MINUTE": "100"})
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="openinfer-stream-") as temp:
        log_path = Path(temp) / "gateway.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "gateway.app.main:app",
                 "--host", "127.0.0.1", "--port", str(port)],
                cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
            )
            try:
                async with httpx.AsyncClient(timeout=2) as http:
                    for _ in range(50):
                        if process.poll() is not None:
                            raise RuntimeError("Gateway exited: " + log_path.read_text()[-1000:])
                        try:
                            ready = await http.get(f"http://127.0.0.1:{port}/health/ready")
                            if ready.status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        await asyncio.sleep(0.1)
                    else:
                        raise TimeoutError("Local gateway did not become ready")

                    def call_sdk():
                        with OpenAI(api_key=KEY, base_url=f"http://127.0.0.1:{port}/v1",
                                    timeout=5, max_retries=0) as sdk:
                            return list(sdk.chat.completions.create(**PAYLOAD))

                    chunks = await asyncio.to_thread(call_sdk)
                    assert all(chunk.model == MODEL for chunk in chunks)
                    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == (
                        "Mock backend received: hello"
                    )
                    assert chunks[-1].choices[0].finish_reason == "stop"
                    metrics = (await http.get(f"http://127.0.0.1:{port}/metrics")).json()
                    assert metrics["completed_requests"] == 1 and metrics["failed_requests"] == 0
                    assert metrics["active_requests"] == 0 and metrics["pending_requests"] == 0
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    print("Mock stream: OpenAI SDK chunks and final metrics OK")


async def check_vllm_adapter() -> None:
    backend = VLLMBackend("http://backend/v1", None, timeout=5)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == PATH
        assert json.loads(request.content)["stream"] is True
        events = [
            {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 0,
             "model": "Qwen/Qwen2.5-1.5B-Instruct",
             "choices": [{"index": 0, "delta": {"content": "GPU text"}, "finish_reason": None}]},
            {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 0,
             "model": "Qwen/Qwen2.5-1.5B-Instruct",
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        body += "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    await backend.client.aclose()
    backend.client = httpx.AsyncClient(base_url="http://backend/v1", transport=httpx.MockTransport(handler))
    app = create_app(backend=backend)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(PATH, json=PAYLOAD, headers={"Authorization": f"Bearer {KEY}"})
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "data: [DONE]\n\n" in response.text
        assert '"model":"openinfer-qwen"' in response.text
        assert "GPU text" in response.text
        assert (await client.get("/metrics")).json()["active_requests"] == 0
    await backend.close()
    print("vLLM adapter: SSE parsing and public model ID OK")


async def check_backend_errors() -> None:
    async def exercise(handler, expected_status: int, expected_code: str) -> None:
        backend = VLLMBackend("http://backend/v1", None, timeout=5)
        await backend.client.aclose()
        backend.client = httpx.AsyncClient(
            base_url="http://backend/v1", transport=httpx.MockTransport(handler)
        )
        app = create_app(backend=backend)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    PATH, json=PAYLOAD, headers={"Authorization": f"Bearer {KEY}"}
                )
                assert response.status_code == expected_status, response.text
                assert expected_code in response.text
                metrics = (await client.get("/metrics")).json()
                assert metrics["failed_requests"] == 1
                assert metrics["active_requests"] == 0 and metrics["pending_requests"] == 0
        finally:
            await backend.close()

    await exercise(
        lambda _request: httpx.Response(503, json={"error": "backend unavailable"}),
        503,
        "backend_error",
    )
    await exercise(
        lambda _request: httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            text='data: {"error":"generation failed"}\n\n',
        ),
        200,
        "event: error",
    )
    print("Backend failures: HTTP error before SSE and error event during SSE OK")


class SlowMockBackend(MockBackend):
    def __init__(self) -> None:
        super().__init__(MODEL)
        self.closed = False

    async def open_chat_stream(self, payload: dict[str, Any]) -> ChatStream:
        async def events() -> AsyncIterator[dict[str, Any] | str]:
            try:
                yield {"id": "chatcmpl-slow", "object": "chat.completion.chunk",
                       "created": 0, "model": MODEL,
                       "choices": [{"index": 0, "delta": {"role": "assistant"},
                                    "finish_reason": None}]}
                await asyncio.sleep(10)
                yield "[DONE]"
            finally:
                self.closed = True

        iterator = events()
        return ChatStream(iterator, iterator.aclose)


async def check_disconnect() -> None:
    backend = SlowMockBackend()
    app = create_app(backend=backend)
    body = json.dumps(PAYLOAD).encode()
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": PATH, "raw_path": PATH.encode(), "query_string": b"",
        "root_path": "", "client": ("127.0.0.1", 0), "server": ("test", 80),
        "headers": [(b"content-type", b"application/json"),
                    (b"authorization", f"Bearer {KEY}".encode())],
    }
    disconnect = asyncio.Event()
    request_sent = False
    saw_chunk = False

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        nonlocal saw_chunk
        if message["type"] == "http.response.body" and message.get("body"):
            saw_chunk = True
            disconnect.set()

    await asyncio.wait_for(app(scope, receive, send), timeout=3)
    assert saw_chunk and backend.closed
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        metrics = (await client.get("/metrics")).json()
    assert metrics["failed_requests"] == 1 and metrics["completed_requests"] == 0
    assert metrics["active_requests"] == 0 and metrics["pending_requests"] == 0
    print("Client disconnect: backend stream closed and admission slot released")


async def check_deadline() -> None:
    previous = os.environ.get("REQUEST_TIMEOUT_SECONDS")
    os.environ["REQUEST_TIMEOUT_SECONDS"] = "0.05"
    try:
        app = create_app(backend=SlowMockBackend())
    finally:
        if previous is None:
            os.environ.pop("REQUEST_TIMEOUT_SECONDS", None)
        else:
            os.environ["REQUEST_TIMEOUT_SECONDS"] = previous
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(PATH, json=PAYLOAD, headers={"Authorization": f"Bearer {KEY}"})
        assert response.status_code == 200
        assert "event: error" in response.text and "request_timeout" in response.text
        metrics = (await client.get("/metrics")).json()
        assert metrics["failed_requests"] == 1 and metrics["active_requests"] == 0
    print("Stream deadline: SSE error and admission cleanup OK")


async def main() -> None:
    await check_mock_sdk()
    await check_vllm_adapter()
    await check_backend_errors()
    await check_disconnect()
    await check_deadline()


if __name__ == "__main__":
    asyncio.run(main())
