from __future__ import annotations

import asyncio
import hmac
import json
import math
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from gateway.app.backends import Backend, BackendError, MockBackend, VLLMBackend


class ClosingStreamingResponse(StreamingResponse):
    """Close the generator when sending stops, including on client disconnect."""

    async def stream_response(self, send) -> None:
        try:
            await super().stream_response(send)
        finally:
            await self.body_iterator.aclose()


def _int_setting(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}")
    return value


def _positive_float_setting(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"{name} must be a positive number")
    return value


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)


class AdmissionController:
    def __init__(self, max_active: int, max_pending: int) -> None:
        self.max_active = max_active
        self.max_pending = max_pending
        self.semaphore = asyncio.Semaphore(max_active)
        self.lock = asyncio.Lock()
        self.active = 0
        self.pending = 0
        self.peak_active = 0
        self.peak_pending = 0

    @asynccontextmanager
    async def slot(self):
        queued_at = time.perf_counter()
        async with self.lock:
            if self.active >= self.max_active and self.pending >= self.max_pending:
                raise HTTPException(status_code=503, detail={
                    "error": {"message": "Gateway is at capacity", "type": "server_error",
                              "code": "overloaded"}
                })
            self.pending += 1
            self.peak_pending = max(self.peak_pending, self.pending)
        acquired = False
        try:
            await self.semaphore.acquire()
            acquired = True
            async with self.lock:
                self.pending -= 1
                self.active += 1
                self.peak_active = max(self.peak_active, self.active)
            yield time.perf_counter() - queued_at
        finally:
            async with self.lock:
                if acquired:
                    self.active -= 1
                else:
                    self.pending -= 1
            if acquired:
                self.semaphore.release()


class RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.limit = per_minute
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.monotonic()
        bucket = self.hits[key]
        while bucket and now - bucket[0] >= 60:
            bucket.popleft()
        if len(bucket) >= self.limit:
            retry_after = max(1, int(60 - (now - bucket[0])))
            raise HTTPException(status_code=429, detail={
                "error": {"message": "Rate limit exceeded", "type": "rate_limit_error",
                          "code": "rate_limit_exceeded"}
            }, headers={"Retry-After": str(retry_after)})
        bucket.append(now)


def create_app(backend: Backend | None = None) -> FastAPI:
    mode = os.getenv("BACKEND_MODE", "mock").lower()
    public_model_id = os.getenv("PUBLIC_MODEL_ID", "openinfer-qwen")
    backend_model_id = os.getenv("BACKEND_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    timeout_seconds = _positive_float_setting("REQUEST_TIMEOUT_SECONDS", 120)
    readiness_timeout_seconds = _positive_float_setting("READINESS_TIMEOUT_SECONDS", 5)
    if backend is None:
        if mode == "mock":
            backend = MockBackend(public_model_id)
        elif mode == "vllm":
            backend = VLLMBackend(
                base_url=os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8001/v1"),
                api_key=os.getenv("BACKEND_API_KEY") or None,
                timeout=timeout_seconds,
            )
        else:
            raise RuntimeError("BACKEND_MODE must be 'mock' or 'vllm'")

    public_keys = [
        item.strip() for item in os.getenv("PUBLIC_API_KEYS", "dev-key-change-me").split(",")
        if item.strip()
    ]
    if not public_keys:
        raise RuntimeError("Set at least one key in PUBLIC_API_KEYS")
    admission = AdmissionController(
        _int_setting("MAX_ACTIVE_REQUESTS", 2),
        _int_setting("MAX_PENDING_REQUESTS", 8, minimum=0),
    )
    limiter = RateLimiter(_int_setting("RATE_LIMIT_PER_MINUTE", 30))
    metrics: dict[str, Any] = {
        "total_requests": 0,
        "completed_requests": 0,
        "failed_requests": 0,
        "latency_seconds": [],
        "queue_wait_seconds": [],
    }

    def record_sample(name: str, value: float) -> None:
        samples = metrics[name]
        samples.append(value)
        if len(samples) > 1000:
            del samples[:-1000]

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await backend.close()

    app = FastAPI(title="OpenInfer Gateway", version="0.1.0", lifespan=lifespan)
    app.state.backend = backend
    app.state.admission = admission

    def authenticate(authorization: str | None) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail={
                "error": {"message": "Missing or invalid API key", "type": "authentication_error",
                          "code": "invalid_api_key"}
            }, headers={"WWW-Authenticate": "Bearer"})
        candidate = authorization.removeprefix("Bearer ")
        if not any(hmac.compare_digest(candidate, valid) for valid in public_keys):
            raise HTTPException(status_code=401, detail={
                "error": {"message": "Missing or invalid API key", "type": "authentication_error",
                          "code": "invalid_api_key"}
            }, headers={"WWW-Authenticate": "Bearer"})
        return candidate

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "backend_mode": mode}

    @app.get("/health/ready")
    async def readiness() -> dict[str, Any]:
        try:
            async with asyncio.timeout(readiness_timeout_seconds):
                models = await backend.list_models()
        except TimeoutError as exc:
            raise HTTPException(status_code=503, detail={
                "status": "not_ready", "reason": "backend readiness timed out"
            }) from exc
        except BackendError as exc:
            raise HTTPException(status_code=503, detail={
                "status": "not_ready", "reason": str(exc)
            }) from exc
        available = {item.get("id") for item in models}
        if not isinstance(backend, MockBackend) and backend_model_id not in available:
            raise HTTPException(status_code=503, detail={
                "status": "not_ready", "reason": "configured model is not available"
            })
        return {"status": "ready", "backend_mode": mode}

    @app.get("/metrics")
    async def get_metrics() -> dict[str, Any]:
        def percentile(samples: list[float], fraction: float) -> float | None:
            values = sorted(samples)
            if not values:
                return None
            return values[min(len(values) - 1, math.ceil(len(values) * fraction) - 1)]

        return {
            "total_requests": metrics["total_requests"],
            "completed_requests": metrics["completed_requests"],
            "failed_requests": metrics["failed_requests"],
            "active_requests": admission.active,
            "pending_requests": admission.pending,
            "peak_active_requests": admission.peak_active,
            "peak_pending_requests": admission.peak_pending,
            "latency_p50_seconds": percentile(metrics["latency_seconds"], 0.50),
            "latency_p95_seconds": percentile(metrics["latency_seconds"], 0.95),
            "queue_wait_p50_seconds": percentile(metrics["queue_wait_seconds"], 0.50),
            "queue_wait_p95_seconds": percentile(metrics["queue_wait_seconds"], 0.95),
        }

    @app.get("/v1/models")
    async def list_models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        authenticate(authorization)
        try:
            async with asyncio.timeout(readiness_timeout_seconds):
                models = await backend.list_models()
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail={
                "error": {"message": "Model listing timed out", "type": "server_error",
                          "code": "backend_timeout"}
            }) from exc
        except BackendError as exc:
            raise HTTPException(status_code=exc.status_code, detail={
                "error": {"message": str(exc), "type": "server_error", "code": "backend_unavailable"}
            }) from exc
        if isinstance(backend, MockBackend):
            return {"object": "list", "data": models}
        matches = [item for item in models if item.get("id") == backend_model_id]
        return {
            "object": "list",
            "data": [{**item, "id": public_model_id} for item in matches],
        }

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        body: ChatRequest,
        response: Response,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any] | StreamingResponse:
        key = authenticate(authorization)
        limiter.check(key)
        if body.model != public_model_id:
            raise HTTPException(status_code=404, detail={
                "error": {"message": f"Model '{body.model}' is not available",
                          "type": "invalid_request_error", "code": "model_not_found"}
            })
        request_id = "req_" + uuid.uuid4().hex
        response.headers["X-Request-ID"] = request_id
        payload = body.model_dump(exclude_none=True)
        payload["model"] = (
            public_model_id if isinstance(backend, MockBackend) else backend_model_id
        )
        started = time.perf_counter()
        metrics["total_requests"] += 1
        if body.stream:
            deadline = started + timeout_seconds
            slot = admission.slot()
            acquired = False
            stream = None
            try:
                queue_wait = await asyncio.wait_for(
                    slot.__aenter__(), timeout=max(0, deadline - time.perf_counter())
                )
                acquired = True
                record_sample("queue_wait_seconds", queue_wait)
                stream = await asyncio.wait_for(
                    backend.open_chat_stream(payload),
                    timeout=max(0, deadline - time.perf_counter()),
                )
            except (TimeoutError, BackendError, HTTPException) as exc:
                metrics["failed_requests"] += 1
                record_sample("latency_seconds", time.perf_counter() - started)
                if acquired:
                    await slot.__aexit__(None, None, None)
                if isinstance(exc, TimeoutError):
                    raise HTTPException(status_code=504, detail={
                        "error": {"message": "Request deadline exceeded", "type": "server_error",
                                  "code": "request_timeout", "request_id": request_id}
                    }, headers={"X-Request-ID": request_id}) from exc
                if isinstance(exc, BackendError):
                    raise HTTPException(status_code=exc.status_code, detail={
                        "error": {"message": str(exc), "type": "server_error",
                                  "code": "backend_error", "request_id": request_id}
                    }, headers={"X-Request-ID": request_id}) from exc
                exc.headers = {**(exc.headers or {}), "X-Request-ID": request_id}
                raise
            except asyncio.CancelledError:
                metrics["failed_requests"] += 1
                record_sample("latency_seconds", time.perf_counter() - started)
                if acquired:
                    await slot.__aexit__(None, None, None)
                raise

            async def stream_body():
                completed = False
                counted = False
                try:
                    async with asyncio.timeout(max(0, deadline - time.perf_counter())):
                        async for event in stream.events:
                            if event == "[DONE]":
                                metrics["completed_requests"] += 1
                                counted = completed = True
                                yield b"data: [DONE]\n\n"
                                break
                            event["model"] = public_model_id
                            yield ("data: " + json.dumps(event, separators=(",", ":")) + "\n\n").encode()
                        if not completed:
                            raise BackendError("Inference backend ended the stream early", 502)
                except (TimeoutError, BackendError) as exc:
                    metrics["failed_requests"] += 1
                    counted = True
                    code = "request_timeout" if isinstance(exc, TimeoutError) else "backend_error"
                    error = {"error": {"message": str(exc) if code == "backend_error"
                                       else "Request deadline exceeded", "type": "server_error",
                                       "code": code, "request_id": request_id}}
                    yield ("event: error\ndata: " + json.dumps(error) + "\n\n").encode()
                finally:
                    if not counted:
                        metrics["failed_requests"] += 1
                    try:
                        await stream.aclose()
                    finally:
                        await slot.__aexit__(None, None, None)
                        record_sample("latency_seconds", time.perf_counter() - started)

            return ClosingStreamingResponse(
                stream_body(), media_type="text/event-stream",
                headers={
                    "X-Request-ID": request_id,
                    "X-Queue-Wait-Seconds": f"{queue_wait:.6f}",
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            async with asyncio.timeout(timeout_seconds):
                async with admission.slot() as queue_wait:
                    response.headers["X-Queue-Wait-Seconds"] = f"{queue_wait:.6f}"
                    record_sample("queue_wait_seconds", queue_wait)
                    result = await backend.chat_completion(payload)
            result["model"] = public_model_id
            metrics["completed_requests"] += 1
            return result
        except TimeoutError as exc:
            metrics["failed_requests"] += 1
            raise HTTPException(status_code=504, detail={
                "error": {"message": "Request deadline exceeded", "type": "server_error",
                          "code": "request_timeout", "request_id": request_id}
            }, headers={"X-Request-ID": request_id}) from exc
        except BackendError as exc:
            metrics["failed_requests"] += 1
            raise HTTPException(status_code=exc.status_code, detail={
                "error": {"message": str(exc), "type": "server_error", "code": "backend_error",
                          "request_id": request_id}
            }, headers={"X-Request-ID": request_id}) from exc
        except HTTPException as exc:
            metrics["failed_requests"] += 1
            exc.headers = {**(exc.headers or {}), "X-Request-ID": request_id}
            raise
        except asyncio.CancelledError:
            metrics["failed_requests"] += 1
            raise
        finally:
            record_sample("latency_seconds", time.perf_counter() - started)

    return app


app = create_app()
