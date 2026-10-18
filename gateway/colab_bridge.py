"""Restricted HTTP bridge for an optional, permitted remote Colab demo.

Point a tunnel at this service, never directly at the vLLM server. It exposes
only model listing and non-streaming chat for the optional remote demo.
"""

from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse


def create_app() -> FastAPI:
    proxy_key = os.getenv("COLAB_BRIDGE_KEY", "")
    vllm_key = os.getenv("VLLM_API_KEY", "")
    vllm_base_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")
    if not proxy_key or not vllm_key:
        raise RuntimeError("COLAB_BRIDGE_KEY and VLLM_API_KEY are required")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(
            base_url=vllm_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {vllm_key}"},
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        yield
        await app.state.client.aclose()

    app = FastAPI(
        title="OpenInfer Colab bridge",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    def authorize(header: str | None) -> None:
        if not header or not header.startswith("Bearer ") or not hmac.compare_digest(
            header.removeprefix("Bearer "), proxy_key
        ):
            raise HTTPException(status_code=401, detail="Invalid bridge credential")

    async def forward(path: str, payload: dict | None = None) -> JSONResponse:
        try:
            if payload is None:
                result = await app.state.client.get(path)
            else:
                result = await app.state.client.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=504, detail="vLLM timed out") from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail="vLLM is unavailable") from exc

        try:
            body = result.json()
        except ValueError as exc:
            raise HTTPException(status_code=502, detail="Invalid vLLM response") from exc
        return JSONResponse(content=body, status_code=result.status_code)

    @app.get("/v1/models")
    async def models(authorization: str | None = Header(default=None)) -> JSONResponse:
        authorize(authorization)
        return await forward("/models")

    @app.post("/v1/chat/completions")
    async def chat(
        request: Request, authorization: str | None = Header(default=None)
    ) -> JSONResponse:
        authorize(authorization)
        try:
            payload = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON") from exc
        if not isinstance(payload, dict) or payload.get("stream") is True:
            raise HTTPException(status_code=400, detail="Only non-streaming JSON requests are supported")
        return await forward("/chat/completions", payload)

    return app


app = create_app()
