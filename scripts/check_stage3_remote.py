"""Verify Codespaces gateway -> tunnel -> Colab vLLM without logging secrets."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import dotenv_values
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
PORT = 8765
BASE_URL = f"http://127.0.0.1:{PORT}"


def main() -> None:
    settings = dotenv_values(ROOT / ".env")
    required = (
        "BACKEND_BASE_URL",
        "BACKEND_API_KEY",
        "BACKEND_MODEL_ID",
        "PUBLIC_MODEL_ID",
        "PUBLIC_API_KEYS",
    )
    missing = [name for name in required if not settings.get(name)]
    if missing:
        raise SystemExit(f"Missing .env settings: {', '.join(missing)}")
    if settings.get("BACKEND_MODE") != "vllm":
        raise SystemExit("Set BACKEND_MODE=vllm in .env")

    env = os.environ.copy()
    env.update({key: value for key, value in settings.items() if value is not None})
    log_path = Path("/tmp/openinfer_stage3_gateway.log")
    with log_path.open("w") as log:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "gateway.app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(PORT),
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            for _ in range(30):
                if server.poll() is not None:
                    raise RuntimeError("Gateway exited during startup")
                try:
                    if httpx.get(f"{BASE_URL}/health", timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.5)
            else:
                raise TimeoutError("Gateway did not start")

            readiness = httpx.get(f"{BASE_URL}/health/ready", timeout=30)
            if readiness.status_code != 200:
                raise RuntimeError(
                    f"Backend readiness returned {readiness.status_code}: {readiness.text[:500]}"
                )

            public_key = settings["PUBLIC_API_KEYS"].split(",")[0].strip()
            client = OpenAI(
                api_key=public_key,
                base_url=f"{BASE_URL}/v1",
                timeout=120,
                max_retries=0,
            )
            model_id = settings["PUBLIC_MODEL_ID"]
            models = client.models.list()
            if model_id not in {item.id for item in models.data}:
                raise RuntimeError(f"Public model {model_id!r} is not listed")

            started = time.perf_counter()
            raw = client.chat.completions.with_raw_response.create(
                model=model_id,
                messages=[{"role": "user", "content": "Say hello in one short sentence."}],
                max_tokens=48,
                temperature=0,
            )
            elapsed = time.perf_counter() - started
            completion = raw.parse()
            metrics = httpx.get(f"{BASE_URL}/metrics", timeout=5).json()
            record = {
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "route": "Codespaces gateway -> Cloudflare Tunnel -> Colab bridge -> vLLM",
                "model_id": model_id,
                "backend_model_id": settings["BACKEND_MODEL_ID"],
                "elapsed_seconds": elapsed,
                "request_id": raw.headers.get("x-request-id"),
                "completion_tokens": (
                    completion.usage.completion_tokens if completion.usage else None
                ),
                "response": completion.choices[0].message.content,
                "gateway_metrics": metrics,
            }
            output = ROOT / "results" / "stage3_remote_integration.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print("PASS: gateway readiness, model listing, and OpenAI SDK chat")
            print(f"Model: {model_id}; latency: {elapsed:.2f}s")
            print("Response:", record["response"])
            print("Saved:", output)
        except Exception:
            log.flush()
            print(f"Gateway log: {log_path}", file=sys.stderr)
            raise
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


if __name__ == "__main__":
    main()
