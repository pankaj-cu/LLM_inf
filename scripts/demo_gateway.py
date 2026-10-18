"""Show the OpenInfer API with a temporary local mock gateway or a running gateway.

Without arguments, starts an isolated mock gateway on localhost, exercises its
OpenAI-compatible API, and shuts it down. For a running real gateway, supply
--base-url and OPENINFER_DEMO_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]


def unused_loopback_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def wait_ready(root_url: str, process: subprocess.Popen | None) -> dict:
    with httpx.Client(timeout=5.0) as client:
        for _ in range(50):
            if process is not None and process.poll() is not None:
                raise RuntimeError("Gateway exited before it became ready")
            try:
                response = client.get(root_url + "/health/ready")
                if response.status_code == 200:
                    return response.json()
                if process is None:
                    raise RuntimeError(
                        f"Gateway is not ready (HTTP {response.status_code}): "
                        f"{response.text[:300]}"
                    )
            except httpx.HTTPError:
                pass  # A newly started container may need a moment to bind its port.
            time.sleep(0.1)
    raise TimeoutError("Gateway did not become ready within 5 seconds")


def run_demo(base_url: str, api_key: str, model: str, local_mock: bool) -> dict:
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        raise ValueError("--base-url must end in /v1")
    root_url = base_url[:-3]
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=30, max_retries=0)
    with httpx.Client(timeout=10.0) as http_client:
        ready = wait_ready(root_url, None)
        available_models = [item.id for item in client.models.list().data]
        if model not in available_models:
            raise RuntimeError(f"Model {model!r} was not listed by the gateway")

        started = time.perf_counter()
        raw = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "Hello OpenInfer. Reply in one short sentence."}],
            max_tokens=32,
            temperature=0,
        )
        elapsed_seconds = time.perf_counter() - started
        response = raw.parse()
        metrics = http_client.get(root_url + "/metrics").json()
        invalid_key_status = None
        if local_mock:
            invalid = http_client.post(
                base_url + "/chat/completions",
                headers={"Authorization": "Bearer invalid-local-demo-key"},
                json={"model": model,
                      "messages": [{"role": "user", "content": "hello"}],
                      "max_tokens": 1},
            )
            invalid_key_status = invalid.status_code

    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend_mode": "mock" if local_mock else ready.get("backend_mode"),
        "ready": ready,
        "models": available_models,
        "chat_model": response.model,
        "chat_content": response.choices[0].message.content,
        "chat_elapsed_seconds": elapsed_seconds,
        "chat_request_id": raw.headers.get("x-request-id"),
        "completion_tokens": response.usage.completion_tokens if response.usage else None,
        "invalid_key_status": invalid_key_status,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="Existing gateway URL ending in /v1")
    parser.add_argument("--model", default="openinfer-qwen")
    parser.add_argument("--output", type=Path, help="Optional path for credential-free JSON")
    args = parser.parse_args()

    if args.base_url:
        api_key = os.getenv("OPENINFER_DEMO_API_KEY")
        if not api_key:
            parser.error("Set OPENINFER_DEMO_API_KEY for an existing gateway")
        record = run_demo(args.base_url, api_key, args.model, local_mock=False)
    else:
        api_key = secrets.token_urlsafe(24)
        port = unused_loopback_port()
        base_url = f"http://127.0.0.1:{port}/v1"
        env = os.environ.copy()
        for name in ("BACKEND_API_KEY", "BACKEND_BASE_URL", "VLLM_API_KEY", "COLAB_BRIDGE_KEY"):
            env.pop(name, None)
        env.update({"BACKEND_MODE": "mock", "PUBLIC_API_KEYS": api_key,
                    "PUBLIC_MODEL_ID": args.model, "RATE_LIMIT_PER_MINUTE": "100"})
        with tempfile.TemporaryDirectory(prefix="openinfer-demo-") as temp_dir:
            log_path = Path(temp_dir) / "gateway.log"
            with log_path.open("w") as log_handle:
                process = subprocess.Popen(
                    [sys.executable, "-m", "uvicorn", "gateway.app.main:app",
                     "--host", "127.0.0.1", "--port", str(port)],
                    cwd=ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT,
                )
                try:
                    try:
                        wait_ready(base_url[:-3], process)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Local mock gateway failed to start: {exc}. "
                            f"Recent log: {log_path.read_text()[-1000:]}"
                        ) from exc
                    record = run_demo(base_url, api_key, args.model, local_mock=True)
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)

    print(json.dumps(record, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print("Saved", args.output)


if __name__ == "__main__":
    main()
