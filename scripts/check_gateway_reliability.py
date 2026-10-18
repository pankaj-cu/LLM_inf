"""Exercise gateway outage and timeout behavior against a local fake backend.

No Colab runtime, tunnel, .env file, or real API key is used. The script starts a
temporary gateway process and HTTP backend on loopback, then saves a result.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PUBLIC_MODEL_ID = "openinfer-qwen"
PUBLIC_KEY = "local-reliability-demo-key"


def unused_loopback_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class FakeBackendHandler(BaseHTTPRequestHandler):
    slow_chat = False
    slow_models = False

    def log_message(self, _format: str, *_args: object) -> None:
        pass

    def send_json(self, status: int, body: dict) -> None:
        content = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass  # The gateway may have already timed out the slow response.

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            if self.slow_models:
                time.sleep(1.2)
            self.send_json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if self.slow_chat:
            time.sleep(1.2)
        self.send_json(200, {
            "id": "chatcmpl-local-fake",
            "object": "chat.completion",
            "created": 0,
            "model": MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })


def summarize(response: httpx.Response, elapsed: float) -> dict:
    body = response.json()
    detail = body.get("detail", {}) if isinstance(body, dict) else {}
    error = detail.get("error", {}) if isinstance(detail, dict) else {}
    return {
        "status_code": response.status_code,
        "elapsed_seconds": elapsed,
        "error_code": error.get("code"),
        "request_id_header": response.headers.get("x-request-id"),
        "request_id_body": error.get("request_id"),
    }


def main() -> None:
    backend = ThreadingHTTPServer(("127.0.0.1", 0), FakeBackendHandler)
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()
    gateway_port = unused_loopback_port()
    gateway_url = f"http://127.0.0.1:{gateway_port}"
    env = os.environ.copy()
    env.update({
        "BACKEND_MODE": "vllm",
        "BACKEND_BASE_URL": f"http://127.0.0.1:{backend.server_address[1]}/v1",
        "BACKEND_API_KEY": "local-fake-backend-key",
        "BACKEND_MODEL_ID": MODEL_ID,
        "PUBLIC_MODEL_ID": PUBLIC_MODEL_ID,
        "PUBLIC_API_KEYS": PUBLIC_KEY,
        "REQUEST_TIMEOUT_SECONDS": "0.4",
        "READINESS_TIMEOUT_SECONDS": "0.3",
        "RATE_LIMIT_PER_MINUTE": "100",
    })
    log_path = Path("/tmp/openinfer-reliability-gateway.log")
    log_handle = log_path.open("w")
    gateway = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "gateway.app.main:app", "--host", "127.0.0.1",
         "--port", str(gateway_port)],
        env=env, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT,
    )
    backend_stopped = False
    record = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "scenarios": {}}
    try:
        with httpx.Client(timeout=3.0) as client:
            for _ in range(40):
                if gateway.poll() is not None:
                    raise RuntimeError("Gateway exited early: " + log_path.read_text()[-2000:])
                try:
                    if client.get(gateway_url + "/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            else:
                raise TimeoutError("Gateway did not start")

            headers = {"Authorization": f"Bearer {PUBLIC_KEY}"}
            payload = {"model": PUBLIC_MODEL_ID,
                       "messages": [{"role": "user", "content": "hello"}],
                       "max_tokens": 1, "stream": False}

            def request(method: str, path: str, **kwargs: object) -> dict:
                started = time.perf_counter()
                response = client.request(method, gateway_url + path, **kwargs)
                return summarize(response, time.perf_counter() - started)

            record["scenarios"]["healthy_readiness"] = request("GET", "/health/ready")
            record["scenarios"]["invalid_key_chat"] = request(
                "POST", "/v1/chat/completions",
                headers={"Authorization": "Bearer wrong-local-key"}, json=payload,
            )
            record["scenarios"]["healthy_chat"] = request(
                "POST", "/v1/chat/completions", headers=headers, json=payload
            )
            FakeBackendHandler.slow_models = True
            record["scenarios"]["slow_readiness"] = request("GET", "/health/ready")
            record["scenarios"]["slow_models"] = request(
                "GET", "/v1/models", headers=headers
            )
            FakeBackendHandler.slow_models = False
            FakeBackendHandler.slow_chat = True
            record["scenarios"]["slow_chat"] = request(
                "POST", "/v1/chat/completions", headers=headers, json=payload
            )

            backend.shutdown()
            backend.server_close()
            backend_thread.join(timeout=2)
            backend_stopped = True
            record["scenarios"]["unavailable_readiness"] = request("GET", "/health/ready")
            record["scenarios"]["unavailable_models"] = request(
                "GET", "/v1/models", headers=headers
            )
            record["scenarios"]["unavailable_chat"] = request(
                "POST", "/v1/chat/completions", headers=headers, json=payload
            )
            record["metrics"] = client.get(gateway_url + "/metrics").json()

        output = ROOT / "results" / "gateway_reliability.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print("Saved", output)
        for name, result in record["scenarios"].items():
            print(name, result)

        expected = {
            "healthy_readiness": 200,
            "invalid_key_chat": 401,
            "healthy_chat": 200,
            "slow_readiness": 503,
            "slow_models": 504,
            "slow_chat": 504,
            "unavailable_readiness": 503,
            "unavailable_models": 503,
            "unavailable_chat": 503,
        }
        errors = [f"{name}: expected {status}, got {record['scenarios'][name]['status_code']}"
                  for name, status in expected.items()
                  if record["scenarios"][name]["status_code"] != status]
        if record["scenarios"]["slow_chat"]["elapsed_seconds"] > 2:
            errors.append("Slow chat was not bounded within 2 seconds")
        if record["scenarios"]["slow_readiness"]["elapsed_seconds"] > 1:
            errors.append("Slow readiness was not bounded within 1 second")
        if record["scenarios"]["slow_models"]["elapsed_seconds"] > 1:
            errors.append("Slow model listing was not bounded within 1 second")
        if record["scenarios"]["unavailable_chat"]["elapsed_seconds"] > 2:
            errors.append("Unavailable chat was not bounded within 2 seconds")
        for name in ("healthy_chat", "slow_chat", "unavailable_chat"):
            item = record["scenarios"][name]
            if not item["request_id_header"]:
                errors.append(f"{name} has no X-Request-ID header")
            if item["request_id_body"] and item["request_id_header"] != item["request_id_body"]:
                errors.append(f"{name} has mismatched request IDs")
        metrics = record["metrics"]
        if record["scenarios"]["invalid_key_chat"]["error_code"] != "invalid_api_key":
            errors.append("Invalid key did not return invalid_api_key")
        if (metrics["total_requests"], metrics["completed_requests"],
                metrics["failed_requests"]) != (3, 1, 2):
            errors.append("Gateway request counters do not match the three chat scenarios")
        if metrics["active_requests"] != 0 or metrics["pending_requests"] != 0:
            errors.append("Gateway left requests active or pending after the scenarios")
        if metrics["latency_p95_seconds"] is None or metrics["latency_p95_seconds"] < 0.3:
            errors.append("Gateway p95 latency did not capture the slow chat")
        if errors:
            raise AssertionError("; ".join(errors))
    finally:
        if not backend_stopped:
            backend.shutdown()
            backend.server_close()
            backend_thread.join(timeout=2)
        gateway.terminate()
        try:
            gateway.wait(timeout=3)
        except subprocess.TimeoutExpired:
            gateway.kill()
            gateway.wait(timeout=3)
        log_handle.close()


if __name__ == "__main__":
    main()
