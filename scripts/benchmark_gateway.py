"""Run a small, repeatable non-streaming concurrency experiment.

Targets either the local OpenInfer gateway or the authenticated Colab bridge.
Both modes use the same prompts so their results can be compared cautiously.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
GATEWAY_PORT = 8766
PROMPT_TEMPLATE = (
    "Explain one practical use of a transformer KV cache in one short sentence. "
    "Request number {number}."
)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


async def wait_for_gateway(process: subprocess.Popen[Any]) -> None:
    url = f"http://127.0.0.1:{GATEWAY_PORT}/health/ready"
    async with httpx.AsyncClient(timeout=15) as client:
        for _ in range(30):
            if process.poll() is not None:
                raise RuntimeError("Gateway exited before becoming ready")
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
    raise TimeoutError("Gateway or remote backend did not become ready")


async def run_level(
    client: httpx.AsyncClient,
    url: str,
    key: str,
    model: str,
    concurrency: int,
    request_count: int,
    max_tokens: int,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one_request(index: int) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(number=index)}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
        async with semaphore:
            started = time.perf_counter()
            try:
                response = await client.post(
                    url, headers={"Authorization": f"Bearer {key}"}, json=payload
                )
                elapsed = time.perf_counter() - started
                try:
                    body = response.json()
                except ValueError:
                    body = {}
                queue_header = response.headers.get("x-queue-wait-seconds")
                return {
                    "request_number": index,
                    "status_code": response.status_code,
                    "elapsed_seconds": elapsed,
                    "queue_wait_seconds": float(queue_header) if queue_header else None,
                    "completion_tokens": (
                        body.get("usage", {}).get("completion_tokens")
                        if isinstance(body, dict) else None
                    ),
                    "request_id": response.headers.get("x-request-id"),
                }
            except httpx.HTTPError as exc:
                return {
                    "request_number": index,
                    "status_code": None,
                    "elapsed_seconds": time.perf_counter() - started,
                    "queue_wait_seconds": None,
                    "completion_tokens": None,
                    "request_id": None,
                    "client_error": type(exc).__name__,
                }

    batch_started = time.perf_counter()
    requests = await asyncio.gather(*(one_request(i) for i in range(request_count)))
    batch_seconds = time.perf_counter() - batch_started
    successes = [item for item in requests if item["status_code"] == 200]
    latency = [item["elapsed_seconds"] for item in successes]
    queue_wait = [
        item["queue_wait_seconds"] for item in successes
        if item["queue_wait_seconds"] is not None
    ]
    output_tokens = sum(item["completion_tokens"] or 0 for item in successes)
    statuses = Counter(str(item["status_code"]) for item in requests)
    return {
        "concurrency": concurrency,
        "request_count": request_count,
        "batch_seconds": batch_seconds,
        "successful_requests": len(successes),
        "error_rate": (request_count - len(successes)) / request_count,
        "status_counts": dict(statuses),
        "requests_per_second": len(successes) / batch_seconds,
        "output_tokens_per_second": output_tokens / batch_seconds,
        "latency_p50_seconds": percentile(latency, 0.50),
        "latency_p95_seconds": percentile(latency, 0.95),
        "queue_wait_p50_seconds": percentile(queue_wait, 0.50),
        "queue_wait_p95_seconds": percentile(queue_wait, 0.95),
        "raw_requests": requests,
    }


async def benchmark(args: argparse.Namespace) -> None:
    settings = dotenv_values(ROOT / ".env")
    if args.target == "gateway":
        required = ("BACKEND_BASE_URL", "BACKEND_API_KEY", "PUBLIC_API_KEYS", "PUBLIC_MODEL_ID")
    else:
        required = ("BACKEND_BASE_URL", "BACKEND_API_KEY", "BACKEND_MODEL_ID")
    missing = [name for name in required if not settings.get(name)]
    if missing:
        raise SystemExit(f"Missing .env settings: {', '.join(missing)}")

    server = None
    log = None
    try:
        if args.target == "gateway":
            env = os.environ.copy()
            env.update({key: value for key, value in settings.items() if value is not None})
            env["RATE_LIMIT_PER_MINUTE"] = "1000"  # Isolate admission from rate limiting.
            if args.max_active is not None:
                env["MAX_ACTIVE_REQUESTS"] = str(args.max_active)
            if args.max_pending is not None:
                env["MAX_PENDING_REQUESTS"] = str(args.max_pending)
            log = Path("/tmp/openinfer_benchmark_gateway.log").open("w")
            server = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "gateway.app.main:app", "--host", "127.0.0.1",
                 "--port", str(GATEWAY_PORT)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            )
            await wait_for_gateway(server)
            base_url = f"http://127.0.0.1:{GATEWAY_PORT}/v1"
            key = settings["PUBLIC_API_KEYS"].split(",")[0].strip()
            model = settings["PUBLIC_MODEL_ID"]
        else:
            base_url = settings["BACKEND_BASE_URL"].rstrip("/")
            key = settings["BACKEND_API_KEY"]
            model = settings["BACKEND_MODEL_ID"]

        timeout = httpx.Timeout(130.0, connect=15.0)
        limits = httpx.Limits(max_connections=max(args.levels) + 4)
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            # Warm-up is deliberately excluded from measured batches.
            for warmup_index in range(2):
                warmup = await run_level(
                    client, f"{base_url}/chat/completions", key, model, 1, 1, args.max_tokens
                )
                if warmup["successful_requests"] != 1:
                    raise RuntimeError(f"Warm-up request {warmup_index + 1} failed: {warmup['status_counts']}")

            levels = []
            for concurrency in args.levels:
                level = await run_level(
                    client, f"{base_url}/chat/completions", key, model,
                    concurrency, args.requests_per_level, args.max_tokens,
                )
                levels.append(level)
                print(
                    f"concurrency={concurrency}: {level['successful_requests']}/"
                    f"{args.requests_per_level} succeeded, "
                    f"{level['requests_per_second']:.2f} requests/s, "
                    f"p95={level['latency_p95_seconds']}s, statuses={level['status_counts']}",
                    flush=True,
                )

            gateway_metrics = None
            if args.target == "gateway":
                metrics_response = await client.get(
                    f"http://127.0.0.1:{GATEWAY_PORT}/metrics"
                )
                metrics_response.raise_for_status()
                gateway_metrics = metrics_response.json()

        record = {
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "target": args.target,
            "model": model,
            "prompt_template": PROMPT_TEMPLATE,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "warmup_requests": 2,
            "requests_per_level": args.requests_per_level,
            "gateway_settings": (
                {
                    "max_active_requests": int(env.get("MAX_ACTIVE_REQUESTS", "2")),
                    "max_pending_requests": int(env.get("MAX_PENDING_REQUESTS", "8")),
                    "rate_limit_per_minute": 1000,
                }
                if args.target == "gateway" else None
            ),
            "gateway_metrics_after_run": gateway_metrics,
            "levels": levels,
        }
        suffix = f"_active{args.max_active}" if args.target == "gateway" and args.max_active else ""
        output = ROOT / "results" / f"benchmark_{args.target}{suffix}_remote.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print("Saved:", output)
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
        if log is not None:
            log.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("gateway", "bridge"), default="gateway")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--requests-per-level", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-active", type=int, default=None,
                        help="Override the gateway's active request cap for this run")
    parser.add_argument("--max-pending", type=int, default=None,
                        help="Override the gateway's pending request cap for this run")
    args = parser.parse_args()
    if (
        args.requests_per_level < max(args.levels)
        or args.max_tokens < 1
        or any(level < 1 for level in args.levels)
        or (args.max_active is not None and args.max_active < 1)
        or (args.max_pending is not None and args.max_pending < 0)
    ):
        parser.error("requests-per-level must cover the largest concurrency; caps must be nonnegative")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
