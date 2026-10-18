"""Paste into a new cell in the Stage 3 Colab notebook after vLLM is running.

This measures the direct localhost vLLM path with the same workload used by
scripts/benchmark_gateway.py. No credentials are written to the result.
"""

import asyncio
import json
import math
import platform
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx
import torch
import vllm
from google.colab import files


assert all(name in globals() for name in ("MODEL_ID", "VLLM_PORT", "vllm_key")), (
    "Run the Stage 3 vLLM setup cells first, in this Colab session."
)
DIRECT_LEVELS = (1, 4, 8, 16)
DIRECT_REQUESTS_PER_LEVEL = 16
DIRECT_MAX_TOKENS = 32
DIRECT_PROMPT_TEMPLATE = (
    "Explain one practical use of a transformer KV cache in one short sentence. "
    "Request number {number}."
)


def direct_gpu_memory_mib():
    line = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()[0]
    used, total = line.split(",")[:2]
    return {"used_mib": int(used.strip()), "total_mib": int(total.strip())}


def direct_p95(values):
    values = sorted(values)
    return values[max(0, math.ceil(0.95 * len(values)) - 1)] if values else None


async def direct_level(client, concurrency, request_count):
    gate = asyncio.Semaphore(concurrency)

    async def one_request(number):
        payload = {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": DIRECT_PROMPT_TEMPLATE.format(number=number)}],
            "max_tokens": DIRECT_MAX_TOKENS,
            "temperature": 0,
            "stream": False,
        }
        async with gate:
            started = time.perf_counter()
            try:
                response = await client.post(
                    f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions", json=payload
                )
                elapsed = time.perf_counter() - started
                try:
                    body = response.json()
                except ValueError:
                    body = {}
                usage = body.get("usage") if isinstance(body, dict) else None
                return {
                    "request_number": number,
                    "status_code": response.status_code,
                    "elapsed_seconds": elapsed,
                    "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, dict) else None,
                    "completion_tokens": usage.get("completion_tokens") if isinstance(usage, dict) else None,
                }
            except httpx.HTTPError as exc:
                return {
                    "request_number": number,
                    "status_code": None,
                    "elapsed_seconds": time.perf_counter() - started,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "client_error": type(exc).__name__,
                }

    memory_before = direct_gpu_memory_mib()
    batch_started = time.perf_counter()
    requests = await asyncio.gather(*(one_request(i) for i in range(request_count)))
    batch_seconds = time.perf_counter() - batch_started
    memory_after = direct_gpu_memory_mib()
    success = [item for item in requests if item["status_code"] == 200]
    latencies = [item["elapsed_seconds"] for item in success]
    output_tokens = sum(item["completion_tokens"] or 0 for item in success)
    return {
        "concurrency": concurrency,
        "request_count": request_count,
        "successful_requests": len(success),
        "status_counts": dict(Counter(str(item["status_code"]) for item in requests)),
        "batch_seconds": batch_seconds,
        "requests_per_second": len(success) / batch_seconds,
        "output_tokens_per_second": output_tokens / batch_seconds,
        "latency_p50_seconds": (
            sorted(latencies)[max(0, math.ceil(0.5 * len(latencies)) - 1)]
            if latencies else None
        ),
        "latency_p95_seconds": direct_p95(latencies),
        "gpu_memory_before": memory_before,
        "gpu_memory_after": memory_after,
        "raw_requests": requests,
    }


async def direct_benchmark():
    timeout = httpx.Timeout(130.0, connect=10.0)
    limits = httpx.Limits(max_connections=20)
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {vllm_key}"},
        timeout=timeout,
        limits=limits,
    ) as client:
        models = await client.get(f"http://127.0.0.1:{VLLM_PORT}/v1/models")
        models.raise_for_status()
        assert MODEL_ID in {item["id"] for item in models.json()["data"]}, "vLLM model unavailable"
        for warmup in range(2):
            result = await direct_level(client, 1, 1)
            if result["successful_requests"] != 1:
                raise RuntimeError(f"Warm-up {warmup + 1} failed: {result['status_counts']}")
        levels = []
        for concurrency in DIRECT_LEVELS:
            result = await direct_level(client, concurrency, DIRECT_REQUESTS_PER_LEVEL)
            levels.append(result)
            print(
                f"concurrency={concurrency}: {result['successful_requests']}/16 succeeded, "
                f"{result['requests_per_second']:.2f} requests/s, "
                f"p95={result['latency_p95_seconds']}s, "
                f"statuses={result['status_counts']}",
                flush=True,
            )
    return levels


# Colab has an event loop already. Run the async workload in a separate thread.
with ThreadPoolExecutor(max_workers=1) as pool:
    direct_levels = pool.submit(lambda: asyncio.run(direct_benchmark())).result()

direct_record = {
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "model_id": MODEL_ID,
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "vllm": vllm.__version__,
        "gpu": torch.cuda.get_device_name(0),
    },
    "method": {
        "path": "Colab Python client -> Colab localhost vLLM",
        "prompt_template": DIRECT_PROMPT_TEMPLATE,
        "max_tokens": DIRECT_MAX_TOKENS,
        "temperature": 0,
        "stream": False,
        "warmup_requests": 2,
        "requests_per_level": DIRECT_REQUESTS_PER_LEVEL,
        "levels": list(DIRECT_LEVELS),
        "gpu_memory_definition": "nvidia-smi whole-GPU memory.used before and after each batch, not peak or per-request KV memory",
    },
    "levels": direct_levels,
}
direct_output = Path("/content/openinfer_direct_concurrency.json")
direct_output.write_text(json.dumps(direct_record, indent=2) + "\n", encoding="utf-8")
print("Saved", direct_output)
files.download(str(direct_output))
