"""Paste this entire file into one cell after Stage 3 starts vLLM in Colab.

It uses MODEL_ID, VLLM_PORT, and vllm_key from the Stage 3 notebook.
The saved result contains prompts and timings, never API keys.
"""

import json
import math
import platform
import random
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import vllm
from google.colab import files
from openai import OpenAI


assert all(name in globals() for name in ("MODEL_ID", "VLLM_PORT", "vllm_key")), (
    "Run the Stage 3 vLLM setup cells first, in this Colab session."
)
metrics_client = OpenAI(
    api_key=vllm_key,
    base_url=f"http://127.0.0.1:{VLLM_PORT}/v1",
    timeout=120,
    max_retries=0,
)
assert any(item.id == MODEL_ID for item in metrics_client.models.list().data), (
    "vLLM is not ready. Restart it using the Stage 3 notebook."
)


def gpu_memory_mib():
    line = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()[0]
    used, total = line.split(",")[:2]
    return {"used_mib": int(used.strip()), "total_mib": int(total.strip())}


def make_prompt(bucket, trial):
    opener = (
        f"Trial {trial}. Explain how a transformer uses a KV cache during "
        "autoregressive inference. Give a detailed technical explanation."
    )
    if bucket == "short":
        return opener
    words = ["attention", "prefill", "decode", "memory", "tokens", "layers", "latency", "cache"]
    rng = random.Random(20261017 + trial)
    context = " ".join(rng.choices(words, k=450))
    return f"Trial {trial}. Context to read: {context}\n{opener}"


def one_stream(bucket, output_limit, trial):
    prompt = make_prompt(bucket, trial)
    memory_before = gpu_memory_mib()
    started = time.perf_counter()
    first_text_at = None
    last_text_at = None
    content_chunks = 0
    usage = None
    finish_reason = None
    stream = metrics_client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=output_limit,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"ignore_eos": True},
    )
    try:
        for chunk in stream:
            arrived = time.perf_counter()
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.choices:
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                if choice.delta.content:
                    if first_text_at is None:
                        first_text_at = arrived
                    last_text_at = arrived
                    content_chunks += 1
    finally:
        stream.close()
    ended = time.perf_counter()
    memory_after = gpu_memory_mib()
    completion_tokens = usage.completion_tokens if usage else None
    ttft = first_text_at - started if first_text_at is not None else None
    tpot = (
        (last_text_at - first_text_at) / (completion_tokens - 1)
        if first_text_at is not None and last_text_at is not None
        and completion_tokens is not None and completion_tokens > 1 and content_chunks > 1
        else None
    )
    return {
        "bucket": bucket,
        "trial": trial,
        "prompt": prompt,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "output_limit": output_limit,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "content_chunks": content_chunks,
        "ttft_seconds": ttft,
        "tpot_estimate_seconds": tpot,
        "elapsed_seconds": ended - started,
        "output_tokens_per_second": completion_tokens / (ended - started) if completion_tokens else None,
        "gpu_memory_before": memory_before,
        "gpu_memory_after": memory_after,
    }


def nearest_rank_p95(values):
    values = sorted(value for value in values if value is not None)
    return values[max(0, math.ceil(0.95 * len(values)) - 1)] if values else None


# Warm-up calls are excluded. A different prompt is used for each measured trial.
for warmup_trial in (-2, -1):
    one_stream("short", 32, warmup_trial)

raw = []
for bucket in ("short", "long"):
    for output_limit in (32, 128):
        for trial in range(5):
            # Keep prompts distinct across conditions to reduce prefix-cache reuse.
            prompt_trial = trial + (100 if bucket == "long" else 0) + (1000 if output_limit == 128 else 0)
            sample = one_stream(bucket, output_limit, prompt_trial)
            raw.append(sample)
            print(
                bucket, output_limit, f"trial={trial + 1}/5",
                f"input={sample['prompt_tokens']}",
                f"output={sample['completion_tokens']}",
                f"TTFT={sample['ttft_seconds']:.3f}s" if sample["ttft_seconds"] is not None else "TTFT unavailable",
                f"elapsed={sample['elapsed_seconds']:.3f}s",
                flush=True,
            )

summary = []
for bucket in ("short", "long"):
    for output_limit in (32, 128):
        group = [item for item in raw if item["bucket"] == bucket and item["output_limit"] == output_limit]
        metric = lambda key: [item[key] for item in group if item[key] is not None]
        summary.append({
            "bucket": bucket,
            "output_limit": output_limit,
            "samples": len(group),
            "median_prompt_tokens": statistics.median(metric("prompt_tokens")),
            "median_completion_tokens": statistics.median(metric("completion_tokens")),
            "median_ttft_seconds": statistics.median(metric("ttft_seconds")) if metric("ttft_seconds") else None,
            "p95_ttft_seconds": nearest_rank_p95(metric("ttft_seconds")),
            "median_tpot_estimate_seconds": (
                statistics.median(metric("tpot_estimate_seconds"))
                if metric("tpot_estimate_seconds") else None
            ),
            "median_elapsed_seconds": statistics.median(metric("elapsed_seconds")),
            "p95_elapsed_seconds": nearest_rank_p95(metric("elapsed_seconds")),
            "median_output_tokens_per_second": (
                statistics.median(metric("output_tokens_per_second"))
                if metric("output_tokens_per_second") else None
            ),
        })

record = {
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "model_id": MODEL_ID,
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "vllm": vllm.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "vllm_dtype_from_stage3_command": "half",
        "vllm_max_model_len_from_stage3_command": 2048,
    },
    "method": {
        "path": "Colab Python client -> Colab localhost vLLM",
        "stream": True,
        "temperature": 0,
        "ignore_eos": True,
        "warmup_requests": 2,
        "repeats_per_condition": 5,
        "concurrency": 1,
        "ttft_definition": "client request start to first nonempty streamed text delta",
        "tpot_estimate_definition": "first to last text delta divided by completion_tokens minus one; chunking can make this approximate",
        "gpu_memory_definition": "nvidia-smi whole-GPU memory.used before and after each request, not peak or per-request KV memory",
    },
    "summary": summary,
    "raw": raw,
}
output_path = Path("/content/openinfer_stream_metrics.json")
output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print("Saved", output_path)
for row in summary:
    print(row)
files.download(str(output_path))
