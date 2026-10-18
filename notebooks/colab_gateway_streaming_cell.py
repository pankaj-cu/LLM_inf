"""Paste into Stage 3 Colab after vLLM starts; upload a fresh source bundle first.

In VS Code, run `python scripts/package_colab.py` and upload the resulting ZIP
when prompted. This cell restarts only the gateway, then checks a live streamed
Qwen response through the gateway on Colab localhost. It saves no credentials.
"""

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import httpx
import torch
import vllm
from google.colab import files
from openai import OpenAI


assert all(name in globals() for name in (
    "MODEL_ID", "PUBLIC_MODEL_ID", "VLLM_PORT", "GATEWAY_PORT", "vllm_key", "public_key"
)), "Run the Stage 3 vLLM setup cells first in this Colab session."

models = httpx.get(
    f"http://127.0.0.1:{VLLM_PORT}/v1/models",
    headers={"Authorization": f"Bearer {vllm_key}"},
    timeout=10,
)
models.raise_for_status()
assert MODEL_ID in {item["id"] for item in models.json()["data"]}, "vLLM model is unavailable"

uploaded = files.upload()
if len(uploaded) != 1:
    raise ValueError("Upload exactly one freshly generated openinfer_stage3_bundle.zip")
bundle_name, bundle_bytes = next(iter(uploaded.items()))
project_root = Path("/content/openinfer")
project_root.mkdir(parents=True, exist_ok=True)

if "gateway_process" in globals() and gateway_process.poll() is None:
    gateway_process.terminate()
    try:
        gateway_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        gateway_process.kill()
        gateway_process.wait(timeout=5)
if "gateway_log_handle" in globals():
    gateway_log_handle.close()

with ZipFile(BytesIO(bundle_bytes)) as bundle:
    expected = {"requirements.txt", "gateway/app/main.py", "gateway/app/backends.py"}
    if not expected.issubset(set(bundle.namelist())):
        raise ValueError("Upload the current OpenInfer source bundle")
    bundle.extractall(project_root)
os.chdir(project_root)

gateway_env = os.environ.copy()
gateway_env.update({
    "BACKEND_MODE": "vllm",
    "BACKEND_BASE_URL": f"http://127.0.0.1:{VLLM_PORT}/v1",
    "BACKEND_API_KEY": vllm_key,
    "PUBLIC_API_KEYS": public_key,
    "PUBLIC_MODEL_ID": PUBLIC_MODEL_ID,
    "BACKEND_MODEL_ID": MODEL_ID,
})
gateway_log = project_root / "gateway.log"
gateway_log_handle = gateway_log.open("w")
gateway_process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "gateway.app.main:app",
     "--host", "127.0.0.1", "--port", str(GATEWAY_PORT)],
    stdout=gateway_log_handle, stderr=subprocess.STDOUT,
    env=gateway_env, cwd=project_root,
)
for attempt in range(30):
    if gateway_process.poll() is not None:
        raise RuntimeError("Gateway exited early. Recent log:\n" + gateway_log.read_text()[-4000:])
    try:
        ready = httpx.get(f"http://127.0.0.1:{GATEWAY_PORT}/health/ready", timeout=5)
        if ready.status_code == 200:
            break
    except httpx.HTTPError:
        pass
    time.sleep(2)
else:
    raise TimeoutError("Gateway did not become ready. Recent log:\n" + gateway_log.read_text()[-4000:])

client = OpenAI(
    api_key=public_key,
    base_url=f"http://127.0.0.1:{GATEWAY_PORT}/v1",
    timeout=120,
    max_retries=0,
)
started = time.perf_counter()
first_text_at = None
text_parts = []
chunk_count = 0
finish_reason = None
stream = client.chat.completions.create(
    model=PUBLIC_MODEL_ID,
    messages=[{"role": "user", "content": "Explain the role of a KV cache in one sentence."}],
    max_tokens=32,
    temperature=0,
    stream=True,
)
try:
    for chunk in stream:
        chunk_count += 1
        assert chunk.model == PUBLIC_MODEL_ID, "Gateway did not publish the public model ID"
        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                if first_text_at is None:
                    first_text_at = time.perf_counter()
                text_parts.append(delta)
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
finally:
    stream.close()
elapsed = time.perf_counter() - started
assert first_text_at is not None and text_parts, "No streamed text arrived"
assert finish_reason is not None, "The stream ended without a finish reason"

record = {
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "path": "Colab Python SDK -> Colab localhost OpenInfer gateway -> Colab localhost vLLM",
    "model_id": MODEL_ID,
    "public_model_id": PUBLIC_MODEL_ID,
    "environment": {
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "vllm": vllm.__version__,
    },
    "request": {"max_tokens": 32, "temperature": 0, "stream": True},
    "result": {
        "chunk_count": chunk_count,
        "ttft_seconds": first_text_at - started,
        "total_seconds": elapsed,
        "finish_reason": finish_reason,
        "text": "".join(text_parts),
    },
}
output = Path("/content/openinfer_gateway_streaming.json")
output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print("Gateway stream succeeded:", chunk_count, "chunks in", round(elapsed, 3), "seconds")
print("Saved", output)
files.download(str(output))
