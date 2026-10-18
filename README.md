# OpenInfer

OpenInfer is a learning project for an authenticated LLM serving gateway backed by vLLM. The intended inference path is FastAPI → vLLM → Qwen on a Colab T4 when a permitted connection is available. A Colab-local integration path is the fallback. The mock backend exists for gateway development and does not count as model inference.

For a quick API demonstration, run `./.venv/bin/python scripts/demo_gateway.py` after installing dependencies. It starts a temporary localhost gateway with the mock backend, calls it through the OpenAI SDK, and shuts it down. See [DEMO.md](DEMO.md) for Docker and live-backend instructions.

## Stage 1: GPU baseline

Run [the Stage 1 Colab notebook](notebooks/stage1_baseline.ipynb) and save its JSON output. The recorded initial run is in [`openinfer_baseline.json`](openinfer_baseline.json).

## Stage 2: run the gateway locally

Use Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start in mock mode to check the API without GPU access:

```bash
export BACKEND_MODE=mock
export PUBLIC_API_KEYS=dev-key-change-me
uvicorn gateway.app.main:app --host 127.0.0.1 --port 8000
```

In another terminal, check health and models:

```bash
curl http://127.0.0.1:8000/health
curl -H 'Authorization: Bearer dev-key-change-me' http://127.0.0.1:8000/v1/models
```

Call the OpenAI-compatible endpoint:

```python
from openai import OpenAI

client = OpenAI(api_key="dev-key-change-me", base_url="http://127.0.0.1:8000/v1")
result = client.chat.completions.create(
    model="openinfer-qwen",
    messages=[{"role": "user", "content": "Hello"}],
)
print(result.choices[0].message.content)
```

`GET /health/ready` checks whether the configured backend and model are available. `GET /metrics` reports request counts, active/pending work, and gateway latency percentiles.

`READINESS_TIMEOUT_SECONDS` defaults to 5 seconds and bounds readiness and model-listing checks. `REQUEST_TIMEOUT_SECONDS` defaults to 120 seconds and bounds chat requests, including queue wait. Error chat responses include an `X-Request-ID` header for tracing.

## Stage 3: real vLLM inference on Colab T4

Follow [the Stage 3 run guide](STAGE3.md) and open [the Colab integration notebook](notebooks/stage3_vllm_colab.ipynb). It runs Qwen with vLLM, calls the model directly and through the gateway, and downloads a JSON integration record. The optional tunnel cells use a restricted bridge if remote access is permitted for your runtime.

The remote path was exercised on 2026-10-17: Codespaces gateway → Cloudflare Tunnel → Colab bridge → vLLM returned a Qwen chat response through the OpenAI Python SDK. See the [raw integration record](results/stage3_remote_integration.json). Its single 10-token completion took 0.46 seconds end to end; this is a connectivity check, not a throughput or latency benchmark.

The [Colab integration record](openinfer_stage3_integration.json) shows a Tesla T4 running vLLM 0.31.0 with `Qwen/Qwen2.5-1.5B-Instruct`. Direct vLLM and Colab-local gateway calls each produced the same 33-token answer. They ran sequentially, so their 2.40-second and 0.54-second times cannot establish a gateway performance advantage.

The [Stage 3 concurrency report](results/benchmark_stage3.md) records 1/4/8/16-client runs. With the gateway's two-active-request cap, throughput leveled near 2.6 requests/s and six requests were rejected at concurrency 16. With a 16-active-request cap, all requests succeeded and throughput reached 11.58 requests/s, close to 11.92 requests/s through the bridge path. These are short remote samples; the report includes raw data and limitations.

A later direct Colab-local vLLM run with the same prompt and output limit reached 25.64 requests/s at concurrency 16, with 0.623-second p95 latency. The bridge path reached 11.92 requests/s and 1.339-second p95. Different client paths and run times prevent attributing that gap to the tunnel alone; see the [report](results/benchmark_stage3.md) and [raw direct result](results/openinfer_direct_concurrency.json).

The [Colab streaming report](results/stream_metrics_stage3.md) records five repeats for short/long prompts and 32/128-token outputs. Median client-observed TTFT rose from about 32 ms for 53–56 input tokens to about 135–136 ms for roughly 570 input tokens; estimated TPOT stayed near 15 ms/token. The report links the raw JSON and explains the measurement limits.

The [gateway reliability report](results/gateway_reliability.md) documents a local controlled backend shutdown and slow-response check. The gateway returned bounded 503/504 responses, kept request IDs on chat errors, and accounted for failed requests in `/metrics`.

## Connect the gateway to vLLM

When the vLLM server is reachable at a permitted URL, configure:

```bash
export BACKEND_MODE=vllm
export BACKEND_BASE_URL=http://127.0.0.1:8001/v1
export BACKEND_API_KEY='the-internal-vllm-key'
export BACKEND_MODEL_ID='Qwen/Qwen2.5-1.5B-Instruct'
export PUBLIC_MODEL_ID=openinfer-qwen
export PUBLIC_API_KEYS='a-separate-client-key'
```

Leave `BACKEND_API_KEY` empty only when the vLLM server was started without `--api-key`. The public client key and backend key serve different purposes; never use the backend key as the client key. vLLM's API key does not protect every route, so keep the backend private behind the gateway or a suitable access control layer.

The gateway currently supports non-streaming chat completions. It rejects `stream=true` until end-to-end SSE support is implemented and validated. Colab is an interactive compute service, not guaranteed hosting; check its current usage rules before exposing a Colab runtime through a tunnel. If a remote connection is not permitted or reliable, run the gateway and vLLM together in the Colab session for the integration demo.

## Current scope

- Implemented: mock and vLLM backends; `/health`, `/health/ready`, `/metrics`, `/v1/models`, and non-streaming `/v1/chat/completions`.
- Implemented: bearer API-key verification, per-key request rate limit, bounded admission, request deadline, request ID response header, and bounded latency sample storage.
- Verified: Colab-local direct and gateway calls and one remote OpenAI SDK request returned Qwen responses on a Tesla T4.
- Verified: 1/4/8/16 concurrency benchmarks for the bridge and gateway with two admission caps; raw timing records and a report are in `results/`.
- Verified: [Colab-local streaming TTFT and TPOT measurements](results/stream_metrics_stage3.md) with prompt/output token counts, total latency, and whole-GPU memory before/after each call.
- Verified: [direct vLLM concurrency experiment](results/openinfer_direct_concurrency.json) with the same prompts and 1/4/8/16 levels as the remote benchmark.
- Verified: [local gateway reliability check](results/gateway_reliability.md) for healthy, slow, and unavailable backend behavior.
- Verified: a [one-command local mock demo](results/demo_local_mock.json) and a [Docker gateway demo](results/demo_docker_mock.json); see [DEMO.md](DEMO.md) to repeat them.
- Later: cancellation verification, more detailed GPU metrics, and a dashboard.
