# Stage 3: run Qwen through vLLM and the gateway

The first integration run is inside one Colab T4 session. It exercises real GPU inference through vLLM and the repository gateway. A remote tunnel experiment is available only where the runtime terms permit it.

## Prepare in VS Code

From the repository root, create an upload bundle of the current gateway source:

```bash
python scripts/package_colab.py
```

This produces `openinfer_stage3_bundle.zip`. It includes uncommitted gateway code, so you do not need to push the repository before the Colab run. Do not put `.env` or credentials in the bundle.

## Run in Colab

1. Upload [the Stage 3 notebook](notebooks/stage3_vllm_colab.ipynb) to Colab.
2. Select a T4 GPU runtime. Use a fresh runtime if a model is still occupying GPU memory from another notebook.
3. Run its cells in order. Upload `openinfer_stage3_bundle.zip` when prompted.
4. Wait for the vLLM server to load `Qwen/Qwen2.5-1.5B-Instruct`. The notebook checks `/v1/models` before sending a prompt.
5. Run the direct and gateway OpenAI SDK calls, then download `openinfer_stage3_integration.json`.
6. Copy that JSON into this repository so we can inspect the measured result and any errors.

The notebook writes `vllm.log` and `gateway.log` in `/content/openinfer`. If a process exits or readiness times out, inspect the last lines of its log in the Colab file browser. vLLM installation can change PyTorch dependencies; if Colab reports that a runtime restart is needed, restart it and continue from the environment check cell.

The JSON records one direct vLLM call and one call through the gateway. It is an integration record, not a concurrency benchmark. Model loading, TTFT, TPOT, and end-to-end comparisons will receive separate measurements later.

The saved [Colab integration record](openinfer_stage3_integration.json) confirms a Tesla T4, vLLM 0.31.0, and identical 33-token direct and gateway responses. The two requests ran in sequence; warm-up makes their elapsed times unsuitable for estimating gateway overhead.

## Remote integration result

The Codespaces gateway successfully reached the Colab vLLM backend through the authenticated bridge and Cloudflare Tunnel on 2026-10-17. The OpenAI SDK listed `openinfer-qwen` and received a 10-token response. The single request took 0.46 seconds end to end. The raw, credential-free record is at [results/stage3_remote_integration.json](results/stage3_remote_integration.json). This proves connectivity for that temporary session; it does not establish steady-state performance or tunnel availability after the session ends.

## Optional remote connection

Google Colab's [usage guidance](https://research.google.com/colaboratory/faq.html) restricts unrelated web-service offerings on managed runtimes and warns that free runtimes can terminate when used primarily through a remote UI. Use the tunnel cells only if your actual runtime use is permitted under the current terms. The Colab-local path above provides a GPU integration route regardless of tunnel availability.

When remote access is permitted, the notebook's optional cells start an authenticated bridge on Colab localhost port `8002`, then start a temporary Cloudflare Quick Tunnel to that bridge. They do not expose vLLM directly. This matters because [vLLM's `--api-key` protection does not cover every server endpoint](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/). The bridge accepts only non-streaming `/v1/models` and `/v1/chat/completions` calls and forwards them to vLLM with a separate key.

In Codespaces or VS Code, set the gateway environment before starting it:

```bash
export BACKEND_MODE=vllm
export BACKEND_BASE_URL='https://YOUR-TEMPORARY-HOST.trycloudflare.com/v1'
export BACKEND_API_KEY='THE-COLAB-BRIDGE-KEY'
export BACKEND_MODEL_ID='Qwen/Qwen2.5-1.5B-Instruct'
export PUBLIC_MODEL_ID=openinfer-qwen
export PUBLIC_API_KEYS='A-DIFFERENT-CLIENT-KEY'
uvicorn gateway.app.main:app --host 127.0.0.1 --port 8000
```

To retrieve the generated bridge key, run `print(bridge_key)` in a temporary private Colab cell, copy it to `BACKEND_API_KEY`, then clear that cell's output. Do not commit the key or a notebook with that output. Call `GET /health/ready` on the Codespaces gateway, then use the OpenAI SDK example in [README.md](README.md) with your public client key. An expired tunnel or stopped Colab session should make readiness return `503` rather than hang.

Cloudflare [Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/) use a new hostname each time and do not support Server-Sent Events. Stage 3 therefore uses non-streaming requests. Keep the URL in `BACKEND_BASE_URL`; never hard-code it in source.

## Concurrent workload

While the Colab bridge and tunnel remain live, run the same non-streaming workload through the Codespaces gateway and directly through the tunnel bridge:

```bash
./.venv/bin/python scripts/benchmark_gateway.py --target gateway
./.venv/bin/python scripts/benchmark_gateway.py --target bridge
```

The script reads the ignored `.env` file, uses 1/4/8/16 concurrent clients, and saves raw timings and summaries in `results/`. The gateway run temporarily raises its per-minute rate limit to 1000 so the admission limit is measured separately; it does not change `.env`. The bridge run bypasses the Codespaces gateway but retains tunnel and bridge overhead. Compare these runs cautiously because they occur at different times, and Colab GPU load can change between them. A 503 at high concurrency may show the configured gateway admission cap working. Neither run measures streaming TTFT or TPOT.

## Next Colab experiment: streaming timing

Keep the Stage 3 Colab runtime and vLLM server running. In that **same Colab notebook tab**, add a code cell and paste the entire contents of [the streaming metrics cell](notebooks/colab_stream_metrics_cell.py), then run it. The updated repository notebook also has this as its last cell, but edits here do not automatically appear in an already-open Colab tab. If the runtime has restarted, rerun Stage 3 through the vLLM startup and readiness cells first. The bridge and tunnel cells are not needed for this experiment.

The cell makes two warm-up requests, then 20 measured streaming requests: five repeats for each combination of short/long prompt and 32/128 output-token limit. It prints progress, saves `/content/openinfer_stream_metrics.json`, and downloads that file. Copy the downloaded JSON into this repository and we can interpret the results together.

The completed run is documented in the [streaming metrics report](results/stream_metrics_stage3.md), with the [raw JSON](results/openinfer_stream_metrics.json) alongside it.

TTFT is measured from the Colab client's request start to its first nonempty text chunk. The TPOT estimate divides the time between first and last text chunks by the number of subsequent generated tokens; HTTP chunk grouping makes this approximate. `nvidia-smi` memory is whole-GPU memory before/after each request, not peak memory or the model's KV cache size. vLLM is set to ignore EOS for this controlled output-length experiment, so generated text may be unnatural. The experiment uses localhost and measures neither remote user-visible TTFT nor pure GPU prefill time. Model revision and prefix-cache state are not pinned, so treat short/long differences as an initial observation.

## Next Colab experiment: direct vLLM concurrency

The remote benchmark's `bridge` path still included the tunnel. For a closer direct-backend comparison, paste [the direct concurrency cell](notebooks/colab_direct_concurrency_cell.py) into another code cell in the same running Colab notebook and run it. It sends the same non-streaming prompt template as the remote benchmark: two warm-up calls, then 16 requests at each concurrency level (1, 4, 8, 16), temperature zero, 32-token output limit. It downloads `/content/openinfer_direct_concurrency.json`; copy that file into `results/` for analysis. The cell is also appended to the repository notebook for future runs.

The completed direct run is in [results/openinfer_direct_concurrency.json](results/openinfer_direct_concurrency.json). The [concurrency report](results/benchmark_stage3.md) compares it with the remote bridge and gateway runs.
