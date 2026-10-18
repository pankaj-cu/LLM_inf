# Stage 4: streaming chat through the gateway

The gateway now accepts `stream=true` on `POST /v1/chat/completions`. It sends OpenAI-compatible Server-Sent Event (SSE) chunks, replaces vLLM's internal model ID with `openinfer-qwen`, and finishes with `data: [DONE]`. The mock backend emits deterministic chunks for local development. A streaming request keeps its admission slot until it completes or the client disconnects.

## Local check

With dependencies installed, run from the repository root:

```bash
./.venv/bin/python scripts/check_gateway_streaming.py
```

This check starts a temporary gateway on `127.0.0.1`, reads its mock stream through the OpenAI Python SDK, checks the vLLM adapter against a fake SSE backend, and simulates a client disconnect. It confirms that the backend stream closes and the gateway's active count returns to zero. It does not use Colab, a GPU, or a tunnel.

## Real vLLM check in Colab

1. In VS Code, run `python scripts/package_colab.py` to create a fresh ignored `openinfer_stage3_bundle.zip` containing the new gateway code.
2. Open [the Stage 3 notebook](notebooks/stage3_vllm_colab.ipynb) in a Colab T4 session. Run its setup cells until vLLM is ready. The Stage 4 cell starts the gateway if needed; the bridge and tunnel cells are unnecessary.
3. Run the notebook's **Stage 4: gateway streaming** cell. It asks for the fresh ZIP, restarts only the gateway, sends one `stream=True` call through the OpenAI SDK, and downloads `openinfer_gateway_streaming.json`.
4. Put the downloaded JSON in `results/`. The file contains timings and generated text, but no API keys.

The Stage 4 cell is also available as [a standalone Python cell](notebooks/colab_gateway_streaming_cell.py) to paste into an already open Colab notebook. The completed run is saved in [the raw JSON](results/openinfer_gateway_streaming.json) and described in [the result report](results/gateway_streaming_stage4.md). One stream proves functionality but is not a latency benchmark.

The optional Cloudflare Quick Tunnel bridge remains non-streaming. Use Colab localhost for the real gateway streaming check. If vLLM disconnects before any stream data arrives, the gateway returns an HTTP error. Once SSE headers have been sent, later backend failures appear as an SSE `error` event, because the HTTP status has already been sent.
