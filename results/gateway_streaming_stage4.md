# Stage 4 gateway streaming check

The [Colab result](openinfer_gateway_streaming.json) was recorded on 2026-10-18 UTC with a Tesla T4, Python 3.13.15, vLLM 0.32.0, and `Qwen/Qwen2.5-1.5B-Instruct`. The OpenAI Python SDK called the OpenInfer gateway on Colab localhost with `stream=true`, `temperature=0`, and `max_tokens=32`; the gateway called vLLM on the same runtime.

| Observation | Result |
| --- | ---: |
| SDK chunks received | 32 |
| Time to first nonempty text delta | 47.2 ms |
| Total client time | 544.7 ms |
| Finish reason | `stop` |

The SDK received text chunks and a final finish reason under the public model ID `openinfer-qwen`. This confirms that the gateway streaming path worked for one real vLLM request in that Colab session. The local [streaming check](../scripts/check_gateway_streaming.py) separately covers backend errors, deadlines, and client disconnect cleanup without a GPU.

This is one request, so the timings are a functional record, not a latency benchmark or comparison with the earlier Stage 3 runs. The prompt asked for the role of a KV cache, but the generated answer describes a general key-value data store rather than the transformer's attention KV cache. The answer should not be presented as evidence of technical correctness. The JSON does not include raw SSE frames, individual chunk timestamps, token usage, or a model revision.
