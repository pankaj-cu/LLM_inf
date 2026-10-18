# Stage 3 concurrency experiment

Recorded 2026-10-17 UTC. The Colab integration record identifies a Tesla T4, vLLM 0.31.0, and `Qwen/Qwen2.5-1.5B-Instruct`. The serving model stayed loaded during the remote runs and the later direct Colab run.

## Method

- Clients: `scripts/benchmark_gateway.py` from Codespaces for remote runs, and the Colab-local `notebooks/colab_direct_concurrency_cell.py` for the direct run. Both used HTTPX and non-streaming OpenAI-compatible chat requests.
- Paths: `direct` = Colab Python client → Colab localhost vLLM; `bridge` = Codespaces → Cloudflare Quick Tunnel → authenticated Colab bridge → vLLM; `gateway` adds the Codespaces FastAPI gateway before the tunnel.
- Same prompt template, temperature 0, `max_tokens=32`, two excluded warm-up requests, then 16 requests at each concurrency level (1, 4, 8, 16).
- Gateway rate limit temporarily set to 1000 requests/minute in the benchmark process; `.env` was not changed.
- Throughput is successful requests or output tokens divided by elapsed batch time. p95 latency is the nearest-rank p95 of successful requests. Queue wait comes from the gateway response header. Failed requests are included in the request and status counts but excluded from latency percentiles.
- Runs were sequential: gateway with active cap 2, bridge, gateway with active cap 16, then direct Colab after the streaming experiment. The default gateway pending cap was 8 in both gateway runs.

| Path and active cap | Concurrency | Success | Requests/s | Output tokens/s | p95 latency (s) | p95 queue wait (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct Colab | 1 | 16/16 | 2.08 | 66.70 | 0.50 | — |
| Direct Colab | 4 | 16/16 | 7.82 | 250.10 | 0.51 | — |
| Direct Colab | 8 | 16/16 | 14.67 | 469.39 | 0.55 | — |
| Direct Colab | 16 | 16/16 | 25.64 | 820.63 | 0.62 | — |
| Gateway, 2 | 1 | 16/16 | 1.32 | 42.13 | 1.24 | 0.00 |
| Gateway, 2 | 4 | 16/16 | 2.49 | 79.55 | 1.94 | 1.18 |
| Gateway, 2 | 8 | 16/16 | 2.62 | 83.90 | 3.05 | 2.29 |
| Gateway, 2 | 16 | 10/16 | 2.64 | 84.50 | 3.78 | 2.99 |
| Bridge | 1 | 16/16 | 1.31 | 41.88 | 1.21 | — |
| Bridge | 4 | 16/16 | 4.53 | 144.90 | 1.27 | — |
| Bridge | 8 | 16/16 | 7.54 | 241.29 | 1.31 | — |
| Bridge | 16 | 16/16 | 11.92 | 381.56 | 1.34 | — |
| Gateway, 16 | 1 | 16/16 | 1.36 | 43.64 | 0.78 | 0.00 |
| Gateway, 16 | 4 | 16/16 | 4.47 | 143.12 | 1.30 | 0.00 |
| Gateway, 16 | 8 | 16/16 | 8.12 | 259.77 | 1.20 | 0.00 |
| Gateway, 16 | 16 | 16/16 | 11.58 | 370.50 | 1.38 | 0.00 |

The cap-2 gateway accepted at most two active requests and eight queued requests. At concurrency 16, six requests received HTTP 503 (`overloaded`). Its successful requests reached 2.99 seconds p95 queue wait. With the cap raised to 16, the gateway observed 16 peak active requests and no failures; throughput rose with concurrency and was close to the bridge run. This supports the conclusion that the cap-2 admission setting was the main bottleneck in this experiment. The cap-16 gateway run had lower throughput than the bridge at concurrency 16 by about 3%, but sequential runs do not establish a stable overhead estimate.

The direct Colab path scaled from 2.08 to 25.64 successful requests/s between concurrency 1 and 16, while p95 latency rose from 0.50 to 0.62 seconds. The bridge path reached 11.92 requests/s and 1.34 seconds p95 at concurrency 16. This comparison shows a large end-to-end difference between the local and remote paths under this short workload. It does not assign the difference to Cloudflare alone: the bridge, internet transit, client location, and changing Colab conditions also contribute. The direct run's before/after whole-GPU memory.used was 11,681 MiB at every level; it cannot show peak or per-request KV-cache memory.

## Limits

These are short, 16-request samples from one temporary Colab session. The output limit is fixed at 32 tokens; prompt lengths, output lengths, GPU load, and network conditions were not varied within this comparison. The bridge path still includes the tunnel and bridge. Non-streaming responses do not provide TTFT or TPOT; those were measured separately with streaming on Colab localhost. We did not record a model revision, vLLM scheduler traces, or GPU memory peaks during load. The Stage 3 notebook started vLLM with `--dtype half`, but the benchmark records do not independently verify precision. The results show serving behavior under this configuration; they do not directly prove when continuous batching formed internally.

Raw data: [direct Colab](openinfer_direct_concurrency.json), [gateway cap 2](benchmark_gateway_remote.json), [bridge](benchmark_bridge_remote.json), [gateway cap 16](benchmark_gateway_active16_remote.json). The [Colab integration record](../openinfer_stage3_integration.json) has GPU and library details, and the [streaming report](stream_metrics_stage3.md) has TTFT/TPOT measurements.

To repeat while the temporary tunnel is live:

```bash
./.venv/bin/python scripts/benchmark_gateway.py --target gateway
./.venv/bin/python scripts/benchmark_gateway.py --target bridge
./.venv/bin/python scripts/benchmark_gateway.py --target gateway --max-active 16
```
