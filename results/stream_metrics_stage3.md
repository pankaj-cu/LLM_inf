# Stage 3 Colab streaming metrics

Recorded 2026-10-17 UTC on a Tesla T4 with `Qwen/Qwen2.5-1.5B-Instruct`, vLLM 0.31.0, and the Stage 3 `half`-precision server configuration. The client called vLLM on Colab localhost. Two warm-up requests were excluded, then five requests were measured for each prompt/output condition at concurrency 1. Temperature was zero; `ignore_eos=true` forced the 32- or 128-token output limit. Each request used a distinct prompt.

| Prompt | Median input tokens | Output tokens | Median TTFT | p95 TTFT | Median TPOT estimate | Median total latency | p95 total latency | Median effective output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short | 53 | 32 | 31.6 ms | 32.2 ms | 14.5 ms | 0.482 s | 0.483 s | 66.4 |
| Short | 56 | 128 | 31.8 ms | 32.7 ms | 14.7 ms | 1.894 s | 1.900 s | 67.6 |
| Long | 569 | 32 | 135.1 ms | 137.2 ms | 15.3 ms | 0.609 s | 0.612 s | 52.5 |
| Long | 579 | 128 | 136.5 ms | 139.2 ms | 15.4 ms | 2.093 s | 2.098 s | 61.2 |

The roughly 570-token prompts had about 104 ms more median TTFT than the roughly 55-token prompts at either output limit. That is consistent with a larger prefill workload. Raising the output limit from 32 to 128 tokens left TTFT nearly unchanged while increasing total latency by about 1.4–1.5 seconds. The client-observed TPOT estimate was roughly 14.5–15.4 ms/token across these conditions. These are observations from this workload, not a pure GPU prefill/decode breakdown.

All 20 responses reported the requested number of completion tokens and `finish_reason="length"`. Each output token arrived in its own nonempty text chunk in this run. The first long-prompt request changed whole-GPU `nvidia-smi` memory.used from 11,661 to 11,681 MiB; subsequent before/after readings stayed at 11,681 MiB. vLLM can reserve GPU memory in advance, and before/after whole-device readings do not isolate KV-cache use or peak memory.

TTFT here means client request start to the first nonempty streamed text delta. TPOT is estimated as time from first to last text delta divided by completion tokens minus one. HTTP delivery, text chunking, and any hidden/special tokens can affect this estimate. Effective output tokens/s divides completion tokens by total request time, including TTFT; it is not the same as decode-only throughput. p95 with only five samples is the slowest sample, so more repetitions would be needed for a stable tail-latency claim. Prompt tokens were recorded from vLLM usage; model revision, prefix-cache state, GPU utilization during generation, and GPU scheduling traces were not recorded. The artificial prompts and forced output length support controlled timing but do not represent a user workload.

Raw data and exact prompt text: [openinfer_stream_metrics.json](openinfer_stream_metrics.json). The experiment code is [colab_stream_metrics_cell.py](../notebooks/colab_stream_metrics_cell.py). Remote concurrency results are in the [Stage 3 concurrency report](benchmark_stage3.md); those used a different workload and should not be subtracted from these local timings as an estimate of network cost.
