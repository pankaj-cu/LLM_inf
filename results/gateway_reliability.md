# Gateway reliability check

The local check used `scripts/check_gateway_reliability.py` with a temporary fake vLLM-compatible backend and a temporary Uvicorn gateway, both bound to `127.0.0.1`. It did not use the Colab runtime, Cloudflare Tunnel, `.env`, or real credentials. The check configured a 0.4-second chat deadline and a 0.3-second readiness/model-listing deadline so failures could be observed quickly.

| Scenario | HTTP status | Observed elapsed | Result |
| --- | ---: | ---: | --- |
| Healthy readiness | 200 | 0.016 s | Ready |
| Invalid-key chat | 401 | 0.002 s | `invalid_api_key` |
| Healthy chat | 200 | 0.003 s | Response and request ID |
| Slow readiness | 503 | 0.300 s | Bounded `not_ready` response |
| Slow model listing | 504 | 0.301 s | `backend_timeout` |
| Slow chat | 504 | 0.401 s | `request_timeout`; matching request IDs in header and body |
| Unavailable readiness | 503 | 0.002 s | `not_ready` response |
| Unavailable model listing | 503 | 0.001 s | `backend_unavailable` |
| Unavailable chat | 503 | 0.002 s | `backend_error`; matching request IDs in header and body |

The gateway recorded three authenticated chat requests: one completed and two failed. The invalid-key request was rejected before admission and is not included in those counters. Active and pending counts returned to zero. Its nearest-rank p95 chat latency was about 0.400 seconds, capturing the timeout. The short sample is a correctness check for bounded failure behavior, not a latency benchmark.

Run the check from the repository root with dependencies installed:

```bash
./.venv/bin/python scripts/check_gateway_reliability.py
```

The script saves the credential-free raw record at [gateway_reliability.json](gateway_reliability.json). It asserts the expected statuses, deadlines, request IDs, and counters and returns a nonzero exit code on a failure. A real Colab or tunnel interruption may fail differently, so this result establishes local gateway behavior against controlled backend failures only.
