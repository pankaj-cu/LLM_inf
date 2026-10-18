# OpenInfer demo

The default demo starts a temporary gateway with the deterministic mock backend. It needs no GPU, Colab session, tunnel, `.env`, or manually chosen API key. This demonstrates the API and gateway controls; the mock response is not LLM inference.

## One-command local demo

From the repository root, after installing `requirements.txt`:

```bash
./.venv/bin/python scripts/demo_gateway.py --output results/demo_local_mock.json
```

The script binds only to `127.0.0.1`, creates a temporary public key, waits for readiness, lists the public model through the OpenAI SDK, sends one chat request, checks that a wrong key is rejected, prints metrics, and stops the gateway. The optional JSON output contains no credentials.

The completed local run is saved in [results/demo_local_mock.json](results/demo_local_mock.json). It returned `openinfer-qwen`, a mock chat response, a request ID, and HTTP 401 for a wrong key.

## Docker demo

Build the gateway image:

```bash
docker build -t openinfer-gateway .
```

Start it with an explicit public key, bound to your machine's localhost:

```bash
docker run --rm --name openinfer-gateway -p 127.0.0.1:8000:8000 \
  -e BACKEND_MODE=mock -e PUBLIC_API_KEYS=local-demo-key openinfer-gateway
```

In another terminal, call the container through the same demo client:

```bash
OPENINFER_DEMO_API_KEY=local-demo-key ./.venv/bin/python scripts/demo_gateway.py \
  --base-url http://127.0.0.1:8000/v1
```

The image contains the gateway and Python dependencies only. `.dockerignore` excludes `.env`, notebooks, results, and generated bundles; the Dockerfile copies only `requirements.txt` and `gateway/`. The image refuses to start without `PUBLIC_API_KEYS` because its container default is empty.

The Docker image was built and exercised with the demo client. Its [saved result](results/demo_docker_mock.json) records a successful model list and chat call. The container was stopped after the check.

## Live vLLM demo

When a permitted vLLM backend and tunnel are actually running, configure the gateway with the ignored `.env` described in [STAGE3.md](STAGE3.md). A Docker run can pass those settings at runtime without baking them into the image:

```bash
docker run --rm --name openinfer-gateway -p 127.0.0.1:8000:8000 \
  --env-file .env openinfer-gateway
```

Set `OPENINFER_DEMO_API_KEY` to one of the *public* keys in `PUBLIC_API_KEYS`, then call:

```bash
./.venv/bin/python scripts/demo_gateway.py --base-url http://127.0.0.1:8000/v1
```

The demo script never reads the backend key, and it does not print or save the public key. The runtime `.env` must use a reachable backend URL from inside the container; a temporary tunnel URL expires when its Colab session ends. If the backend is unavailable, the readiness check reports that condition instead of presenting a mock response as GPU inference.

For a Docker-free live demo, start the gateway with `./.venv/bin/uvicorn gateway.app.main:app --env-file .env --host 127.0.0.1 --port 8000`, then run the same demo client command above.

The dependency versions in `requirements.txt` are ranges, so a future rebuild may select newer versions. The saved benchmark and integration records are the evidence for the current project behavior; the mock demo is for showing the API flow.
