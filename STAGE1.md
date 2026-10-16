# Stage 1: inference baseline

This experiment loads `Qwen/Qwen2.5-1.5B-Instruct` with Transformers, generates from a fixed prompt, and writes model, environment, load-time, generation-time, token-count, throughput, and GPU-memory details to JSON. It uses greedy decoding so repeated runs are easier to compare. The Colab version is [notebooks/stage1_baseline.ipynb](notebooks/stage1_baseline.ipynb); the Python script is an alternate command-line version.

## Run on Colab

1. Open `notebooks/stage1_baseline.ipynb` in Colab. You can upload the file or open it from the repository in Colab.
2. Select **Runtime → Change runtime type → T4 GPU** (if available).
3. Run the notebook cells from top to bottom. The final cell creates and downloads `openinfer_baseline.json`.

The first run downloads the model and can take several minutes. Keep the resulting JSON with the project; do not compare measurements from different hardware as if they were equivalent. The script also runs on CPU, but CPU results are only a functional baseline.

To change the prompt or workload, edit `PROMPT`, `MAX_NEW_TOKENS`, `WARMUP_RUNS`, or `MEASURED_RUNS` in the notebook. Record those settings with any result you share. For command-line use, the equivalent options are available in `scripts/baseline_inference.py`.

## What the measurements mean

- `model_load_seconds` includes loading the tokenizer and model into memory.
- Each `generation_seconds` is a warmed-up, non-streaming generation duration; it is not time-to-first-token.
- `output_tokens_per_second` is generated token count divided by total generation duration.
- GPU memory is PyTorch's peak allocated memory for the measured generations, after warm-up.

This is an initial baseline, not a full benchmark. We will add streaming TTFT/TPOT measurements and controlled concurrency later, once the serving API exists.
