"""Run a small Hugging Face causal LM and save a reproducible baseline."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PROMPT = "Explain how a KV cache helps autoregressive text generation in three sentences."


def synchronize_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/baseline.json"))
    args = parser.parse_args()

    if args.max_new_tokens < 1 or args.runs < 1 or args.warmup_runs < 0:
        parser.error("max-new-tokens and runs must be positive; warmup-runs cannot be negative")

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    synchronize_if_cuda()
    model_load_seconds = time.perf_counter() - load_started

    messages = [{"role": "user", "content": args.prompt}]
    rendered_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(rendered_prompt, return_tensors="pt").to(device)
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    def generate_once() -> tuple[float, int, str]:
        synchronize_if_cuda()
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        synchronize_if_cuda()
        elapsed = time.perf_counter() - started
        generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
        generated_tokens = int(generated_ids.numel())
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return elapsed, generated_tokens, text

    for _ in range(args.warmup_runs):
        generate_once()

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    run_records = []
    for index in range(args.runs):
        elapsed, generated_tokens, generated_text = generate_once()
        run_records.append(
            {
                "run": index + 1,
                "generation_seconds": elapsed,
                "generated_tokens": generated_tokens,
                "output_tokens_per_second": generated_tokens / elapsed if elapsed else None,
                "generated_text": generated_text,
            }
        )

    gpu = None
    peak_gpu_memory_bytes = None
    if use_cuda:
        gpu = torch.cuda.get_device_name(0)
        peak_gpu_memory_bytes = torch.cuda.max_memory_allocated()

    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model,
        "prompt": args.prompt,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            "prompt_tokens": prompt_tokens,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.runs,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": device,
            "gpu": gpu,
            "dtype": str(dtype).removeprefix("torch."),
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        },
        "model_load_seconds": model_load_seconds,
        "runs": run_records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"Device: {device}" + (f" ({gpu})" if gpu else ""))
    print(f"Model load: {model_load_seconds:.2f}s; prompt tokens: {prompt_tokens}")
    for record in run_records:
        print(
            f"Run {record['run']}: {record['generation_seconds']:.2f}s, "
            f"{record['generated_tokens']} output tokens, "
            f"{record['output_tokens_per_second']:.2f} tokens/s"
        )
    print(f"Saved results to {args.output}")
    print("Sample output:\n" + run_records[-1]["generated_text"])


if __name__ == "__main__":
    main()
