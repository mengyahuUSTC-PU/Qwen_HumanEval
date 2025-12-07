"""Generate HumanEval completions via a vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests
from datasets import load_dataset

DEFAULT_MODEL = "qwen3-0.6b"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"


def batched(iterable: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    """Yield lists of size <= batch_size from the iterable."""
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def make_request(
    base_url: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: List[str],
    request_timeout: float,
) -> str:
    """Call the OpenAI-style completions endpoint and return the text."""
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": stop,
    }

    resp = requests.post(
        f"{base_url}/completions",
        json=payload,
        timeout=request_timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["text"]


def run_inference(args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset = load_dataset("openai_humaneval", split=args.split)
    if args.limit > 0:
        dataset = dataset.select(range(args.limit))

    total = len(dataset)
    log_every = max(1, args.log_every)
    results: list[dict[str, Any]] = []

    for idx, record in enumerate(dataset, start=1):
        task_id = record["task_id"]
        prompt = record["prompt"]

        attempt = 0
        while True:
            try:
                completion = make_request(
                    base_url=args.base_url,
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop=args.stop,
                    request_timeout=args.request_timeout,
                )
                break
            except Exception as exc:  # pragma: no cover
                attempt += 1
                if attempt > args.retries:
                    raise RuntimeError(
                        f"Failed to generate for {task_id}"
                    ) from exc
                sleep_time = args.backoff ** attempt
                retry_msg = (
                    f"Generation failed for {task_id} ({exc}); "
                    f"retrying in {sleep_time:.1f}s"
                )
                print(retry_msg, file=sys.stderr)
                time.sleep(sleep_time)

        results.append(
            {
                "task_id": task_id,
                "prompt": prompt,
                "completion": completion,
            }
        )

        if idx % log_every == 0 or idx == total:
            progress_msg = f"Completed {idx}/{total} tasks"
            print(progress_msg, file=sys.stderr)

    return results


def save_results(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("VLLM_API_URL", DEFAULT_BASE_URL),
        help="Base URL of the OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("VLLM_MODEL_NAME", DEFAULT_MODEL),
        help="Model name registered with the endpoint",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="HumanEval split to evaluate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Optional cap on the number of samples (useful for smoke tests)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to sample for each completion",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling top-p",
    )
    parser.add_argument(
        "--stop",
        nargs="*",
        default=["\nclass", "\ndef", "\nif __name__"],
        help="Stop sequences for generation",
    )
    parser.add_argument(
        "--output",
        default="outputs/humaneval_completions.jsonl",
        help="Path to write the JSONL completions",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="HTTP timeout per request (seconds)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retry attempts per sample",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=2.0,
        help="Exponential backoff base between retries",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="How frequently to print progress updates (in samples)",
    )

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    try:
        results = run_inference(args)
    except Exception as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        return 1

    save_results(results, Path(args.output))
    print(f"Wrote {len(results)} completions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
