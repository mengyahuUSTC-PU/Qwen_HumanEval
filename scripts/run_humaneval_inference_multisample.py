"""Generate multiple HumanEval completions per task via a vLLM endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from datasets import load_dataset


class _SafeFormatDict(dict):
    """Return placeholder text if a template key is missing."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - defensive
        return "{" + key + "}"


DEFAULT_MODEL = "qwen3-0.6b"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"


def batched(iterable: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def request_completions(
    *,
    base_url: str,
    model: str,
    prompt: str,
    samples_per_task: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    presence_penalty: Optional[float],
    chat_template_kwargs: Optional[Dict[str, Any]],
    use_chat: bool,
    instruction_template: str,
    system_message: Optional[str],
    stop: List[str],
    request_timeout: float,
) -> List[str]:
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": samples_per_task,
    }

    if top_k is not None and top_k > 0:
        payload["top_k"] = top_k

    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty

    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs

    endpoint = "completions"
    if use_chat:
        formatted_prompt = instruction_template.format_map(
            _SafeFormatDict(prompt=prompt)
        )
        messages: List[Dict[str, str]] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": formatted_prompt})
        payload["messages"] = messages
        endpoint = "chat/completions"
    else:
        payload["prompt"] = prompt
        if stop:
            payload["stop"] = stop

    resp = requests.post(
        f"{base_url}/{endpoint}",
        json=payload,
        timeout=request_timeout,
    )
    if not resp.ok:
        detail = resp.text.strip()
        raise requests.HTTPError(
            f"{resp.status_code} response from {endpoint}: {detail}",
            response=resp,
        )
    data = resp.json()

    completions: List[str] = []
    for choice in data.get("choices", []):
        if use_chat:
            message = choice.get("message") or {}
            completions.append(message.get("content", ""))
        else:
            completions.append(choice.get("text", ""))

    return completions


def run_inference(args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset = load_dataset("openai_humaneval", split=args.split)
    if args.limit > 0:
        dataset = dataset.select(range(args.limit))

    total_tasks = len(dataset)
    results: list[dict[str, Any]] = []

    for idx, record in enumerate(dataset, start=1):
        task_id = record["task_id"]
        prompt = record["prompt"]

        attempt = 0
        while True:
            try:
                completions = request_completions(
                    base_url=args.base_url,
                    model=args.model,
                    prompt=prompt,
                    samples_per_task=args.samples_per_task,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    presence_penalty=args.presence_penalty,
                    chat_template_kwargs=args.chat_template_kwargs,
                    use_chat=args.use_chat,
                    instruction_template=args.instruction_template,
                    system_message=args.system_message,
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
                print(
                    f"Generation failed for {task_id} ({exc}); "
                    f"retrying in {sleep_time:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(sleep_time)

        if not completions:
            print(
                f"No completions returned for {task_id}",
                file=sys.stderr,
            )
            continue

        for sample_idx, completion in enumerate(completions):
            results.append(
                {
                    "task_id": task_id,
                    "sample_id": f"{task_id}__{sample_idx}",
                    "prompt": prompt,
                    "completion_index": sample_idx,
                    "completion": completion,
                }
            )

        if idx % args.log_every == 0 or idx == total_tasks:
            print(
                f"Completed {idx}/{total_tasks} tasks", file=sys.stderr
            )

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
        help="Optional cap on the number of samples (for smoke tests)",
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=10,
        help="Number of completions to request per task",
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
        default=0.8,
        help="Sampling temperature for stochastic decoding",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling cutoff (omit to disable)",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="Presence penalty to encourage novel tokens",
    )
    parser.add_argument(
        "--use-chat",
        action="store_true",
        help="Use chat completions endpoint instead of plain completions",
    )
    parser.add_argument(
        "--instruction-template",
        default=(
            "You are a precise Python coding assistant.\n\n"
            "Follow this workflow strictly:\n"
            "1. Output `Reasoning:` followed by a short paragraph that "
            "explains your plan for this task, mentally runs through at least "
            "two illustrative inputs (including ones hinted in the prompt), "
            "and double-checks that loops or accumulators update the correct "
            "variables before any comparisons.\n"
            "2. If that mental simulation exposes a bug, describe the fix in "
            "the same paragraph before moving on.\n"
            "3. On the next lines write `Answer:`, then the final Python "
            "code.\n"
            "Starter code:\n```python\n{prompt}\n```"
        ),
        help="Template applied when --use-chat is set (must include {prompt})",
    )
    parser.add_argument(
        "--system-message",
        default=(
            "You are a meticulous Python coding assistant."
            " Follow the user's instructions precisely."
        ),
        help="Optional system message sent ahead of the user prompt",
    )
    parser.add_argument(
        "--stop",
        nargs="*",
        default=[],
        help="Stop sequences for generation",
    )
    parser.add_argument(
        "--chat-template-kwargs",
        default="",
        help="JSON string passed to chat_template_kwargs",
    )
    parser.add_argument(
        "--output",
        default="outputs/humaneval_completions_multi.jsonl",
        help="Path to write the JSONL completions",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
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
        default=5,
        help="Progress logging frequency (in tasks)",
    )
    args = parser.parse_args(argv)

    if args.top_k is not None and args.top_k <= 0:
        args.top_k = None

    if args.chat_template_kwargs:
        try:
            args.chat_template_kwargs = json.loads(
                args.chat_template_kwargs
            )
        except json.JSONDecodeError as json_err:  # pragma: no cover
            parser.error(
                f"Invalid JSON for chat-template-kwargs: {json_err}"
            )
    else:
        args.chat_template_kwargs = {}

    if not args.system_message:
        args.system_message = None

    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.samples_per_task < 1:
        print("samples-per-task must be >= 1", file=sys.stderr)
        return 1

    try:
        results = run_inference(args)
    except Exception as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        return 1

    save_results(results, Path(args.output))
    total_rows = len(results)
    print(
        f"Wrote {total_rows} completions "
        f"({args.samples_per_task} per task) to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
