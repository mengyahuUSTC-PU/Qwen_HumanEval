"""Run a small grid search over decoding parameters for HumanEval inference."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_SCRIPT = REPO_ROOT / "scripts" / "run_humaneval_inference_multisample.py"
POSTPROCESS_SCRIPT = (
    REPO_ROOT / "scripts" / "postprocess_humaneval_completions.py"
)
EVAL_SCRIPT = REPO_ROOT / "scripts" / "evaluate_humaneval.py"

BASE_CMD = [
    sys.executable,
    str(RUN_SCRIPT),
    "--use-chat",
    "--samples-per-task",
    "1",
    "--best-of",
    "4",
    "--max-tokens",
    "2048",
    "--chat-template-kwargs",
    '{"enable_thinking": false}',
]


@dataclass(frozen=True)
class Params:
    temperature: float
    top_p: float
    top_k: int | None

    def tag(self) -> str:
        top_k_str = "none" if self.top_k is None else str(self.top_k)
        return (
            f"temp{self.temperature:.2f}_topp{self.top_p:.2f}_"
            f"topk{top_k_str}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="*",
        default=[0.6, 0.7, 0.8],
    )
    parser.add_argument(
        "--top-ps",
        type=float,
        nargs="*",
        default=[0.7, 0.8, 0.9],
    )
    parser.add_argument(
        "--top-ks",
        type=int,
        nargs="*",
        default=[15, 20, 25],
        help="Use 0 to indicate top_k disabled",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=164,
        help="Number of HumanEval tasks to sample for each configuration",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("outputs/grid_search"),
        help="Directory to store intermediate completions/results",
    )
    parser.add_argument(
        "--sudo",
        action="store_true",
        help="Use sudo when running the evaluation Docker container",
    )
    return parser.parse_args()


def iter_params(
    temps: Iterable[float],
    top_ps: Iterable[float],
    top_ks: Iterable[int],
) -> List[Params]:
    combos: List[Params] = []
    for temp, top_p, raw_top_k in itertools.product(temps, top_ps, top_ks):
        top_k = None if raw_top_k <= 0 else raw_top_k
        combos.append(Params(temperature=temp, top_p=top_p, top_k=top_k))
    return combos


def run_command(cmd: List[str]) -> None:
    printable = [str(item) for item in cmd]
    print("::", " ".join(printable))
    subprocess.run(printable, check=True)


def run_inference(params: Params, limit: int, workspace: Path) -> Path:
    tag = params.tag()
    output_path = workspace / f"humaneval_completions_{tag}.jsonl"

    cmd = BASE_CMD + [
        "--temperature",
        f"{params.temperature}",
        "--top-p",
        f"{params.top_p}",
        "--output",
        str(output_path),
        "--limit",
        str(limit),
    ]
    if params.top_k is not None:
        cmd += ["--top-k", str(params.top_k)]
    run_command(cmd)
    return output_path


def postprocess(input_path: Path, workspace: Path) -> Path:
    processed = workspace / f"{input_path.stem}_post.jsonl"
    cmd: list[str] = [
        str(sys.executable),
        str(POSTPROCESS_SCRIPT),
        "--input",
        str(input_path),
        "--output",
        str(processed),
    ]
    run_command(cmd)
    return processed


def evaluate(input_path: Path, workspace: Path, sudo: bool) -> Path:
    results_path = workspace / f"{input_path.stem}_eval.json"
    cmd: list[str] = [
        str(sys.executable),
        str(EVAL_SCRIPT),
        "--completions",
        str(input_path),
        "--results",
        str(results_path),
        "--ks",
        "1",
    ]
    if sudo:
        cmd.append("--sudo")
    run_command(cmd)
    return results_path


def load_pass_at_1(results_path: Path) -> float:
    with results_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return float(data.get("pass@1", 0.0))


def main() -> int:
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    combos = iter_params(args.temperatures, args.top_ps, args.top_ks)
    summary: list[dict[str, float | str]] = []

    for params in combos:
        print(f"\n=== Evaluating {params.tag()} ===")
        try:
            completions = run_inference(params, args.tasks, workspace)
            processed = postprocess(completions, workspace)
            results = evaluate(processed, workspace, args.sudo)
            pass_at_1 = load_pass_at_1(results)
            summary.append({
                "params": params.tag(),
                "pass@1": pass_at_1,
            })
            print(f"pass@1: {pass_at_1:.4f}")
        except subprocess.CalledProcessError as exc:
            print(f"Configuration {params.tag()} failed: {exc}")

    summary_path = workspace / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nGrid search results written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
