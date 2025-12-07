"""Run HumanEval functional correctness evaluation inside a Docker sandbox."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

CONTAINER_MOUNT = "/workspace"
DEFAULT_COMPLETIONS = "../outputs/humaneval_completions.jsonl"
DEFAULT_RESULTS = "../outputs/humaneval_eval_results.json"


def run(cmd: list[str]) -> None:
    """Execute a command, streaming stdout/stderr, and raise if it fails."""
    subprocess.run(cmd, check=True)


def build_docker_command(
    *,
    completions_path: Path,
    results_path: Path,
    timeout: int,
    workers: int,
    ks: list[int],
) -> list[str]:
    host_dir = completions_path.parent
    container_completions = Path(CONTAINER_MOUNT) / completions_path.name
    container_results = Path(CONTAINER_MOUNT) / results_path.name

    code = textwrap.dedent(
        f"""
        import json
        from pathlib import Path

        from human_eval.evaluation import evaluate_functional_correctness

        result = evaluate_functional_correctness(
            sample_file="{container_completions}",
            k={ks},
            n_workers={workers},
            timeout={timeout},
        )
        Path("{container_results}").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        """
    )

    python_cmd = (
        "pip install --quiet human-eval && "
        "python - <<'PY'\n"
        f"{code}"
        "PY"
    )

    return [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{host_dir}:{CONTAINER_MOUNT}",
        "-w",
        CONTAINER_MOUNT,
        "-e",
        "PYTHONUNBUFFERED=1",
        "python:3.10-slim",
        "bash",
        "-lc",
        python_cmd,
    ]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--completions",
        default=DEFAULT_COMPLETIONS,
        help="Path to the HumanEval completions JSONL (host path)",
    )
    parser.add_argument(
        "--results",
        default=DEFAULT_RESULTS,
        help="Path to write evaluation summary JSON (host path)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="Timeout per test case inside the evaluator (seconds)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of parallel worker processes",
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        default=["1"],
        help="List of k values to evaluate pass@k",
    )
    parser.add_argument(
        "--sudo",
        action="store_true",
        help="Run docker commands with sudo",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    base_dir = Path(__file__).resolve().parent
    completions_path = base_dir.joinpath(args.completions).resolve()
    results_path = base_dir.joinpath(args.results).resolve()

    if not completions_path.is_file():
        print(
            f"Completions file not found: {completions_path}",
            file=sys.stderr,
        )
        return 1

    results_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        ks = sorted({int(value) for value in args.ks})
    except ValueError as exc:
        print(f"Invalid k value provided: {exc}", file=sys.stderr)
        return 1

    cmd = build_docker_command(
        completions_path=completions_path,
        results_path=results_path,
        timeout=args.timeout,
        workers=args.workers,
        ks=ks,
    )

    if args.sudo:
        cmd = ["sudo"] + cmd

    print(":: Running HumanEval evaluation inside Docker ::")
    run(cmd)

    if results_path.is_file():
        with results_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        print(json.dumps(summary, indent=2))
    else:
        print(
            "Evaluation finished but result file not found.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
