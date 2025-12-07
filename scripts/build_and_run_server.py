"""Utility to build and launch a vLLM-based server for Qwen models.

This script wraps Docker CLI commands so we can reproducibly build the image
and start an OpenAI-compatible endpoint that serves Qwen/Qwen3-0.6B by default.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run a command and stream output, raising on failure."""
    subprocess.run(cmd, env=env, check=True)


def build_image(args: argparse.Namespace) -> None:
    context_path = Path(args.context).resolve()
    dockerfile_path = context_path / args.dockerfile
    if not dockerfile_path.is_file():
        raise FileNotFoundError(f"Dockerfile not found at {dockerfile_path}")

    cmd = [
        "docker",
        "build",
        "-t",
        args.image_name,
        "-f",
        str(dockerfile_path),
        str(context_path),
    ]
    if args.hf_token:
        cmd.extend(["--build-arg", f"HUGGINGFACE_TOKEN={args.hf_token}"])

    print(":: Building Docker image ::")
    run(cmd)


def launch_container(args: argparse.Namespace) -> None:
    server_env = {
        "MODEL_NAME": args.model_name,
        "SERVED_MODEL_NAME": args.served_model_name or args.model_name,
        "PORT": str(args.container_port),
        "HOST": args.host,
        "TENSOR_PARALLEL_SIZE": str(args.tensor_parallel_size),
        "MAX_MODEL_LEN": str(args.max_model_len),
        "MAX_NUM_SEQS": str(args.max_num_seqs),
    }
    if args.hf_token:
        server_env["HUGGINGFACE_TOKEN"] = args.hf_token

    env_args: list[str] = []
    for key, value in server_env.items():
        env_args.extend(["-e", f"{key}={value}"])

    cmd = [
        "docker",
        "run",
        "--rm",
        "-d" if args.detach else "-it",
        "--name",
        args.container_name,
        "-p",
        f"{args.host_port}:{args.container_port}",
    ] + env_args

    for volume in args.volume:
        cmd.extend(["-v", volume])

    if args.gpu:
        cmd.extend(["--gpus", args.gpu])

    cmd.append(args.image_name)

    if args.vllm_extra_args:
        cmd.extend(shlex.split(args.vllm_extra_args))

    print(":: Launching container ::")
    run(cmd)

    if args.detach:
        print(
            "Container started in detached mode. Attach logs with: "
            f"docker logs -f {args.container_name}"
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--context",
        default="..",
        help="Docker build context directory relative to this script",
    )
    parser.add_argument(
        "--dockerfile",
        default="Dockerfile",
        help="Dockerfile path relative to the build context",
    )
    parser.add_argument(
        "--image-name",
        default="qwen3-vllm",
        help="Tag to assign to the built Docker image",
    )
    parser.add_argument(
        "--container-name",
        default="qwen3-vllm",
        help="Name for the running container",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface the server binds to inside the container",
    )
    parser.add_argument(
        "--host-port",
        type=int,
        default=8000,
        help="Host port to expose the API server on",
    )
    parser.add_argument(
        "--container-port",
        type=int,
        default=8000,
        help="Internal container port for the API server",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-0.6B",
        help="Model identifier to load with vLLM",
    )
    parser.add_argument(
        "--served-model-name",
        default="qwen3-0.6b",
        help="Public name exposed by the OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel degree for multi-GPU setups",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum sequence length to serve",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=16,
        help="Maximum concurrent sequences the server accepts",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HUGGINGFACE_TOKEN", ""),
        help="Optional Hugging Face token for gated models",
    )
    parser.add_argument(
        "--gpu",
        default="",
        help="GPU runtime spec passed to docker run --gpus (e.g. 'all')",
    )
    parser.add_argument(
        "--volume",
        action="append",
        default=[],
        help="Bind mount definitions (repeatable), e.g. host:container",
    )
    parser.add_argument(
        "--vllm-extra-args",
        default="",
        help="Extra flags appended to the vLLM server command",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Run the container in detached mode",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip docker build if the image already exists",
    )

    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    args.context = str((script_dir / args.context).resolve())

    if os.environ.get("DOCKER_BUILDKIT", "") == "":
        os.environ["DOCKER_BUILDKIT"] = "1"

    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if not args.no_build:
        build_image(args)
    else:
        print(":: Skipping build step ::")

    launch_container(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except subprocess.CalledProcessError as exc:
        print(
            f"Command failed with exit code {exc.returncode}",
            file=sys.stderr,
        )
        raise SystemExit(exc.returncode)
