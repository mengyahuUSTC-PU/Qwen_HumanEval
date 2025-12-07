#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-0.6B}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-${MODEL_NAME}}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}

# Allow users to pass additional flags via VLLM_EXTRA_ARGS
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}

if [[ -n "${HUGGINGFACE_TOKEN:-}" ]]; then
  export HF_TOKEN="${HUGGINGFACE_TOKEN}"
  export HUGGINGFACEHUB_API_TOKEN="${HUGGINGFACE_TOKEN}"
fi

# Ensure cache directory exists to avoid surprises
mkdir -p "${HF_HOME:-/root/.cache/huggingface}"

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  ${VLLM_EXTRA_ARGS}
