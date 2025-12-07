# syntax=docker/dockerfile:1

# Base image: default to CPU-friendly vLLM build but allow overrides at build time
ARG BASE_IMAGE=vllm/vllm-openai:latest
FROM ${BASE_IMAGE}

# Default configuration values (can be overridden at runtime)
ENV MODEL_NAME="Qwen/Qwen3-0.6B" \
    SERVED_MODEL_NAME="qwen3-0.6b" \
    HOST="0.0.0.0" \
    PORT="8000" \
    TENSOR_PARALLEL_SIZE="1" \
    HF_HOME="/root/.cache/huggingface" \
    HF_HUB_ENABLE_HF_TRANSFER="1"

# Optional Hugging Face token build-arg (noop unless provided)
ARG HUGGINGFACE_TOKEN
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

# Copy entrypoint helper
COPY docker/entrypoint.sh /opt/run_vllm.sh
RUN chmod +x /opt/run_vllm.sh

EXPOSE 8000

# Run vLLM OpenAI-compatible server
ENTRYPOINT ["/opt/run_vllm.sh"]
