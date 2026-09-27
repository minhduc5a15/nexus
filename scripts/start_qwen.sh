#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
server_bin="${NEXUS_LLAMA_SERVER:-${project_root}/.local-runtime/llama-b10809/llama-server}"
model_path="${NEXUS_QWEN_MODEL:-${project_root}/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
model_id="${NEXUS_MODEL_ID:-qwen3-4b-instruct-2507-q4_k_m}"

if [[ ! -x "${server_bin}" ]]; then
  echo "Không tìm thấy llama-server: ${server_bin}" >&2
  exit 1
fi
if [[ ! -f "${model_path}" ]]; then
  echo "Không tìm thấy model: ${model_path}" >&2
  exit 1
fi

extra_args=()
if [[ -n "${NEXUS_LLAMA_LOG_PROMPTS_DIR:-}" ]]; then
  extra_args+=(--log-prompts-dir "${NEXUS_LLAMA_LOG_PROMPTS_DIR}")
fi

exec "${server_bin}" \
  --model "${model_path}" \
  --alias "${model_id}" \
  --host 127.0.0.1 \
  --port "${NEXUS_QWEN_PORT:-8087}" \
  --ctx-size "${NEXUS_QWEN_CONTEXT:-4096}" \
  --parallel 1 \
  --threads "${NEXUS_QWEN_THREADS:-6}" \
  --gpu-layers "${NEXUS_QWEN_GPU_LAYERS:-all}" \
  --fit off \
  --jinja \
  "${extra_args[@]}"
