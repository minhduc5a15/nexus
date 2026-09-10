#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
server_bin="${NEXUS_LLAMA_SERVER:-${project_root}/.local-runtime/llama-b10809/llama-server}"
model_path="${NEXUS_QWEN_MODEL:-${project_root}/models/Qwen3-1.7B-Q8_0.gguf}"

if [[ ! -x "${server_bin}" ]]; then
  echo "Không tìm thấy llama-server: ${server_bin}" >&2
  exit 1
fi
if [[ ! -f "${model_path}" ]]; then
  echo "Không tìm thấy model: ${model_path}" >&2
  exit 1
fi

exec "${server_bin}" \
  --model "${model_path}" \
  --host 127.0.0.1 \
  --port "${NEXUS_QWEN_PORT:-8087}" \
  --ctx-size "${NEXUS_QWEN_CONTEXT:-4096}" \
  --parallel 1 \
  --threads "${NEXUS_QWEN_THREADS:-6}" \
  --gpu-layers "${NEXUS_QWEN_GPU_LAYERS:-all}" \
  --fit on \
  --jinja
