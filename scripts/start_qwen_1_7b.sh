#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NEXUS_QWEN_MODEL="${NEXUS_QWEN_MODEL:-${project_root}/models/Qwen3-1.7B-Q8_0.gguf}"
export NEXUS_MODEL_ID="${NEXUS_MODEL_ID:-qwen3-1.7b-q8_0}"
exec "${project_root}/scripts/start_qwen.sh"
