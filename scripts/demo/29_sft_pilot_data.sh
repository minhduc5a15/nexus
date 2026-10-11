#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-.venv/bin/python}"
mode="${1:-audit}"

case "$mode" in
  audit)
    PYTHONPATH=. "$python_bin" scripts/build_sft_pilot_v1.py --check
    PYTHONPATH=. "$python_bin" scripts/audit_sft_pilot.py
    ;;
  tokens)
    PYTHONPATH=. "$python_bin" scripts/inspect_sft_tokens.py \
      --dataset evals/sft_pilot_v1.json \
      --case sft_create_delimited \
      --all-cases \
      --summary-only \
      --tokenizer Qwen/Qwen3-4B-Instruct-2507
    ;;
  *)
    echo "Cách dùng: $0 [audit|tokens]" >&2
    exit 2
    ;;
esac
