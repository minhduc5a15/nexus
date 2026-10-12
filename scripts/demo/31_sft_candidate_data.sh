#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-.venv/bin/python}"

printf '\n=== Kiểm tra bộ 400 mẫu SFT candidate; không gọi model, GPU hoặc final holdout ===\n\n'
PYTHONPATH=. "$python_bin" scripts/build_sft_candidate_v2.py --check
PYTHONPATH=. "$python_bin" scripts/audit_sft_candidate_v2.py
