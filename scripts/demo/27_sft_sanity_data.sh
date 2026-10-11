#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "${demo_root}"

heading 'Audit 16 mẫu SFT sanity — không gọi model, tool hoặc SQLite'
"${demo_python}" -m scripts.audit_sft_data

if [[ "${1:-}" == "--tokenize" ]]; then
  heading 'Render cả 16 mẫu bằng tokenizer Qwen đã khóa và kiểm tra loss mask'
  HF_HOME="${demo_root}/.local-runtime/huggingface" \
    "${demo_python}" -m scripts.inspect_sft_tokens \
    --all-cases --tokenizer Qwen/Qwen3-4B-Instruct-2507
else
  heading 'Xem target có cấu trúc của một mẫu'
  "${demo_python}" -m scripts.inspect_sft_tokens --case sft_create_delimited
  printf '\nThêm --tokenize để dùng tokenizer Qwen thật. Cài requirements-sft-inspection.txt trước nếu cần.\n'
fi
