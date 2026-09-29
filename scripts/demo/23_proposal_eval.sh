#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
new_demo_directory proposal-eval
demo_report="${demo_run_dir}/report.json"
heading 'Chấm proposal độc lập — response giả lập, không phải benchmark Qwen'
printf 'Bộ mặc định có 6 lỗi chủ đích; exit code 1 là kết quả mong đợi của demo đầy đủ.\n'
cd -- "${demo_root}"
demo_status=0
"${demo_python}" -m scripts.eval_proposals \
  --mode scripted \
  --prompt-version v13 \
  --tool-routing all \
  --output "${demo_report}" "$@" || demo_status=$?
printf '\nReport: %s\n' "${demo_report}"
printf 'Chỉ số tự động không chấm chất lượng câu trả lời; xem reply_review trong report.\n'
exit "${demo_status}"
