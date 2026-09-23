#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
new_demo_directory completion-eval
demo_report="${demo_run_dir}/report.json"
heading 'Đánh giá scripted cho complete_task theo ID'
cd -- "${demo_root}"
demo_status=0
"${demo_python}" -m scripts.eval_session \
  --mode scripted \
  --prompt-version v9 \
  --cases "${demo_root}/evals/completion_feature_v1.json" \
  --output "${demo_report}" "$@" || demo_status=$?
printf '\nReport: %s\n' "${demo_report}"
exit "${demo_status}"
