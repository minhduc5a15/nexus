#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
new_demo_directory delete-eval
demo_report="${demo_run_dir}/report.json"
heading 'Đánh giá scripted cho delete_task theo ID có xác nhận'
cd -- "${demo_root}"
demo_status=0
"${demo_python}" -m scripts.eval_session \
  --mode scripted \
  --prompt-version v11 \
  --cases "${demo_root}/evals/delete_feature_v1.json" \
  --output "${demo_report}" "$@" || demo_status=$?
printf '\nReport: %s\n' "${demo_report}"
exit "${demo_status}"
