#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
new_demo_directory completion-smoke
demo_report="${demo_run_dir}/report.json"
heading 'Smoke Qwen v9 cho complete_task theo ID'
cd -- "${demo_root}"
demo_status=0
"${demo_python}" -m scripts.eval_session \
  --mode live \
  --prompt-version v9 \
  --temperature 0.0 \
  --cases "${demo_root}/evals/completion_smoke_v1.json" \
  --output "${demo_report}" "$@" || demo_status=$?
printf '\nReport: %s\n' "${demo_report}"
exit "${demo_status}"
