#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
# Each run owns its report. Other evaluator options are passed through.
for demo_arg in "$@"; do
  case "${demo_arg}" in
    --output|--output=*) printf 'Script tự tạo đường dẫn output mới cho mỗi lần chạy.\n' >&2; exit 2 ;;
  esac
done
new_demo_directory benchmark
demo_report="${demo_run_dir}/report.json"
heading 'Benchmark thật (mặc định: dataset v2, prompt v1, temperature 0)'
printf 'Có thể thêm --case ID, --prompt-version v3 hoặc --temperature 0.7.\n'
cd -- "${demo_root}"
demo_status=0
"${demo_python}" -m scripts.eval_qwen \
  --cases "${demo_root}/evals/current_scope_tasks_v2.json" \
  --prompt-version v1 --temperature 0 \
  "$@" --output "${demo_report}" || demo_status=$?
if [[ -s "${demo_report}" ]]; then
  heading 'Tổng kết'
  "${demo_python}" "${demo_dir}/_demo.py" report "${demo_report}"
fi
printf '\nExit code evaluator: %s (1 có thể do ca trượt hoặc lỗi; xem báo cáo).\n' "${demo_status}"
exit "${demo_status}"
