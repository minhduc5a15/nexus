#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
for demo_arg in "$@"; do
  case "${demo_arg}" in
    --output|--output=*|--cases|--cases=*|--mode|--mode=*)
      printf 'Script tự chọn dataset, mode và đường dẫn output.\n' >&2
      exit 2
      ;;
  esac
done
new_demo_directory session-smoke
demo_report="${demo_run_dir}/report.json"
heading 'Smoke test AgentSession với Qwen thật'
printf 'Cần chạy scripts/start_qwen.sh ở terminal khác.\n'
cd -- "${demo_root}"
demo_status=0
"${demo_python}" -m scripts.eval_session \
  --mode live \
  --cases "${demo_root}/evals/session_smoke_v1.json" \
  --prompt-version v1 --temperature 0 \
  "$@" --output "${demo_report}" || demo_status=$?
printf '\nReport: %s\n' "${demo_report}"
printf 'Exit code evaluator: %s (1 nghĩa là có ca trượt; xem trace trong report).\n' "${demo_status}"
exit "${demo_status}"
