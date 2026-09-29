#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
new_demo_directory tool-routing-diagnostic
demo_report="${demo_run_dir}/report.json"
heading 'Diagnostic Qwen v13 với classified tool routing'
cd -- "${demo_root}"
demo_status=0
"${demo_python}" -m scripts.eval_session \
  --mode live \
  --prompt-version v13 \
  --tool-routing classified \
  --temperature 0 \
  --cases "${demo_root}/evals/tool_routing_diagnostic_v1.json" \
  --output "${demo_report}" "$@" || demo_status=$?
printf '\nReport: %s\n' "${demo_report}"
exit "${demo_status}"
