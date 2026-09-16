#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
new_demo_directory ai
heading 'Qwen thật: cần scripts/start_qwen.sh ở terminal khác'
if (( $# )); then
  "${demo_python}" "${demo_dir}/_demo.py" live --directory "${demo_run_dir}" --prompt "$*"
else
  "${demo_python}" "${demo_dir}/_demo.py" live --directory "${demo_run_dir}"
fi
