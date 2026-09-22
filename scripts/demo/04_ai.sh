#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
new_demo_directory ai
heading 'Qwen thật: cần scripts/start_qwen.sh ở terminal khác'
if [[ "${1:-}" == '--interactive' ]]; then
  if (( $# != 1 )); then
    echo 'Dùng: 04_ai.sh --interactive (không kèm prompt)' >&2
    exit 2
  fi
  "${demo_python}" "${demo_dir}/_demo.py" live --directory "${demo_run_dir}" --interactive
elif (( $# )); then
  "${demo_python}" "${demo_dir}/_demo.py" live --directory "${demo_run_dir}" --prompt "$*"
else
  "${demo_python}" "${demo_dir}/_demo.py" live --directory "${demo_run_dir}"
fi
