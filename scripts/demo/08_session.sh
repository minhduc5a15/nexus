#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
new_demo_directory session
heading 'Session CREATE hai lượt và LIST; model LIST được giả lập'
"${demo_python}" "${demo_dir}/_demo.py" session --directory "${demo_run_dir}"
