#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
new_demo_directory offline
heading 'Runtime và SQLite thật; response model được giả lập'
"${demo_python}" "${demo_dir}/_demo.py" offline --directory "${demo_run_dir}"
