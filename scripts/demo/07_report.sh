#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
heading 'Đọc báo cáo benchmark đã lưu; không gọi model'
"${demo_python}" "${demo_dir}/_demo.py" report "$@"
