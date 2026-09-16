#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
heading 'Policy offline: yêu cầu + proposal → quyết định'
"${demo_python}" "${demo_dir}/_demo.py" policy
