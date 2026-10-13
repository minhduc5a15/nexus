#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "${demo_root}"

heading 'Tái tạo dataset v3 và kiểm tra artifact đã track'
"${demo_python}" -m scripts.build_sft_candidate_v3 --check

heading 'Audit 600 nhãn, quota taxonomy và leakage — không gọi model'
"${demo_python}" -m scripts.audit_sft_candidate_v3
