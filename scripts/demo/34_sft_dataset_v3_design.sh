#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "${demo_root}"

heading 'Audit taxonomy lỗi và quota dataset v3/final holdout v2 — không gọi model'
"${demo_python}" -m scripts.audit_sft_dataset_v3_design
