#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "${demo_root}"

heading 'Audit tokenizer và dữ liệu candidate v3 — offline'
"${demo_python}" -m scripts.audit_sft_candidate_v3

heading 'Audit protocol và ngân sách QLoRA v3 được đề xuất — chưa gọi Modal'
"${demo_python}" -m scripts.audit_sft_candidate_training_plan_v3

heading 'Audit runner Modal v3 bị khóa ở local và remote — chưa gọi Modal'
"${demo_python}" -m scripts.sft_candidate_training_protocol_v3
