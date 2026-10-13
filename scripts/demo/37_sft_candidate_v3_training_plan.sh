#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "${demo_root}"

stage="${1:-preflight}"
modal_bin="${demo_root}/.venv/bin/modal"

case "${stage}" in
  preflight)
    heading 'Audit tokenizer và dữ liệu candidate v3 — offline'
    "${demo_python}" -m scripts.audit_sft_candidate_v3
    heading 'Audit protocol, ngân sách và runner QLoRA v3 — chưa gọi Modal'
    "${demo_python}" -m scripts.audit_sft_candidate_training_plan_v3
    "${demo_python}" -m scripts.sft_candidate_training_protocol_v3
    ;;
  modal-preflight)
    heading 'Kiểm tra local entrypoint bằng Modal CLI — không gọi remote function hoặc GPU'
    "${modal_bin}" run modal_sft_candidate_v3.py
    ;;
  train)
    run_id="${2:?Cách dùng: $0 train RUN_ID}"
    heading 'Train candidate v3 trên T4 — không automatic retry'
    "${modal_bin}" run modal_sft_candidate_v3.py --stage train --run-id "${run_id}"
    ;;
  development)
    run_id="${2:?Cách dùng: $0 development RUN_ID TRAIN_RUN_ID}"
    train_run_id="${3:?Cách dùng: $0 development RUN_ID TRAIN_RUN_ID}"
    heading 'So base, step 150 và step 300 trên development đã lộ'
    "${modal_bin}" run modal_sft_candidate_v3.py \
      --stage development --run-id "${run_id}" --train-run-id "${train_run_id}"
    ;;
  *)
    printf 'Stage phải là preflight, modal-preflight, train hoặc development.\n' >&2
    exit 2
    ;;
esac
