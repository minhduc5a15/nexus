#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "${demo_root}"

stage="${1:-preflight}"
modal_bin="${demo_root}/.venv/bin/modal"

case "${stage}" in
  preflight)
    heading 'Preflight SFT pilot — không gọi Modal, model, GPU hoặc final holdout'
    "${demo_python}" -m scripts.sft_pilot_protocol
    ;;
  modal-preflight)
    heading 'Preflight pilot qua Modal CLI — không gọi remote function hoặc GPU'
    "${modal_bin}" run modal_sft_pilot.py
    ;;
  train)
    run_id="${2:?Cách dùng: $0 train RUN_ID}"
    heading 'Train QLoRA 100 mẫu trong 200 step trên T4'
    "${modal_bin}" run modal_sft_pilot.py --stage train --run-id "${run_id}"
    ;;
  development)
    run_id="${2:?Cách dùng: $0 development RUN_ID TRAIN_RUN_ID}"
    train_run_id="${3:?Cách dùng: $0 development RUN_ID TRAIN_RUN_ID}"
    heading 'So sánh base và adapter trên 80 ca development'
    "${modal_bin}" run modal_sft_pilot.py \
      --stage development --run-id "${run_id}" --train-run-id "${train_run_id}"
    ;;
  audit)
    heading 'Xác minh report train, development và review thủ công đã tải về'
    "${demo_python}" -m scripts.sft_pilot_protocol \
      --train-report evals/results/sft-pilot-train-20261011a/sft-pilot/train.json \
      --train-run-id sft-pilot-train-20261011a \
      --development-report evals/results/sft-pilot-development-20261011b/sft-pilot/development.json \
      --development-run-id sft-pilot-development-20261011b \
      --review evals/sft_pilot_development_review_v1.json \
      --failed-development-state evals/results/sft-pilot-development-20261011a/sft-pilot/development-state.json
    ;;
  *)
    printf 'Stage phải là preflight, modal-preflight, train, development hoặc audit.\n' >&2
    exit 2
    ;;
esac
