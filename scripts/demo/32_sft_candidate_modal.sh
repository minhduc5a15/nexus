#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "${demo_root}"

stage="${1:-preflight}"
modal_bin="${demo_root}/.venv/bin/modal"

case "${stage}" in
  preflight)
    heading 'Preflight candidate v2 — đã duyệt ngân sách, chưa gọi Modal/GPU/final holdout'
    "${demo_python}" -m scripts.sft_candidate_training_protocol
    ;;
  modal-preflight)
    heading 'Kiểm tra runner bằng Modal CLI — không gọi remote function hoặc GPU'
    "${modal_bin}" run modal_sft_candidate.py
    ;;
  train)
    run_id="${2:?Cách dùng: $0 train RUN_ID}"
    heading 'Train candidate v2 trên T4 — chỉ chạy sau khi manifest ghi nhận phê duyệt ngân sách'
    "${modal_bin}" run modal_sft_candidate.py --stage train --run-id "${run_id}"
    ;;
  development)
    run_id="${2:?Cách dùng: $0 development RUN_ID TRAIN_RUN_ID}"
    train_run_id="${3:?Cách dùng: $0 development RUN_ID TRAIN_RUN_ID}"
    heading 'So base, checkpoint epoch 1 và epoch 2 trên development đã lộ'
    "${modal_bin}" run modal_sft_candidate.py \
      --stage development --run-id "${run_id}" --train-run-id "${train_run_id}"
    ;;
  audit)
    heading 'Xác minh report train, development và review candidate v2 đã tải về'
    "${demo_python}" -m scripts.sft_candidate_training_protocol \
      --train-report evals/results/sft-candidate-v2-train-20261012a/sft-candidate-v2/train.json \
      --train-run-id sft-candidate-v2-train-20261012a \
      --development-report evals/results/sft-candidate-v2-development-20261013a/sft-candidate-v2/development.json \
      --development-run-id sft-candidate-v2-development-20261013a \
      --review evals/sft_candidate_development_review_v2.json
    ;;
  *)
    printf 'Stage phải là preflight, modal-preflight, train, development hoặc audit.\n' >&2
    exit 2
    ;;
esac
