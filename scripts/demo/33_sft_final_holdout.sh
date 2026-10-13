#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "${demo_root}"

stage="${1:-labels}"
modal_bin="${demo_root}/.venv/bin/modal"

case "${stage}" in
  labels)
    heading 'Review thủ công đủ 120 nhãn final holdout — không gọi model hoặc GPU'
    "${demo_python}" -m scripts.review_holdout_labels interactive
    ;;
  label-check)
    heading 'Kiểm tra tiến độ review nhãn final holdout'
    "${demo_python}" -m scripts.review_holdout_labels check
    ;;
  label-gate)
    heading 'Yêu cầu đủ 120 nhãn được human reviewer chấp thuận'
    "${demo_python}" -m scripts.review_holdout_labels check --require-complete
    ;;
  preflight)
    heading 'Audit offline protocol final holdout — không gọi Modal, model hoặc GPU'
    "${demo_python}" -m scripts.sft_final_holdout_protocol
    ;;
  modal-preflight)
    heading 'Kiểm tra local entrypoint của runner — không gọi remote function hoặc GPU'
    "${modal_bin}" run modal_sft_final_holdout.py
    ;;
  run)
    run_id="${2:?Cách dùng: $0 run RUN_ID}"
    heading 'Chạy base và step-200 đúng một lượt trên final holdout đã duyệt'
    "${modal_bin}" run modal_sft_final_holdout.py --stage run --run-id "${run_id}"
    ;;
  audit)
    run_id="${2:?Cách dùng: $0 audit RUN_ID}"
    report="evals/results/${run_id}/sft-final-holdout-v1/report.json"
    heading 'Tính lại scoring và acceptance từ raw final-holdout report đã tải về'
    "${demo_python}" -m scripts.sft_final_holdout_protocol \
      --report "${report}" --run-id "${run_id}" \
      --review evals/sft_final_holdout_review_v1.json
    ;;
  *)
    printf 'Stage hiện có: labels, label-check, label-gate, preflight, modal-preflight, run hoặc audit.\n' >&2
    exit 2
    ;;
esac
