#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "${demo_root}"

dataset="evals/model_proposal_holdout_v2.json"
review="evals/model_proposal_holdout_human_review_v2.json"
stage="${1:-audit}"

case "${stage}" in
  audit)
    heading 'Tái tạo và audit final holdout v2 — không gọi model'
    "${demo_python}" -m scripts.build_model_proposal_holdout_v2 --check
    "${demo_python}" -m scripts.audit_model_proposal_holdout_v2
    ;;
  labels)
    heading 'Review thủ công 120 nhãn final holdout v2'
    "${demo_python}" -m scripts.review_holdout_labels \
      --dataset "${dataset}" --review "${review}" interactive
    ;;
  check)
    heading 'Xem tiến độ review final holdout v2'
    "${demo_python}" -m scripts.review_holdout_labels \
      --dataset "${dataset}" --review "${review}" check
    ;;
  gate)
    heading 'Yêu cầu đủ 120 nhãn được project owner duyệt'
    "${demo_python}" -m scripts.review_holdout_labels \
      --dataset "${dataset}" --review "${review}" check --require-complete
    ;;
  *)
    printf 'Stage hiện có: audit, labels, check hoặc gate.\n' >&2
    exit 2
    ;;
esac
