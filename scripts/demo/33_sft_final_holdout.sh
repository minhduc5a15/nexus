#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "${demo_root}"

stage="${1:-labels}"

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
  *)
    printf 'Stage hiện có: labels, label-check hoặc label-gate.\n' >&2
    exit 2
    ;;
esac
