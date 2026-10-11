#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "${demo_root}"

stage="${1:-preflight}"
modal_bin="${demo_root}/.venv/bin/modal"

case "${stage}" in
  preflight)
    heading 'Preflight QLoRA sanity — chỉ kiểm tra file, hash, cấu hình và gate'
    "${demo_python}" -m scripts.sft_sanity_protocol
    heading 'Preflight free generation — khóa base/adapter và không mở eval'
    "${demo_python}" -m scripts.sft_generation_protocol
    ;;
  modal-preflight)
    heading 'Preflight qua Modal CLI — không gọi function hoặc GPU'
    "${modal_bin}" run modal_sft_sanity.py
    ;;
  probe)
    run_id="${2:?Cách dùng: $0 probe RUN_ID}"
    heading 'Một backward/optimizer step trên T4'
    "${modal_bin}" run modal_sft_sanity.py --stage probe --run-id "${run_id}"
    ;;
  overfit)
    run_id="${2:?Cách dùng: $0 overfit RUN_ID PROBE_RUN_ID}"
    probe_run_id="${3:?Cách dùng: $0 overfit RUN_ID PROBE_RUN_ID}"
    heading 'Overfit 16 mẫu trong 64 step, sau đó save/reload adapter'
    "${modal_bin}" run modal_sft_sanity.py \
      --stage overfit --run-id "${run_id}" --probe-run-id "${probe_run_id}"
    ;;
  generation)
    run_id="${2:?Cách dùng: $0 generation RUN_ID OVERFIT_RUN_ID}"
    overfit_run_id="${3:?Cách dùng: $0 generation RUN_ID OVERFIT_RUN_ID}"
    heading 'So sánh free generation của base và adapter trên 16 mẫu training'
    "${modal_bin}" run modal_sft_sanity.py \
      --stage generation --run-id "${run_id}" --overfit-run-id "${overfit_run_id}"
    ;;
  *)
    printf 'Stage phải là preflight, modal-preflight, probe, overfit hoặc generation.\n' >&2
    exit 2
    ;;
esac
