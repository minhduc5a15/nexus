#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "${demo_root}"

stage="${1:-preflight}"
run_id="${2:-sft-v3-orchestration-canary-20261016a}"

case "${stage}" in
  preflight)
    heading 'Kiểm tra launcher deployed-spawn — offline'
    "${demo_python}" -m unittest tests.test_modal_sft_v3_control tests.test_modal_sft_v3_control_protocol tests.test_modal_sft_candidate_v3
    "${demo_python}" -m py_compile scripts/modal_sft_v3_control.py modal_sft_candidate_v3.py
    "${demo_python}" -m scripts.modal_sft_v3_control_protocol
    ;;
  deploy)
    heading 'Deploy runner đã khóa; không chạy function hoặc GPU'
    "${demo_root}/.venv/bin/modal" deploy modal_sft_candidate_v3.py
    ;;
  launch)
    heading 'Spawn canary CPU và thoát launcher ngay sau khi lưu FunctionCall ID'
    "${demo_python}" -m scripts.modal_sft_v3_control launch-canary "${run_id}" --duration-seconds 90
    ;;
  status)
    heading 'Poll canary bằng FunctionCall ID; không cancel remote input'
    "${demo_python}" -m scripts.modal_sft_v3_control status "${run_id}"
    ;;
  audit)
    heading 'Audit decision artifact và raw evidence — không gọi Modal'
    "${demo_python}" -m scripts.audit_modal_sft_v3_canary
    ;;
  inspect-checkpoint)
    heading 'Hash checkpoint recovery trên Volume — CPU-only, read-only'
    "${demo_python}" -m scripts.modal_sft_v3_control inspect-checkpoint \
      sft-candidate-v3-train-20261014c 25
    ;;
  *)
    printf 'Stage phải là preflight, deploy, launch, status, audit hoặc inspect-checkpoint.\n' >&2
    exit 2
    ;;
esac
