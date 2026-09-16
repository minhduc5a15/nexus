#!/usr/bin/env bash
# Shared setup; source this file from an executable demo.
set -euo pipefail

demo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
demo_root="$(cd -- "${demo_dir}/../.." && pwd)"
demo_python="${NEXUS_DEMO_PYTHON:-${demo_root}/.venv/bin/python}"
if [[ ! -x "${demo_python}" ]]; then
  printf 'Không tìm thấy Python: %s\nHãy tạo .venv theo README hoặc đặt NEXUS_DEMO_PYTHON.\n' "${demo_python}" >&2
  exit 1
fi
export PYTHONPATH="${demo_root}/src:${demo_root}${PYTHONPATH:+:${PYTHONPATH}}"

new_demo_directory() {
  mkdir -p "${demo_root}/evals/results/demos"
  demo_run_dir="$(mktemp -d "${demo_root}/evals/results/demos/${1}-XXXXXX")"
  printf 'Dữ liệu demo: %s\n' "${demo_run_dir}"
}

heading() {
  printf '\n=== %s ===\n' "$1"
}
