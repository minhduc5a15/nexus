#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
heading 'Unit test offline'
cd -- "${demo_root}"
if (( $# )); then
  "${demo_python}" -m unittest -v "$@"
else
  "${demo_python}" -m unittest discover -s tests -v
fi
