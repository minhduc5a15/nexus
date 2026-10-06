#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "${demo_root}"
heading 'Preflight đối chứng Qwen3-30B-A3B — không gọi Modal hoặc model'
"${demo_python}" -m scripts.audit_model_control
heading 'Mười ca canary được sao nguyên từ development v1'
"${demo_python}" - <<'PY'
import json
from pathlib import Path

value = json.loads(Path('evals/model_control_canary_v1.json').read_text(encoding='utf-8'))
for index, case in enumerate(value['cases'], 1):
    expected = case['expected']
    print(f"{index:>2}. {case['id']:<20} {expected['behavior']:<9} {expected['action'] or '-':<10} {case['prompt'].replace(chr(10), ' / ')}")
PY
printf '\nĐây chỉ là kiểm tra file/hash/contract. Không có GPU, inference, tool, SQLite hoặc holdout.\n'
