#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "${demo_root}"
heading 'Xác minh audit lỗi Q4/Q8 từ bốn raw development report — không gọi model'
"${demo_python}" -m scripts.audit_model_errors
heading 'Danh sách quyết định cho dữ liệu huấn luyện'
"${demo_python}" - <<'PY'
import json
from pathlib import Path
value = json.loads(Path('evals/model_proposal_error_audit_v1.json').read_text())
for case in value['cases']:
    causes = sorted({item['primary_cause'] for item in case['failure_attribution'].values()})
    print(f"{case['id']:<18} {case['training_disposition']:<20} {','.join(causes)}")
PY
printf '\nAudit chỉ đọc development report đã có; final holdout không được mở.\n'
