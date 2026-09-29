#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "${demo_root}"
heading 'Kiểm tra hash, quota và family — không chạy model trên development/holdout'
"${demo_python}" -m scripts.audit_proposal_data --verify-lock
new_demo_directory proposal-review
heading 'Tạo report bằng fixture pilot; 6 lỗi proposal cố ý, không phải điểm model'
eval_status=0
"${demo_python}" -m scripts.eval_proposals --mode scripted --output "${demo_run_dir}/report.json" > "${demo_run_dir}/evaluator.log" || eval_status=$?
if [[ "${eval_status}" -ne 1 ]]; then
  echo "Fixture phải có exit code 1; thực tế: ${eval_status}" >&2
  exit 1
fi
"${demo_python}" - "${demo_run_dir}/report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding='utf-8'))
assert r['mode'] == 'scripted'
assert r['summary']['automatic'] == {'pass': 18, 'fail': 6, 'error': 0}
print('Fixture: 18 pass, 6 fail cố ý; xem evaluator.log để đọc từng ca.')
PY
heading 'Tạo review riêng gắn hash; chưa chấm thay người review'
"${demo_python}" -m scripts.review_proposals init --report "${demo_run_dir}/report.json" --output "${demo_run_dir}/review.json"
"${demo_python}" -m scripts.review_proposals check --report "${demo_run_dir}/report.json" --review "${demo_run_dir}/review.json"
printf '\nĐiền reviewer và criteria/evidence trong: %s\n' "${demo_run_dir}/review.json"
printf 'Giữ nguyên report: %s\n' "${demo_run_dir}/report.json"
printf 'Sau đó dùng check --require-complete theo evals/README.proposal-data.md.\n'
