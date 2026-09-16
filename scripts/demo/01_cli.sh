#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
new_demo_directory cli
demo_db="${demo_run_dir}/tasks.db"
nexus_demo() { "${demo_python}" -m nexus.cli.main --db "${demo_db}" "$@"; }

heading 'Danh sách ban đầu'
nexus_demo list
heading 'Thêm một dòng: dấu phẩy và chữ “và” không tách task'
nexus_demo add 'mua sữa, gọi mẹ và học Python'
heading 'Thêm nhiều dòng qua stdin, có một dòng trống'
printf 'tưới cây\n\nđọc sách\n' | nexus_demo add
heading 'Thêm nội dung trùng: vẫn tạo ID mới'
nexus_demo add 'tưới cây'
heading 'Danh sách cuối cùng (4 task)'
nexus_demo list
printf '\nXem lại bằng:\n  %q -m nexus.cli.main --db %q list\n' "${demo_python}" "${demo_db}"
