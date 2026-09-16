# Chạy demo NEXUS và xem output

Chạy từ thư mục dự án. Các script cũng chạy được từ thư mục khác nếu dùng đường
dẫn đầy đủ. Mặc định dùng `.venv/bin/python` và source trong `src/` của checkout
hiện tại. Có thể đặt `NEXUS_DEMO_PYTHON=/đường/dẫn/python` để chọn interpreter khác.

| Script | Bạn sẽ thấy gì? | Cần Qwen? |
| --- | --- | --- |
| `01_cli.sh` | Thêm một dòng, nhiều dòng, bỏ dòng trống, giữ task trùng, xem danh sách | Không |
| `02_policy.sh` | 11 prompt/proposal mẫu và quyết định allow/reject/needs_clarification kèm reason | Không |
| `03_agent_offline.sh` | Trace model giả lập → policy thật → tool thật → SQLite thật → reply | Không |
| `04_ai.sh` | Output Qwen thật, toàn bộ trace và database sau mỗi lượt | Có |
| `05_benchmark.sh` | Chạy scoring v2, in summary và reply từng ca | Có |
| `06_tests.sh` | Tên từng unit test và kết quả | Không |
| `07_report.sh` | Đọc report JSON có sẵn, không gọi lại model | Không |

## Bắt đầu bằng ba demo offline

```sh
./scripts/demo/01_cli.sh
./scripts/demo/02_policy.sh
./scripts/demo/03_agent_offline.sh
```

`01_cli.sh` kết thúc với 4 task. `02_policy.sh` kiểm tra policy trực tiếp, không
mở database. Proposal ở `02` và response model ở `03` được dựng sẵn để minh họa,
không phải bằng chứng về năng lực Qwen.

`03_agent_offline.sh` dùng cùng database demo qua các lượt. Lượt đầu tạo một
task, lượt nhiều dòng tạo thêm hai task; các proposal sai tiếp theo bị chặn và
không làm đổi dữ liệu. Cuối cùng `list_tasks` đọc đúng ba task đó.

## Chạy AI thật

Terminal thứ nhất:

```sh
./scripts/start_qwen.sh
```

Đợi server báo model đã nạp và lắng nghe. Ở terminal thứ hai:

```sh
./scripts/demo/04_ai.sh
```

Mặc định script chạy ba yêu cầu liên tiếp trong cùng database demo: thêm việc,
xem danh sách, rồi đưa một câu trần thuật. Cấu hình là prompt `v1`, temperature
`0`; kết quả thực tế phụ thuộc model.

Có thể đưa câu của bạn:

```sh
./scripts/demo/04_ai.sh "Thêm việc mua sữa vào danh sách giúp tôi."
./scripts/demo/04_ai.sh $'Ghi lại hai việc sau, mỗi dòng một việc:\nmua sữa\ngọi mẹ'
```

Mỗi lần chạy script tạo database mới. Nó không phải phiên chat có memory:
các lượt chia sẻ database, nhưng mỗi request model chỉ nhận yêu cầu hiện tại.
Biến `QWEN_ENDPOINT` dùng được khi server chạy ở địa chỉ khác.

## Benchmark và báo cáo

Chạy 20 ca bằng dataset v2 đang có trong checkout:

```sh
./scripts/demo/05_benchmark.sh
```

Chạy nhanh một vài ca hoặc chọn prompt khác:

```sh
./scripts/demo/05_benchmark.sh --case create_one --case bare_statement
./scripts/demo/05_benchmark.sh --prompt-version v3 --temperature 0
```

Script giữ exit code của evaluator: `1` có thể do ca trượt hoặc lỗi runtime.
Đọc các trạng thái và trường lỗi để phân biệt. Báo cáo vẫn được in khi có ca
trượt. Script tự chọn output mới; không truyền `--output`.

Đọc báo cáo gần nhất được tìm thấy trong `evals/results/`:

```sh
./scripts/demo/07_report.sh
```

Hoặc chỉ định đúng file JSON được script benchmark in ra:

```sh
./scripts/demo/07_report.sh evals/results/policy-contract-2026-09-16/after-final/report.json
```

## Chạy test

```sh
./scripts/demo/06_tests.sh
./scripts/demo/06_tests.sh tests.test_policy_contract
```

## Dữ liệu sinh ra

Các script `01`, `03`, `04`, `05` tạo thư mục riêng dưới
`evals/results/demos/`, được Git ignore. Mỗi lần chạy có đường dẫn mới và được
in ngay từ đầu; database chính của ứng dụng không được sử dụng.

- `tasks.db`: database của demo CLI/agent.
- `trace.json`: chi tiết mỗi lượt agent offline hoặc live.
- `report.json`: kết quả evaluator của demo benchmark.

File demo được giữ lại để bạn mở bằng SQLite viewer hoặc đọc JSON sau khi chạy.
`_common.sh` và `_demo.py` là phần dùng chung; chạy các script đánh số để sử dụng.
Khi dùng xong AI/benchmark, nhấn Ctrl+C tại terminal chạy server để giải phóng GPU.
