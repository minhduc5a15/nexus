# Chạy demo NEXUS và xem output

Chạy từ thư mục dự án. Các script cũng chạy được từ thư mục khác nếu dùng đường
dẫn đầy đủ. Mặc định dùng `.venv/bin/python` và source trong `src/` của checkout
hiện tại. Có thể đặt `NEXUS_DEMO_PYTHON=/đường/dẫn/python` để chọn interpreter khác.

| Script | Bạn sẽ thấy gì? | Cần Qwen? |
| --- | --- | --- |
| `01_cli.sh` | Thêm một dòng, nhiều dòng, bỏ dòng trống, giữ task trùng, xem danh sách | Không |
| `02_policy.sh` | 11 prompt/proposal mẫu và quyết định allow/reject/needs_clarification kèm reason | Không |
| `03_agent_offline.sh` | Từng bước request/response giả lập → proposal → policy thật → tool thật → SQLite trước/sau → reply | Không |
| `04_ai.sh` | Cùng luồng với Qwen thật; in trace và SQLite trước/sau mỗi lượt | Có |
| `05_benchmark.sh` | Chạy scoring v2, in summary và reply từng ca | Có |
| `06_tests.sh` | Tên từng unit test và kết quả | Không |
| `07_report.sh` | Đọc report JSON có sẵn, không gọi lại model | Không |
| `08_session.sh` | CREATE thiếu nội dung → hỏi lại → lưu nhiều dòng → hủy → LIST; cho thấy trạng thái session | Không |
| `09_session_eval.sh` | Chấm 13 chuỗi hội thoại giả lập theo state, tool, SQLite, reply và ghi ngoài yêu cầu | Không |
| `10_session_smoke.sh` | Chạy bảy ca tích hợp nhỏ qua `AgentSession` và Qwen thật | Có |
| `11_completion_eval.sh` | Chấm scripted COMPLETE theo ID, gồm sai ID và không suy ID từ nội dung | Không |
| `12_completion_smoke.sh` | Chạy smoke Qwen v9 cho CREATE → COMPLETE → LIST và các nhánh completion | Có |
| `13_edit_eval.sh` | Chấm scripted EDIT theo ID, gồm continuation, giữ trạng thái và proposal sai | Không |
| `14_edit_smoke.sh` | Chạy smoke Qwen v10 cho sửa nội dung theo ID | Có |

## Bắt đầu bằng ba demo offline

```sh
./scripts/demo/01_cli.sh
./scripts/demo/02_policy.sh
./scripts/demo/03_agent_offline.sh
./scripts/demo/08_session.sh
./scripts/demo/09_session_eval.sh
./scripts/demo/11_completion_eval.sh
./scripts/demo/13_edit_eval.sh
```

`01_cli.sh` kết thúc với 4 task. `02_policy.sh` kiểm tra policy trực tiếp, không
mở database. Proposal ở `02` và response model ở `03` được dựng sẵn để minh họa,
không phải bằng chứng về năng lực Qwen.

`03_agent_offline.sh` dùng cùng database demo qua các lượt. Lượt đầu tạo một
task, lượt nhiều dòng tạo thêm hai task; các proposal sai tiếp theo bị chặn và
không làm đổi dữ liệu. Cuối cùng `list_tasks` đọc đúng ba task đó. Console in
theo thứ tự: SQLite trước lượt, request gửi model, response thô, proposal đã
giải mã, quyết định policy, tool đã chạy, trạng thái/reply và SQLite sau lượt.
`trace.json` giữ cả model request/response để mở lại sau.

`08_session.sh` minh họa phần hội thoại hai lượt mới. Lệnh CREATE thiếu nội
dung không gọi model; tin nhắn kế tiếp được lưu nguyên văn thành hai task. Demo
cho thấy hủy yêu cầu đang chờ và một lệnh LIST mới. Chỉ lượt LIST dùng response
model giả lập; SQLite và các nhánh session là thật.

`09_session_eval.sh` chạy 13 cuộc hội thoại, tổng cộng 32 lượt, bằng proposal
model cố định. Báo cáo chấm riêng chuyển state, tool đã chạy, SQLite, reply và
ghi ngoài yêu cầu. Khi session đang chờ nội dung CREATE, tin nhắn tự do kế tiếp
được coi là content theo quyền từ lượt trước; chỉ lệnh mới hoặc hủy rõ ràng mới
thay thế trạng thái chờ.

## Chạy AI thật

Terminal thứ nhất:

```sh
./scripts/start_qwen.sh
```

Đợi server báo model đã nạp và lắng nghe. Ở terminal thứ hai:

```sh
./scripts/demo/04_ai.sh
./scripts/demo/10_session_smoke.sh
./scripts/demo/12_completion_smoke.sh
./scripts/demo/14_edit_smoke.sh
```

Mặc định script chạy ba yêu cầu liên tiếp trong cùng database demo: thêm việc,
xem danh sách, rồi đưa một câu trần thuật. Cấu hình demo `04` là prompt `v1`, temperature
`0`; kết quả thực tế phụ thuộc model.

Để tự gõ nhiều yêu cầu và xem trace từng lượt trên cùng một database demo:

```sh
./scripts/demo/04_ai.sh --interactive
```

Gõ `/exit` hoặc nhấn `Ctrl+D` để kết thúc. Đây vẫn là các lượt `run_turn` độc
lập, chưa phải `AgentSession`; danh sách SQLite được giữ trong suốt phiên demo.

Để dùng hội thoại nhiều lượt thật thay vì demo trace, chạy:

```sh
.venv/bin/nexus chat
.venv/bin/nexus chat --trace
```

Lệnh này giữ một `AgentSession`: “Thêm việc” có thể hỏi lại và dùng tin nhắn kế
tiếp làm nội dung. Nó dùng database chính mặc định; đặt `--db` trước `chat` nếu
muốn thử với file riêng. Mỗi dòng là một tin nhắn. Biến thể `--trace` in request
và response model, quyết định runtime, call, state session trước/sau và SQLite
của từng lượt; lệnh không có `--trace` chỉ in phản hồi dành cho người dùng.

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

Các script `01`, `03`, `04`, `05`, `08` tạo thư mục riêng dưới
`evals/results/demos/`, được Git ignore. Mỗi lần chạy có đường dẫn mới và được
in ngay từ đầu; database chính của ứng dụng không được sử dụng.

- `tasks.db`: database của demo CLI/agent.
- `trace.json`: chi tiết mỗi lượt agent offline hoặc live.
- `report.json`: kết quả evaluator của demo benchmark.

File demo được giữ lại để bạn mở bằng SQLite viewer hoặc đọc JSON sau khi chạy.
`_common.sh` và `_demo.py` là phần dùng chung; chạy các script đánh số để sử dụng.
Khi dùng xong AI/benchmark, nhấn Ctrl+C tại terminal chạy server để giải phóng GPU.
