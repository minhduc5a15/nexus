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
| `10_session_smoke.sh` | Chạy bảy ca tích hợp hiện hành qua `AgentSession`, Qwen và prompt v9 | Có |
| `11_completion_eval.sh` | Chấm scripted COMPLETE theo ID, gồm sai ID và không suy ID từ nội dung | Không |
| `12_completion_smoke.sh` | Chạy smoke Qwen v9 cho CREATE → COMPLETE → LIST và các nhánh completion | Có |
| `13_edit_eval.sh` | Chấm scripted EDIT theo ID, gồm continuation, giữ trạng thái và proposal sai | Không |
| `14_edit_smoke.sh` | Chạy smoke Qwen v10 cho sửa nội dung theo ID | Có |
| `15_delete_eval.sh` | Chấm scripted DELETE, xác nhận, stale snapshot và lỗi trước/sau commit | Không |
| `16_delete_smoke.sh` | Chạy smoke Qwen v11 cho DELETE có xác nhận và regression bốn hành động cũ | Có |
| `17_deadline_eval.sh` | Chấm scripted DEADLINE, parser, continuation, rollback và safety | Không |
| `18_deadline_smoke.sh` | Chạy smoke Qwen v12 cho DEADLINE và regression năm hành động cũ | Có |
| `19_deadline_query_eval.sh` | Chấm scripted query hôm nay/ngày mai/quá hạn, pending scope và safety | Không |
| `20_deadline_query_smoke.sh` | Chạy smoke Qwen v13 cho query deadline và toàn bộ regression | Có |
| `21_tool_routing_eval.sh` | Chấm mapping, fallback, tool ngoài route và batch bằng response cố định | Không |
| `22_tool_routing_diagnostic.sh` | Chạy diagnostic Qwen v13 với classified routing | Có |
| `23_proposal_eval.sh` | Chấm 24 proposal giả lập, có sáu lỗi cố ý; không chạy tool hoặc SQLite | Không |
| `24_proposal_data_review.sh` | Kiểm tra khóa dataset và tạo review riêng từ fixture pilot | Không |
| `25_model_error_audit.sh` | Xác minh 19 ca lỗi Q4/Q8, attribution và quyết định SFT/routing | Không |
| `26_model_control_preflight.sh` | Kiểm tra protocol đối chứng Qwen3-30B, model lock, canary và giới hạn chi phí | Không |
| `27_sft_sanity_data.sh` | Audit 16 mẫu SFT; thêm `--tokenize` để xem assistant-only loss mask bằng tokenizer Qwen | Không |
| `28_sft_qlora_modal.sh` | Preflight QLoRA; `probe`/`overfit` train adapter, `generation` so base với adapter trên Modal T4 | Tải model HF khi chạy GPU |
| `29_sft_pilot_data.sh` | Audit và tokenize bộ 100 mẫu SFT pilot trước training | Chỉ tải tokenizer khi dùng `tokens` |
| `30_sft_pilot_modal.sh` | Preflight, train 200 step và so base/adapter trên 80 ca development | Tải model HF và dùng Modal T4 ở stage GPU |
| `31_sft_candidate_data.sh` | Audit bộ 400 mẫu candidate, cân bằng action, review và leakage | Không |
| `32_sft_candidate_modal.sh` | Preflight protocol train 400 mẫu và hai checkpoint; GPU bị khóa đến khi duyệt ngân sách | Chỉ ở stage train/development |

## Bắt đầu bằng ba demo offline

```sh
./scripts/demo/01_cli.sh
./scripts/demo/02_policy.sh
./scripts/demo/03_agent_offline.sh
./scripts/demo/08_session.sh
./scripts/demo/09_session_eval.sh
./scripts/demo/11_completion_eval.sh
./scripts/demo/13_edit_eval.sh
./scripts/demo/15_delete_eval.sh
./scripts/demo/17_deadline_eval.sh
./scripts/demo/19_deadline_query_eval.sh
./scripts/demo/21_tool_routing_eval.sh
./scripts/demo/25_model_error_audit.sh
./scripts/demo/26_model_control_preflight.sh
./scripts/demo/27_sft_sanity_data.sh
./scripts/demo/28_sft_qlora_modal.sh
./scripts/demo/29_sft_pilot_data.sh
./scripts/demo/30_sft_pilot_modal.sh
./scripts/demo/31_sft_candidate_data.sh
./scripts/demo/32_sft_candidate_modal.sh
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

`15_delete_eval.sh` chạy 13 case/27 lượt bằng proposal cố định. Nó kiểm tra DELETE
không chạy trước xác nhận, chọn đúng một ID, snapshot bị stale, rollback, lỗi
formatter sau commit, batch call và xóa ngoài yêu cầu. `16_delete_smoke.sh` chạy
Qwen 4B với prompt v11; lần đo hiện tại đạt 7/8 case, 14/16 lượt và không có
mutation ngoài yêu cầu. Các ca DELETE đều đạt; regression EDIT trượt, nên v9
vẫn là prompt mặc định.

`17_deadline_eval.sh` chạy 14 case/29 lượt và kiểm tra deadline trước/sau từng
lượt cùng `unrequested_deadline_change`. `18_deadline_smoke.sh` chạy Qwen 4B
với prompt v12; lượt đã xác minh đạt 8/9 case, 14/15 lượt, không có mutation
ngoài yêu cầu. Ca trượt là regression EDIT không gọi tool, nên v9 vẫn là prompt
mặc định.


`19_deadline_query_eval.sh` chạy 9 case/18 lượt cho ba scope, ranh giới chồng
lấp hôm nay–quá hạn, completed/null deadline, pending, isolation, scope bị đổi
và batch call. `20_deadline_query_smoke.sh` chạy Qwen 4B với prompt v13; lượt
đã xác minh đạt 12/13 case, 19/20 lượt. Cả ba luồng query mới và regression
CREATE đều đạt, không có mutation ngoài yêu cầu; regression EDIT không gọi
tool, nên v9 tiếp tục là mặc định.

`21_tool_routing_eval.sh` chạy 5/5 ca offline và kiểm tra cả trace routing.
`22_tool_routing_diagnostic.sh` chạy 14 ca/21 lượt live với classified routing.
Lần đo Qwen 4B hiện tại đạt toàn bộ diagnostic và smoke chuẩn, nhưng latency
median tăng từ 1,020 lên 1,414 giây do prompt cache kém ổn định khi schema thay
đổi. Vì vậy classified vẫn cần truyền bằng cờ và mặc định vẫn là `all`.

## Chạy AI thật

Terminal thứ nhất:

```sh
./scripts/download_qwen4.sh  # chỉ cần lần đầu
./scripts/start_qwen.sh
```

Đợi server báo model đã nạp và lắng nghe. Ở terminal thứ hai:

```sh
./scripts/demo/04_ai.sh
./scripts/demo/10_session_smoke.sh
./scripts/demo/12_completion_smoke.sh
./scripts/demo/14_edit_smoke.sh
./scripts/demo/16_delete_smoke.sh
./scripts/demo/18_deadline_smoke.sh
./scripts/demo/20_deadline_query_smoke.sh
./scripts/demo/22_tool_routing_diagnostic.sh
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

Các script sinh dữ liệu (`01`, `03`, `04`, `05`, `08`–`24`) tạo thư mục riêng dưới
`evals/results/demos/`, được Git ignore. Mỗi lần chạy có đường dẫn mới và được
in ngay từ đầu; database chính của ứng dụng không được sử dụng.

- `tasks.db`: database của demo CLI/agent.
- `trace.json`: chi tiết mỗi lượt agent offline hoặc live.
- `report.json`: kết quả evaluator của demo benchmark.

File demo được giữ lại để bạn mở bằng SQLite viewer hoặc đọc JSON sau khi chạy.
`_common.sh` và `_demo.py` là phần dùng chung; chạy các script đánh số để sử dụng.
Các lệnh `ask`, `chat` và evaluator mặc định dùng model ID `qwen3-4b-instruct-2507-q4_k_m`; dùng `--model` để ghi đè. Khi dùng xong AI/benchmark, nhấn Ctrl+C tại terminal chạy server để giải phóng GPU.

`23_proposal_eval.sh` kiểm tra evaluator độc lập với policy/thực thi. Kết quả
mong đợi là 18 proposal đạt, sáu trượt và exit code 1; unit test xác minh chính
sáu lỗi này. Đây không phải điểm Qwen. Report giữ request, raw response, từng
trường arguments và checklist review câu trả lời. Xem
[hướng dẫn chi tiết](../../evals/README.proposals.md).

`24_proposal_data_review.sh` không chạy model trên development/holdout. Nó tạo review
pending để bạn điền bằng chứng; xem [hướng dẫn review](../../evals/README.proposal-data.md).

`25_model_error_audit.sh` không gọi model, tool hoặc SQLite. Nó kiểm tra audit
tracked với bốn raw development report trong `evals/results/modal-gpu-20260930-downloaded/`,
gồm hash, coverage, raw observation, attribution và quy tắc không dùng lỗi expose
sai tool làm mục tiêu SFT. Final holdout không được đọc.

`26_model_control_preflight.sh` chỉ kiểm tra protocol Qwen3-30B-A3B trên máy cục
bộ. Nó xác minh runner và manifest không lệch nhau, canary gồm đúng mười ca sao
nguyên từ development, stage mặc định là preflight, download không dùng GPU,
development cần canary hoàn tất và image Modal không chứa final holdout. Công cụ
`scripts.audit_model_control` còn nhận `--canary-report` và
`--development-report` để xác minh artifact đã tải; development report chỉ được
chấp nhận khi liên kết đúng canary gate, đủ 80 ca theo đúng thứ tự và không có
generation error.

`27_sft_sanity_data.sh` kiểm tra dataset đầu tiên dành cho SFT mà không train.
Mặc định script chỉ audit contract, balance và leakage. Với `--tokenize`, nó dùng
revision tokenizer Qwen đã khóa để xác minh system/user token đều mang label
`-100`, còn assistant reply hoặc tool call là phần duy nhất chịu loss. Xem
[ghi chú SFT sanity](../../evals/README.sft-sanity.md).

`28_sft_qlora_modal.sh` mặc định chỉ audit protocol offline. Stage `probe` chạy
một optimizer step và ghi bằng chứng base frozen/LoRA gradient; stage `overfit`
đòi run ID của probe đã đạt, train 64 step rồi save/reload adapter để so hash.
Stage `generation` sinh bằng base rồi adapter trên 16 mẫu training và không thực
thi proposal. Development và final holdout không được đóng gói vào image. Lệnh
và kết quả thực tế nằm trong [ghi chú SFT sanity](../../evals/README.sft-sanity.md).

`29_sft_pilot_data.sh` kiểm tra bộ 100 mẫu pilot mà chưa train. Chế độ mặc định
đối chiếu JSON với builder, kiểm tra schema, phân bố, grounding, câu trả lời
không tự nhận đã thao tác và leakage với development/final holdout. Chế độ
`tokens` dùng tokenizer Qwen đã khóa revision để xác minh assistant-only mask và
giới hạn 2.048 token cho cả 100 record:

```sh
./scripts/demo/29_sft_pilot_data.sh
./scripts/demo/29_sft_pilot_data.sh tokens
```

`30_sft_pilot_modal.sh` mặc định chỉ xác minh protocol và ngân sách offline.
Stage `train` chỉ đóng gói 100 mẫu pilot, không có development/final holdout.
Stage `development` cần đúng run train đã hoàn tất và so free generation của
base với adapter trên 80 ca diagnostic. Cả hai stage chỉ đánh giá proposal;
không policy, tool hoặc SQLite nào được gọi:

```sh
./scripts/demo/30_sft_pilot_modal.sh train my-pilot-train
./scripts/demo/30_sft_pilot_modal.sh development my-pilot-dev my-pilot-train
```

Stage `audit` không gọi Modal/GPU; nó xác minh lại report train, development,
failed-run evidence và review thủ công của run ngày 2026-10-11 bằng hash và raw
output đã tải về:

```sh
./scripts/demo/30_sft_pilot_modal.sh audit
```

`31_sft_candidate_data.sh` audit bộ candidate v2 gồm 400 mẫu và tùy chọn chạy
tokenizer để kiểm tra giới hạn 2.048 token cùng assistant-only mask:

```sh
./scripts/demo/31_sft_candidate_data.sh
./scripts/demo/31_sft_candidate_data.sh tokens
```

`32_sft_candidate_modal.sh` quản lý protocol candidate v2. Stage `train` lưu
adapter tại step 100 và 200; stage `development` so base và hai checkpoint trên
80 ca development. Stage `audit` chỉ đọc artifact đã tải về, đối chiếu hash,
raw output, automatic scoring và đủ 24 quyết định review non-call; nó không gọi
Modal, model, GPU hay final holdout:

```sh
./scripts/demo/32_sft_candidate_modal.sh audit
```

`33_sft_final_holdout.sh` quản lý human review và protocol final holdout. Review
có thể tạm dừng bằng `q` và tiếp tục sau:

```sh
./scripts/demo/33_sft_final_holdout.sh labels
./scripts/demo/33_sft_final_holdout.sh label-check
./scripts/demo/33_sft_final_holdout.sh label-gate
```

Sau khi `label-gate` đạt, `preflight` kiểm tra toàn bộ hash, candidate, budget và
runner mà không gọi Modal. Stage `run` mới gọi T4; `audit` dùng report đã tải về
để tính lại score, acceptance và xác minh đủ 72 quyết định review non-call:

```sh
./scripts/demo/33_sft_final_holdout.sh preflight
./scripts/demo/33_sft_final_holdout.sh run my-final-run
./scripts/demo/33_sft_final_holdout.sh audit my-final-run
```

Runner không thực thi proposal. Nó lưu từng case vào Modal Volume và cùng run ID
chỉ tiếp tục phần còn thiếu nếu hạ tầng bị ngắt; một run đã hoàn tất không thể bị
ghi đè. Final run hiện tại là `sft-final-holdout-20261013a`; candidate step 200
không đạt gate và đã bị loại trước bước xuất GGUF.

`34_sft_dataset_v3_design.sh` kiểm tra taxonomy 24 ca lỗi, quota 600 mẫu cho
dataset v3 và quota 120 ca cho final holdout v2. Script chỉ đọc artifact và
report local nếu có; nó không tạo dữ liệu, gọi model, GPU, tool hay database:

```sh
./scripts/demo/34_sft_dataset_v3_design.sh
```

`35_sft_candidate_v3.sh` tái tạo 600 record từ builder, xác minh 400 record v2
được giữ nguyên, kiểm tra cân bằng 60 mẫu cho mỗi action, quota taxonomy và
leakage với hai tập evaluation đã lộ. Script không tokenize, train hoặc gọi
model:

```sh
./scripts/demo/35_sft_candidate_v3.sh
```

`36_final_holdout_v2.sh` tái tạo và audit 120 ca blind holdout mới, sau đó cho
project owner review từng nhãn. Stage `audit`, `check` và `gate` không gọi model;
`gate` chỉ đạt khi đủ 120 quyết định `approved`:

```sh
./scripts/demo/36_final_holdout_v2.sh audit
./scripts/demo/36_final_holdout_v2.sh labels
./scripts/demo/36_final_holdout_v2.sh check
./scripts/demo/36_final_holdout_v2.sh gate
```

`35_sft_candidate_v3.sh tokens` chạy tokenizer từ cache local trên đủ 600 mẫu và
kiểm tra assistant-only loss mask. `37_sft_candidate_v3_training_plan.sh` audit
cấu hình 300 optimizer step cùng phép tính trần Modal 2,60 USD. Cả hai đều
offline. Run đầu được duyệt ngày 14/10/2026 đã bị heartbeat local hủy ở step
184. Project owner đã duyệt một lượt thay thế với trần bổ sung 2,60 USD và hai
run ID cố định trong manifest. Hai stage dài chạy với `modal run --detach`;
checkpoint Trainer được ghi thẳng vào Volume và commit sau mỗi lần save:

```sh
./scripts/demo/35_sft_candidate_v3.sh tokens
./scripts/demo/37_sft_candidate_v3_training_plan.sh preflight
./scripts/demo/37_sft_candidate_v3_training_plan.sh train RUN_ID
./scripts/demo/37_sft_candidate_v3_training_plan.sh development RUN_ID TRAIN_RUN_ID
```
