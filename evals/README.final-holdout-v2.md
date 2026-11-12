# Final Holdout v2: QLoRA v4 step 320

Ngày đánh giá: 12/11/2026 (Asia/Ho_Chi_Minh). Đây là lần inference đầu tiên trên bộ final holdout v2 đã được chủ dự án duyệt 120/120 nhãn. Kể từ lần chạy này, bộ v2 là **diagnostic đã lộ**, không còn là holdout mù cho các candidate về sau.

## Cấu hình và bằng chứng

- Protocol cố định: `evals/sft_final_holdout_manifest_v2.json`, SHA-256 `ea268886950af27c89437513f39dcd7b3d540fe3d9406f9594299043a8531c65`.
- Dataset: `evals/model_proposal_holdout_v2.json`, SHA-256 `a1b9aead11ca80eeab34a827c2f1047b34a7c1c1f491de4123b42930ec2100ed`; 84 ca action, 12 ca hỏi lại, 24 ca no-action.
- Nhãn người dùng duyệt: `evals/model_proposal_holdout_human_review_v2.json`, SHA-256 `72db4c29ff3e0f3e4bccfa254b53ef2fdf85df9c78ef8867a9d574ad5898c0ec`.
- Model gốc `Qwen3-4B-Instruct-2507` revision `cdbee75f17c01a7cc42f958dc650907174af0554`; so sánh cùng v3 step 300 và v4 step 320. Request dùng prompt v13, routing `all`, seed 42, greedy, tối đa 128 token. Trọng số nền được nạp NF4 trên T4; đây chưa phải phép thử GGUF Q4_K_M trên laptop.
- Run ID `sft-final-holdout-v2-20261107a`. Raw report nằm trong `evals/results/sft-final-holdout-v2-20261107a-report.json` (Git ignore), SHA-256 `cde1dcd95e7ef323cb0d2613c131951fe5310c4d6f7b9f72f5ba25ab500c9ea1`.
- Inference chỉ sinh proposal; không gọi policy, tool, session hoặc SQLite. Không sửa JSON, không retry generation error. Auditor đã chấm lại cả 360 response từ raw text: 120/120 mỗi model, 0 generation error.
- Function call chính `fc-01M77TB09QBT7ZV23R71ACABPD` bị ngắt sau 91/120 ca v4 (`KeyboardInterrupt`). Modal tự resume chính call đó, dùng cùng run ID và chỉ sinh phần thiếu; state cuối là `complete`, `resume_count=1`. Call `fc-01M77WEX0F1PNTY95KKFZRQGZR` gửi thêm khi call đầu được báo `failure` đã bị hủy trước khi sinh case nào.

## Kết quả

| Cấu hình | Exact toàn bộ | Action exact | Hỏi lại | No-action | Tool call ngoài quyền | Review ngữ nghĩa control |
|---|---:|---:|---:|---:|---:|---:|
| Base | 100/120 | 73/84 | 11/12 | 16/24 | 9 | 19/36 |
| v3 step 300 | 115/120 | 80/84 | 12/12 | 23/24 | 1 | 35/36 |
| v4 step 320 | 115/120 | 81/84 | 12/12 | 22/24 | 2 | 33/36 |

Review ngữ nghĩa do Codex thực hiện cho toàn bộ 108 câu trả lời control, không phải xác nhận của chủ dự án. Sidecar `evals/sft_final_holdout_review_v2.json` (SHA-256 `c7b339c105f0eb2a9a142d1128169a94945e71606c76b45f25b666caa965bf08`) giữ prompt, raw text, SHA của từng response, quyết định và lý do; `scripts/review_sft_final_holdout_v2.py` xác minh nó khớp raw report. Review này không thay đổi điểm tự động.

| Action | Base | v3 step 300 | v4 step 320 |
|---|---:|---:|---:|
| CREATE | 8/12 | 12/12 | 10/12 |
| LIST | 12/12 | 12/12 | 12/12 |
| COMPLETE | 12/12 | 12/12 | 12/12 |
| EDIT | 7/12 | 11/12 | 12/12 |
| DELETE | 12/12 | 12/12 | 12/12 |
| DEADLINE | 11/12 | 10/12 | 11/12 |
| DUE | 11/12 | 11/12 | 12/12 |

So với v3, v4 sửa bốn ca: `final2_edit_11`, `final2_deadline_11`, `final2_due_11`, `final2_no_action_12`. Nó làm hỏng bốn ca trước đó v3 làm đúng: `final2_create_02`, `final2_create_09`, `final2_no_action_04`, `final2_no_action_18`.

- `create_02`: v4 thêm khoảng trắng đầu content; `create_09`: v4 làm mất dấu tiếng Việt trong content. Đây là lỗi dữ liệu verbatim.
- `no_action_04`: v4 gọi `set_task_deadline` dù người dùng chỉ thuật lại ví dụ trên bảng. Đây là tool call ngoài quyền mới so với cả base lẫn v3.
- `no_action_18`: v4 gọi `set_task_deadline(id=4108, when="none")` cho yêu cầu gỡ deadline, trong khi chức năng gỡ deadline chưa tồn tại. V3 từ chối đúng. Base cũng sai ở ca này nhưng gọi **DELETE**, nên gate tự động chỉ dựa trên tag `unexpected_call` không phân biệt được hai tool sai khác nhau; review thủ công ghi nhận sự khác biệt đó.
- `clarify_12`: v4 không hỏi rõ ý định của câu “Ngày mai liệu có gì đáng lo không?” mà trả lời như đã truy vấn task ngày mai. Automatic no-tool vẫn `pass`; review ngữ nghĩa đánh `fail`.

Median latency T4 lần lượt là **3,10 s** (base), **4,17 s** (v3), **4,41 s** (v4); p95 là 5,92 / 6,57 / 7,09 s. Prompt token mỗi cấu hình là 155.468. Đây là số đo trên Modal, không phải độ trễ triển khai GGUF trên laptop. Chi phí compute ước tính từ thời gian thực thi hai container và bảng giá T4/CPU/RAM trong manifest khoảng 0,57 USD, chưa phải hóa đơn Modal.

## Quyết định

V4 đạt ngưỡng tuyệt đối 81/84 action và giảm 15/20 lỗi của base, nhưng **không vượt các gate đã khóa**: CREATE giảm so với v3, no-action giảm, có tool call ngoài quyền mới và semantic control thấp hơn v3. Do đó v4 step 320 **không đủ điều kiện xuất GGUF hay rollout**. V3 vẫn chỉ là đối chứng nghiên cứu; kết quả final này không tự chọn v3 làm bản triển khai. Model mặc định của ứng dụng chưa đổi.

Bộ holdout v2 đã lộ. Nếu tiếp tục cải thiện model, chỉ dùng v2 để phân tích lỗi; một phép đánh giá mù mới cần dữ liệu final khác và được khóa trước khi chọn candidate. Không dùng các câu v2 hoặc biến thể gần để train rồi gọi lại v2 là bằng chứng generalization.

Kiểm tra lại tại local:

```bash
.venv/bin/python scripts/sft_final_holdout_protocol_v2.py --report evals/results/sft-final-holdout-v2-20261107a-report.json --json
.venv/bin/python -m scripts.review_sft_final_holdout_v2 check
```
