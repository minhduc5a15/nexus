# Pilot đánh giá model proposal

Evaluator có bộ kiểm thử response giả lập và pilot live 24 ca. Chưa mở rộng
development/holdout và chưa training. Prompt/runtime mặc định của ứng dụng vẫn
là v9/all; phép so sánh proposal dùng v13 để mô tả đủ bảy hành động.

## Chạy và quan sát

```sh
./scripts/demo/23_proposal_eval.sh
./scripts/demo/23_proposal_eval.sh --case create_literal --case delete_direct
./scripts/demo/23_proposal_eval.sh --tool-routing classified
.venv/bin/python -m unittest tests.test_eval_proposals -v
```

Không cần llama-server. Mỗi demo tạo report mới trong `evals/results/demos/` và
không mở database. Report không bị ghi đè nếu đường dẫn đã tồn tại.

Bộ đầy đủ gồm 24 response giả lập: **18 đạt kiểm tra tự động, 6 trượt chủ đích**.
Evaluator trả exit code 1 khi có proposal sai hoặc lỗi generation; đây là kết
quả mong đợi của demo đầy đủ, không có nghĩa unit test thất bại. 18/24 không phải
điểm của Qwen và không phải số câu trả lời đã được review đạt.

| Case | Lỗi cố ý trong fixture | Evaluator cần phát hiện |
| --- | --- | --- |
| create_literal | Bỏ dấu chấm sau lời dẫn có dấu hai chấm | content_truncated, content_punctuation_missing |
| complete_id | ID 7 bị đổi thành 8 | id_mismatch |
| edit_literal | Nói đã sửa nhưng không đề xuất tool | missing_call; câu trả lời vẫn cần review |
| deadline_absolute | Viết lại cụm ngày giờ dù nghĩa tương đương | when_mismatch, when_not_verbatim |
| due_today | Đổi today thành tomorrow | scope_mismatch |
| negated_delete | Đề xuất xóa trong câu “Đừng xóa…” | unexpected_call, call_without_authorization |

## Luồng dữ liệu và trách nhiệm

```text
Nhãn do người viết dataset xác định ───────────────────────┐
                                                        ▼
User prompt → build_model_request → generate(payload) → scorer → report
                  │                         │
          cùng builder với runtime    giữ nguyên response
```

`build_model_request()` trả payload và routing trace, được dùng chung với
`run_turn()`. Nó sao chép settings/tool schema để callback không làm hỏng request
của lượt sau. Evaluator không gọi `run_turn()`, `AgentSession`, `policy_for_tool()`
hay `execute_tool()`; không mở SQLite và không nhận đường dẫn database.

Routing có thể dùng classifier để chọn schema. Kết quả classifier **không phải
nhãn**. Gold proposal đúng nhưng khác route vẫn được chấm đúng về model, đồng
thời ghi `outside_route_calls`. Việc proposal đó được cấp quyền hoặc thực thi
là câu hỏi của evaluator hệ thống, không phải của evaluator này.

`evaluate_case(case, generate, *, prompt_version, settings, tool_routing)` gọi
callback đúng một lần. Callback là adapter model nhận payload, trả response JSON
đã giải mã theo envelope chat-completion. Không tự retry, không sửa JSON và
không lấy JSON trong phần text để biến thành tool call. Adapter của nhà cung cấp
khác phải chuẩn hóa envelope trước khi gọi scorer, không sửa arguments.

## Dataset

`model_proposal_pilot_v1.json` có 14 ca hành động (hai ca mỗi action), sáu ca cần
hỏi lại và bốn ca không thao tác. Đây là development pilot, không phải holdout.

Mỗi ca lưu `id`, `prompt`, `family`, `rationale` và `expected`:

- `behavior`: `call`, `clarify` hoặc `no_action`.
- `action`: một trong bảy action hoặc null khi không xác định hành động task.
- `accepted_calls`: các **phương án thay thế cho một call**, không phải batch.
- `missing_slots`: phần cần hỏi lại, chỉ có giá trị với `clarify`.

Không suy gold từ policy/classifier. Ví dụ `list_paraphrase` có ý định LIST rõ,
được gán nhãn LIST ngay cả khi grammar hiện hành có thể không nhận diện được.
ID không cần tồn tại trong một database: evaluator đang đo proposal theo input,
không kiểm tra việc thực thi. DELETE đúng được tính là proposal đúng, chưa có bất
kỳ xác nhận hoặc xóa dữ liệu nào.

File `model_proposal_pilot_responses_v1.json` chứa response giả lập riêng theo ID.
Nó bao gồm cả proposal sai và lời tự nhận thành công, không phải output model thật.

## Cách đọc report

- `model_request`, `raw_response`: payload trước khi gọi callback và bản sao JSON
  response trả về. Không giữ nguyên byte HTTP; chuỗi arguments vẫn được lưu nguyên.
- `proposed_calls`: giải mã arguments một lần để chấm; invalid JSON giữ chuỗi gốc.
- `checks`: đúng tên tool, schema, toàn bộ proposal và từng trường.
- `error_tags`: những sai khác quan sát được, không suy luận ý định bên trong model.
- `reply_review`: luôn để `pending` khi có response; người review kiểm tra đúng
  phần thiếu, cách nói tương đương và không xác nhận thành công trước tool result.
- `policy_status`, `system_action_status`: `not_evaluated`, tuyệt đối không biến
  “không ghi dữ liệu” thành bằng chứng policy an toàn hoặc hệ thống hoàn thành.

Một response hỏi lại và một response “Đã lưu xong!” đều có thể không gọi tool.
Ở ca thiếu content, cả hai vượt kiểm tra **không đề xuất thao tác**; chỉ câu hỏi
phù hợp mới có thể vượt review câu trả lời. Chưa có cơ chế tự chấm ngữ nghĩa reply.

So sánh arguments phân biệt kiểu dữ liệu: true, 1.0, “1” đều không được xem là
integer 1. Không đổi chữ hoa, bỏ khoảng trắng, chuẩn hóa dấu câu hoặc newline.
JSON key order không ảnh hưởng kết quả; JSON có key trùng hoặc NaN bị từ chối.
Các phương án gold phải khớp nguyên call, không ghép ID của phương án này với
content của phương án khác để được tính đúng.

`accuracy` chỉ là tỷ lệ vượt kiểm tra proposal tự động trên response chấm được.
Lỗi kết nối/generation được ghi riêng và loại khỏi mẫu số; summary luôn in số
lỗi và mẫu số để không che mất failure. `action_exact` chỉ tính ca cần call;
`by_behavior` và `unexpected_call` tách hỏi lại/no-action khỏi hành động.
Trường không liên quan được ghi null, không cộng thành một điểm đúng giả.

## Tái tạo thí nghiệm và giới hạn

Report lưu commit, diff source đang sửa, hash code, source mới chưa commit,
hash dataset/fixture, settings, routing và request/response từng ca. Báo cáo được
flush sau từng ca để giữ phần đã chạy nếu tiến trình bị ngắt. Không lưu API key.

CLI có nhánh live cho lát cắt kế tiếp; bắt buộc khai báo `--model-file` và
`--runtime-version` để ghi hash GGUF và build inference. Metadata này do người
chạy cung cấp; cần đối chiếu server thực tế trước khi dùng làm bằng chứng.
`--seed` chỉ được thêm vào payload khi người chạy cung cấp. Không tự cài runtime,
tải model hoặc khởi động server trong evaluator.

Schema checker chỉ hỗ trợ object với trường string/integer và các constraint
hiện có của bảy tool; đây không phải thư viện JSON Schema tổng quát. Transport
exception tách khỏi lỗi output; response lỗi cấu trúc hoặc bị cắt được tính là
proposal failure, không được sửa để đạt. Chưa đo cold/warm hoặc p95 có kiểm soát;
`generation_seconds` và `elapsed_seconds` chỉ là số đo của từng lượt.

Review xong lát cắt này mới mở rộng lên development 80 ca, khóa final holdout mới,
chạy baseline local và chốt ngân sách cho model đối chứng/training.

## Pilot live 29/09/2026

Đã chạy Qwen 4B Q4_K_M với v13/all và v13/classified, cùng 24 ca, temperature 0,
seed 42, một pass mỗi cấu hình. Cả hai đạt **20/24** tự động, exact action
**12/14**. Classified tăng đúng tên tool từ 13/14 lên 14/14 nhưng EDIT còn sai
content; không đổi mặc định ứng dụng. Đây là kết quả live, khác với 18/24 cố ý
của fixture offline ở trên.

Báo cáo cục bộ: [review pilot](../reports/model-proposal-pilot-2026-09-29.md).
Bằng chứng và [lệnh tái tạo](results/proposal-pilot-20260929/reproduce.md) nằm
trong `evals/results/proposal-pilot-20260929/` (Git ignore): raw all/classified,
comparison, review riêng, hashes, chat template, sampling và log server.

Lần thử Vulkan timeout được giữ riêng; cặp so sánh hoàn tất chạy CPU vì driver
NVIDIA không hoạt động. Chưa dùng pilot này để kết luận latency cold/warm hoặc
hiệu năng GPU. Không thực thi tool hoặc SQLite, không chấm policy/end state.
Review reply do Codex thực hiện, chưa có người review độc lập; raw report vẫn
giữ reply_review pending, kết quả đọc nằm trong các sidecar review-*.json.

### Tóm tắt checkpoint pilot (được Git theo dõi)

| Chỉ số live, v13 | all | classified |
| --- | ---: | ---: |
| Đạt tự động | 20/24 | 20/24 |
| Exact proposal nhóm hành động | 12/14 | 12/14 |
| Đúng tên tool nhóm hành động | 13/14 | 14/14 |
| Call khi chưa đủ yêu cầu | 2/10 | 2/10 |

Bốn ca sai là `create_literal`, `edit_literal`, `bare_statement`, `ambiguous_today`.
Classified giúp EDIT gọi đúng tool nhưng content vẫn thiếu dấu chấm. Hai mode
cùng bỏ dấu chấm ở CREATE, tự thêm từ câu trần và tự chọn today từ câu mơ hồ.
Review reply còn thấy ví dụ thời gian ngoài grammar; classified hỏi lại ID đã có.
Chưa thể quy lỗi hướng dẫn thời gian hoàn toàn cho model vì prompt/schema chưa
mô tả đủ grammar. Không chấm policy, mutation hay hiểu lịch sử hội thoại ở đây.

CPU, temperature 0, seed 42; llama.cpp build 10809 (`5266f24da`).
Dataset SHA-256: `bff0021a3c4c0b08a316411307aeb6026ceaa09d2890f3d1d5e81473e1a3574a`.
Model SHA-256: `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597`.
Cặp CPU đủ 48 response, không lỗi generation. Lần Vulkan timeout được lưu riêng.
308 unit test đạt; 48 raw response chấm lại khớp; prompt/schema và mặc định v9/all
không đổi. Số liệu này là pilot đã xem, không phải bằng chứng holdout mù.

## Development/holdout và review có hash

Xem [hướng dẫn dữ liệu và review](README.proposal-data.md): development 80 ca,
final holdout 120 ca chưa inference, audit lịch sử và CLI review sidecar.
Chạy `./scripts/demo/24_proposal_data_review.sh` để xem workflow offline.
