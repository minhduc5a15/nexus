# Dataset proposal và review lời trả lời

Checkpoint evaluator/pilot: `ffe8b2d`. Lát cắt này chỉ chuẩn bị dữ liệu và workflow
review. Không chạy baseline 80 ca, không inference trên holdout, không thay prompt,
policy, model hoặc mặc định v9/all. Không training.

## Các tập đã khóa

| Tập | Tổng | Action | Hỏi lại/không thao tác | Trạng thái |
| --- | ---: | ---: | ---: | --- |
| [Development](model_proposal_development_v1.json) | 80 | 56, mỗi action 8 | 24 | 24 pilot đã xem + 56 ca mới |
| [Final holdout](model_proposal_holdout_v1.json) | 120 | 84, mỗi action 12 | 36 | Chưa gọi model, chưa xem kết quả |

Pilot được giữ nguyên ID, prompt, nhãn, rationale và family; chỉ bổ sung metadata
cho development. Ví dụ `known_fields.id=2` ở ca thiếu thời gian giúp người review
nhìn thấy dữ liệu đã có. Metadata này được ghi vào report nhưng **không gửi vào
request model**. Các report cũ không có known_fields vẫn đọc được: người review
phải đối chiếu prompt, không suy rằng thiếu metadata nghĩa là user chưa cung cấp.

Các chiều có trong dữ liệu: lời lịch sự/đời thường, vị trí ID/thời gian trong câu,
lỗi gõ nhẹ ở framing, pha tiếng Anh hoặc yêu cầu tiếng Anh, nội dung nhiều dòng,
dòng trống, chữ hoa, dấu câu, cụm hành động/phủ định nằm trong content, thời gian
tuyệt đối/tương đối, thiếu dữ liệu, nhiều ID, phủ định và ngoài phạm vi.

`expected.accepted_calls` là các phương án cho **một** call. DELETE đúng nhãn chỉ
là proposal; không có quyền xóa trước xác nhận. Content/when được gán nguyên văn;
when của các ca hành động nằm trong grammar parser hiện có. ID không cần tồn tại
trong database để chấm proposal. Câu có ý định rõ nhưng classifier chưa hiểu vẫn
được gán nhãn theo contract, không theo classifier.

## Chia theo family và đối chiếu lịch sử

[Family catalog](model_proposal_families_v1.json) ghi framing cụ thể và split sở hữu.
Không có family nằm trong cả hai tập. Phần hành động holdout gồm **42 family,
mỗi family hai biến thể**; không được coi 84 ca đó là 84 khung câu độc lập. Thay
ID/content trong cùng một family không tạo bằng chứng generalization mới.

[Chỉ mục lịch sử](model_proposal_history_v1.json) giữ 572 bản ghi prompt tham chiếu
(có lặp giữa các bộ cũ), trích từ JSON eval có sẵn và few-shot user messages.
Chỉ mục có hash nguồn; được lưu trong Git để audit không phụ thuộc file report
cũ bị ignore. Đây là dữ liệu kiểm tra overlap, không phải tập training mới.
Không sửa dataset/holdout lịch sử.

Audit so holdout với development và lịch sử: NFKC/casefold, che gold content/when
và chữ số để phát hiện câu chỉ đổi payload hoặc ID. Việc chuẩn hóa này **chỉ dùng
sàng lọc trùng lặp**, không dùng chấm arguments của model. Scorer vẫn so nguyên văn.

Kết quả: không có trùng literal hoặc trùng sau che payload/ID; năm cặp có độ giống
lexical từ 0,75 trở lên. Đã đọc năm cặp và 16 cặp gần nhất; quyết định giữ cùng lý do
nằm trong [family review](model_proposal_family_review_v1.json). Các cặp giữ lại có
framing English/code-switch hoặc hành động khác, không chỉ đổi ID/tên task.

**Giới hạn:** tác giả/reviewer là Codex; chưa có người review độc lập. Similarity
không chứng minh không trùng nghĩa. Holdout là tập chưa xem **kết quả model**,
không phải tập bí mật mà tác giả chưa từng đọc. Chỉ mục không bao phủ mọi trao đổi
ngoài repo hoặc dữ liệu pretraining. Family catalog phải được kiểm tra lại khi tạo
training: không đưa họ câu holdout hoặc biến thể gần của chúng vào train.

## Kiểm tra khóa mà không gọi model

```sh
.venv/bin/python -m scripts.audit_proposal_data --verify-lock
./scripts/demo/24_proposal_data_review.sh
```

Lệnh đầu kiểm tra quota, bảo toàn pilot, family, hash và tính lại audit khớp kết
quả đã khóa. Năm cảnh báo similarity vẫn được hiển thị cùng xác nhận có review;
không giấu cảnh báo bằng cách nâng ngưỡng. [Manifest](model_proposal_manifest_v1.json)
khóa dataset, catalog, history index, audit và quyết định review.

Hash phát hiện file đổi; không phải cơ chế cấm đọc hoặc cấm chạy holdout. Chưa chạy
holdout cho đến khi candidate và cấu hình được chốt. Nếu phải sửa nhãn trước đó,
tạo phiên bản mới và ghi rõ lý do; không âm thầm thay đáp án sau khi thấy response.

Script demo chỉ audit hai tập, sau đó tạo report bằng **fixture pilot giả lập**
(18 pass, 6 fail cố ý). Nó không gửi bất kỳ ca development/holdout nào tới model.
Report và review mới nằm trong thư mục được in dưới `evals/results/demos/`.

## Tạo và điền review riêng

Giả sử `report.json` là output proposal evaluator đã có:

```sh
.venv/bin/python -m scripts.review_proposals init \
  --report /duong/dan/report.json --output /duong/dan/review.json
```

`init` không ghi đè file có sẵn và không sửa report. Điền `reviewer.name` và
`reviewer.kind` (`human` hoặc `assistant`); không gọi review của assistant là đánh
giá độc lập của con người. Mỗi tiêu chí chứa `result` và `evidence`:

| Tiêu chí | Cách review |
| --- | --- |
| `asks_missing` | Có hỏi đúng phần thiếu không? Chỉ áp dụng ca clarify. |
| `avoids_reasking_known` | Có yêu cầu lại dữ liệu user đã nêu không? |
| `no_premature_success` | Có tự nhận side effect thành công trước tool result không? |
| `examples_supported` | Nếu đưa ví dụ, ứng dụng hiện tại có hỗ trợ chúng không? |

`result` nhận `pending`, `pass`, `fail`, `not_applicable`. Bắt buộc ghi bằng chứng
khi chấm pass/fail hoặc tự chọn not_applicable. Chẳng hạn: “User đã nêu ID 2,
reply vẫn yêu cầu cung cấp ID” là bằng chứng fail cho avoids_reasking_known.
“Không đưa ví dụ nào trong reply” là lý do not_applicable cho examples_supported.
Không tự đặt not_applicable cho asks_missing của ca clarify hoặc no_premature_success.

Không dùng một câu trả lời mẫu để exact-match. Hỏi “Bạn muốn đặt hạn lúc nào?” và
“Cho tôi biết thời hạn của việc này nhé” có thể cùng đúng phần thiếu. Một câu trả
lời chứa “Đã đặt hạn!” khi chưa có tool result phải bị đánh dấu, kể cả no-tool
proposal đã pass. Parser thuần Python có thể dùng kiểm chứng ví dụ thời gian;
không cần thực thi tool hoặc mở database.

```sh
.venv/bin/python -m scripts.review_proposals check \
  --report /duong/dan/report.json --review /duong/dan/review.json \
  --require-complete
```

Review gắn SHA-256 với toàn bộ byte report, raw response và ngữ cảnh từng ca.
Không sửa ID, reply, gold hoặc automatic_status trong sidecar. Ghép nhầm run,
đổi report, xóa/đổi thứ tự ca hoặc sửa dấu vân tay đều bị từ chối. Dữ liệu review
chỉ chứng minh gắn đúng artifact; không tự chứng minh judgement của reviewer đúng.

| Exit code | Ý nghĩa |
| --- | --- |
| 0 | File hợp lệ; với require-complete, mọi reply có thể review đã được chấm và không có lỗi generation chưa đánh giá. Không có nghĩa mọi reply đều pass. |
| 1 | Có pending/not_evaluated khi yêu cầu complete. |
| 2 | Hash, schema, bằng chứng hoặc đường dẫn không hợp lệ. |

Điểm reply luôn tách khỏi proposal. Fail proposal không tự biến thành fail reply,
và pass bốn tiêu chí reply không chứng minh hành động đúng. Ví dụ model có thể
đề xuất sai tool nhưng không tự xác nhận thành công. Các vấn đề ngoài bốn tiêu chí
(như lộ nhãn nội bộ) ghi ở `notes`, không tự gộp thành một điểm chất lượng tổng.
Generation lỗi được giữ là not_evaluated, không tính thành reply tốt.

## Điểm dừng và bước sau

Lát cắt này dừng sau dataset, audit, khóa hash, test và workflow review. Bước sau
mới chọn trước 14 ca lặp, chuẩn bị preflight backend và chạy development với
v13/all, v13/classified. Không chạy holdout để chọn prompt/routing hoặc sửa theo ca.
