# Bản nháp SFT v4: bổ sung cách diễn đạt có đối chứng

Trạng thái: **Codex đã review 40 nhãn; project owner đã duyệt đủ 40/40 nhãn ở bản sau sửa; chưa dùng để train**. Quyết định assistant và hash từng ca nằm trong [`sft_candidate_v4_label_review.json`](sft_candidate_v4_label_review.json). Dataset [`sft_candidate_v4_draft.json`](sft_candidate_v4_draft.json) hiện có SHA-256 `6b30bb0cebcb253a8d0357c1a1767b38e9d8172385360d192321e038ee09c31b`. Nó giữ nguyên 600 ví dụ v3 (SHA-256 `a0e54748aed6926d48e7ddc5f6674954bba7a19c9eab9f469cc09db80c7b83e7`) rồi thêm 40 ví dụ, không sửa file v3 hay dataset đánh giá.

Project owner ban đầu đánh dấu 35 ca đúng và 5 ca cần sửa vì dấu `#` trước ID không tự nhiên trong lời người dùng: `v4_deadline_02`, `v4_deadline_07`, `v4_complete_02`, `v4_edit_02`, `v4_delete_02`. Bản nháp trước sửa có SHA-256 `1ae2f972d5497d59777aac5d6c4fc3e872ce95168c8705218f41d7e9a3a7da71`; bản gốc và review gốc được giữ cục bộ ngoài Git. Năm prompt mới chỉ bỏ ký tự `#`, giữ nguyên nhãn action, tool và arguments. Sau lần duyệt lại, sidecar local ghi 40 `approved`, không còn `pending` hay `needs_correction`; hash sidecar là `577392f4126f84aea0c103931794212996a65b0b5314c957393fd8c0196515b5`. Cổng review nhãn không đồng nghĩa với cấp phép chạy GPU.

| Phần bổ sung | Số ca | Mục đích |
| --- | ---: | --- |
| DEADLINE call | 10 | Khung yêu cầu đặt hạn, ID và `when` nguyên văn, cả hôm nay/ngày mai. |
| LIST call | 10 | LIST toàn bộ khi câu nhắc task đã hoàn thành. |
| Clarify/no-action | 10 | Thiếu ID/thời gian/scope, nhiều ID, phủ định, câu trích dẫn, yêu cầu lọc chỉ completed. |
| Action khác | 10 | Mỗi action CREATE/COMPLETE/EDIT/DELETE/DUE hai call để giảm lệch phân phối. |

Sau bổ sung: 450 call, 95 clarify, 95 no-action. DEADLINE/LIST mỗi action có 70 call; năm action còn lại mỗi action 62 call. Audit xác nhận tất cả `when` thuộc grammar hiện tại; các call giữ nguyên arguments từ prompt; không có câu trùng nguyên văn, trùng dạng đã mask hoặc có similarity ≥0,92 với development v2, final holdout v2 và diagnostic cặp vừa chạy. Đây chỉ là screening từ vựng, không chứng minh các họ câu độc lập. Chạy lại: `.venv/bin/python -m scripts.audit_sft_candidate_v4_draft`.

## Khả năng tương thích giữa nhãn và policy

Ở checkpoint `a1346ee`, khi đưa 30 gold call mới qua `policy_for_tool()` offline, chỉ **6 call** được cho phép, còn **24 call** bị từ chối/hỏi lại. Sau lát cắt DEADLINE/LIST có **22/30** call tương thích. Lát cắt tiếp theo xử lý tám khung còn lại; hiện **30/30** tương thích, trong đó DELETE chỉ đi đến bước hỏi xác nhận, chưa xóa task. Bảng nguyên nhân dưới đây mô tả trạng thái *trước* khi sửa:

Theo action, số ca bị chặn là DEADLINE 8/10, LIST 8/10, CREATE 1/2, COMPLETE 2/2, EDIT 2/2, DELETE 2/2 và DUE 1/2. Theo reason: `bare_statement` 17, `unsupported_action` 4, `content_boundary_mismatch` 2, `multiple_task_ids` 1. Các số này là kết quả chạy policy trên **bản nháp** và sẽ đổi nếu nhãn/câu được sửa sau review.

- `Cho tôi xem danh sách, kể cả các việc đã hoàn thành.` được gán LIST theo contract sản phẩm, nhưng policy trả `unsupported_action`.
- `Tôi làm xong task #864 rồi; cập nhật trạng thái.` được gán COMPLETE, nhưng policy trả `bare_statement`.
- `Đặt hạn việc 358 ở 8 giờ sáng mai, đúng ID đó.` được gán DEADLINE cho ID 358, nhưng policy đọc số 8 của giờ là ID thứ hai và hỏi lại.

Sửa policy theo grammar có ranh giới đã cho phép 20/20 call DEADLINE và LIST của bản nháp, đồng thời chặn `list_tasks` cho các câu lọc theo hạn trong các bộ regression được quét. Tám khoảng lệch CREATE 1, COMPLETE 2, EDIT 2, DELETE 2, DUE 1 được xử lý bằng khung câu đầy đủ. EDIT cho phép bỏ dấu kết câu ở yêu cầu tự nhiên nhưng giữ nguyên dấu câu sau `thành:`; ID, content, scope sai vẫn bị chặn. DELETE luôn yêu cầu xác nhận. Test `run_turn` với SQLite xác nhận LIST gồm task completed, DEADLINE đặt đúng ID và đề xuất sai ID không gây ghi dữ liệu.

| Ca | Ranh giới được công nhận | Chặn khi |
| --- | --- | --- |
| `v4_create_02` | `Lưu một task mới có nội dung:` là lời dẫn; phần sau dấu `:` là content nguyên văn. | Cắt ngắn hoặc đổi chữ hoa content. |
| `v4_complete_01` | `Đánh dấu việc ID đã hoàn thành` nêu đúng một ID. | ID proposal khác ID trong câu. |
| `v4_complete_02` | `Tôi làm xong task ID rồi; cập nhật trạng thái` có lời yêu cầu cập nhật rõ. | Chỉ kể đã làm xong, thiếu lời yêu cầu hoặc có phủ định. |
| `v4_edit_01` | Dấu chấm cuối câu yêu cầu tự nhiên có thể bỏ khỏi content. | Bỏ từ trong content hoặc đổi ID. |
| `v4_edit_02` | Sau `thành:` là content literal, giữ cả dấu chấm nếu có. | Cắt mất dấu câu sau dấu `:`. |
| `v4_delete_01` | `Xóa việc ID khỏi danh sách` là DELETE một ID. | Sai ID; proposal vẫn chỉ mở bước xác nhận. |
| `v4_delete_02` | `Bỏ hẳn task ID; ứng dụng hãy hỏi xác nhận` yêu cầu xóa có xác nhận. | Câu thay `hãy hỏi` bằng `đừng hỏi`, sai ID hoặc chưa xác nhận. |
| `v4_due_query_02` | `Cho tôi biết task nào đã quá hạn` mang scope `overdue`. | Proposal đổi sang `today`/`tomorrow` hoặc gọi LIST toàn bộ. |

Trong lúc kiểm tra false allow của LIST, tôi đã dùng prompt và nhãn của **final holdout v2** để quét policy. Vì vậy tập đó không còn là phép đo *mù cho policy layer* hoặc end-to-end system action. Nó vẫn chưa được gửi tới model, chưa đi vào dữ liệu train và có thể dùng cho phép đo **model proposal-only** theo protocol đã khóa. Một phép đo hệ thống mù trong tương lai cần bộ dữ liệu mới.

Gold nhãn được gán theo ý định và contract; **policy không được dùng làm đáp án chuẩn**. Codex và project owner đã review từng nhãn. Khả năng tương thích 30/30 không chứng minh model sẽ sinh đúng proposal, và cũng không chứng minh policy bao phủ mọi cách nói. Không nên huấn luyện model học cách từ chối một yêu cầu hợp lệ chỉ để khớp policy hiện tại.

## Gate tiếp theo

1. Cổng review local đã đạt 40/40; chạy lại bằng `.venv/bin/python -m scripts.audit_sft_candidate_v4_owner_review --require-complete`. Sidecar review chi tiết chỉ ở local, bị Git ignore.
2. [Preflight v4](sft_candidate_v4_offline_preflight.json) khóa hash dataset, builder, sidecar review và [development v3](model_proposal_development_v3.json). Audit: `.venv/bin/python -m scripts.audit_sft_candidate_v4_preflight`. Bộ development có 40 ca, chia 4 call cho mỗi action và 12 control, chưa đưa tới model. Xem [ghi chú development v3](README.development-v3.md).
3. Codex đã review nhãn development; đây là dữ liệu chẩn đoán, không phải holdout mù. Cần khóa runner và ngân sách riêng trước khi train hoặc inference. Preflight hiện đặt `training_authorized=false` và `development_inference_authorized=false`.
4. Final holdout v2 tiếp tục đóng đối với model. Nếu cần chấm end-to-end mù, phải tạo tập system holdout mới vì prompt/nhãn của v2 đã được xem khi kiểm tra policy.

Chưa thay model/prompt/routing mặc định, chưa chạy tokenizer hoặc GPU cho bản nháp này.
