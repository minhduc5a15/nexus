# Chẩn đoán cặp cách diễn đạt: base so với QLoRA step 300

Đây là **diagnostic đã lộ**, không phải final holdout. Bộ 20 câu gồm 10 cặp; hai câu trong mỗi cặp có cùng hành động và gold arguments nhưng khác khung diễn đạt. Năm câu A tái sử dụng lỗi đã đọc ở development v1/v2. Không dùng kết quả này để tuyên bố generalization hoặc chọn checkpoint bằng final holdout.

## Provenance và cách chạy

- Dataset: [`model_proposal_framing_diagnostic_v1.json`](model_proposal_framing_diagnostic_v1.json), SHA-256 `16fec3a3cb1908ccc708e2eedfa6e50c21e416949893b450cd22111aba27d5d6`.
- Run plan: [`model_proposal_framing_diagnostic_v1_run_plan.json`](model_proposal_framing_diagnostic_v1_run_plan.json), SHA-256 `8b9777bd777884a6381a93326c3d3eface8ac115d3fedaaf08c6000cc3532a0d`.
- Raw report: `evals/results/framing-base-step300-20261021/sft-framing-diagnostic-v1/diagnostic.json` (Git ignored), SHA-256 `b88644e609e5cd900aa89b57a079f462a455229f6e24dcdb35dfa2a04e28967f`.
- Application checkpoint trước chẩn đoán: `a1346ee`. Plan khóa hash dataset, runner, scorer, prompt và tool schema; các module dựng request còn lại được xác định bởi checkpoint này.
- Qwen3-4B-Instruct-2507 revision khóa trong plan; base NF4 so với cùng base + adapter step 300. Prompt v13, routing `all`, greedy decoding, seed 42, tối đa 128 token. Cùng renderer và scoring; chỉ sinh proposal, không gọi policy/tool/SQLite.
- Chạy lại audit offline: `.venv/bin/python -m scripts.audit_model_proposal_framing_diagnostic_v1` và `.venv/bin/python -m scripts.audit_model_proposal_framing_diagnostic_v1_report`. Runner Modal: `.venv/bin/modal run modal_sft_framing_diagnostic_v1.py --stage preflight`; `--stage run` dùng GPU và từ chối run ID đã có.

## Kết quả

| Nhóm | Base | Step 300 |
| --- | ---: | ---: |
| Toàn bộ | 20/20 | 13/20 |
| DEADLINE | 12/12 | 8/12 |
| LIST | 8/8 | 5/8 |

Cả bảy lỗi step 300 đều là `missing_call`: model không gọi tool, rồi trả lời như thể thiếu dữ liệu hoặc ứng dụng không hỗ trợ yêu cầu. Không có lỗi đổi ID/`when` trong những ca đã gọi tool. Base đúng cả 20 câu, nhưng tập này được chọn vì đã biết một số lỗi của candidate, nên khoảng cách 20/20 so với 13/20 không đại diện cho toàn bộ workload. Trên development v2 cân bằng 80 ca, step 300 từng tăng từ 67/80 lên 78/80; hai kết quả cần đọc cùng nhau.

| Cặp | Câu A step 300 | Câu B step 300 | Quan sát |
| --- | --- | --- | --- |
| deadline 01–03 | 3/3 lỗi | 3/3 đúng | Cùng ID/thời gian; khung “Đặt hạn việc … lúc …” bị bỏ, khung khác gọi đúng. |
| deadline 04 | đúng | đúng | Thay ID/thời gian mới không khiến mọi câu có khung A đều lỗi. |
| deadline 05 | lỗi | đúng | Với `21:40 hôm nay`, đổi từ ID-trước sang thời-gian-trước phục hồi call. |
| deadline 06 | đúng | đúng | Khung pha tiếng Anh vẫn có thể gọi đúng. |
| list 01 | lỗi | lỗi | Đổi từ “danh mục” sang “liệt kê mọi việc” vẫn chưa đủ. |
| list 02 | lỗi | đúng | Một biến thể “đọc lại toàn bộ task” phục hồi call. |
| list 03–04 | 2/2 đúng | 2/2 đúng | Không phải mọi câu nhắc task đã xong/completed đều lỗi. |

Ví dụ lỗi cần lưu ý:

- `Đặt hạn việc 6 lúc 8 giờ 30 phút tối ngày mai.` → “hãy cung cấp ID”, dù ID `6` đã có.
- `Đặt hạn việc 1104 lúc 21:40 hôm nay.` → model nói không hỗ trợ deadline theo “hôm nay” và yêu cầu `today/tomorrow/overdue`. Đây là nhầm `set_task_deadline` với phạm vi của `list_tasks_by_deadline`; parser thời gian thực tế chấp nhận `21:40 hôm nay`.
- `Mở danh mục việc, tính cả những mục đã hoàn thành.` → model nói chỉ xem task chưa hoàn thành. Contract `list_tasks` thực tế hiển thị toàn bộ task, kể cả completed; chỉ `list_tasks_by_deadline` lọc task chưa hoàn thành.

Kiểm tra dữ liệu SFT v3 thấy 60 ví dụ call DEADLINE và 60 call LIST, nên đây không phải thiếu toàn bộ action. Trong 60 DEADLINE call không có prompt chứa đúng cụm “đặt hạn việc”, có 4 câu chứa “đặt hạn cho việc”. Trong 60 LIST call có 4 câu chứa “đã xong”, không có “đã hoàn thành” hoặc “completed”. Đây là **dấu hiệu về coverage cách diễn đạt**, chưa chứng minh nguyên nhân huấn luyện hay tác động của một ví dụ cụ thể. Các phản hồi sai về khả năng ứng dụng cũng cần được kiểm tra bằng dữ liệu đối chứng trước khi sửa nhãn.

## Quyết định tại mốc này

Chưa mở final holdout v2, chưa huấn luyện tiếp, chưa đổi prompt/tool schema/routing hoặc model mặc định. Lát cắt kế tiếp nên tạo một bản bổ sung dữ liệu nhỏ, có đối chứng: các call DEADLINE/LIST ở đúng khung còn hụt, các cách nói gần giống nhưng thiếu ID/thời gian hoặc không yêu cầu action, và regression của năm action khác. Chọn cấu hình bằng development; nếu cải thiện nhưng làm giảm action khác, không đưa vào ứng dụng. Cần đánh giá trên câu mới tách theo họ diễn đạt trước khi mở final holdout.
