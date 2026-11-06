# Development v3 cho SFT v4

[`model_proposal_development_v3.json`](model_proposal_development_v3.json) là bộ **diagnostic 40 ca**, SHA-256 `2121c258d8acf18cf877330fa922d91f61184baf1c7679c128006a3f621bc12d`. Nó chưa được gửi tới model. Codex soạn và review nhãn theo contract sản phẩm, không lấy classifier, policy hoặc output của model làm đáp án. Chủ dự án chưa duyệt riêng bộ này; nếu phát hiện nhãn tranh cãi, sửa trước khi inference và khóa hash mới. Không đưa các ca này vào training.

| Nhóm | Số ca | Khung câu cần kiểm tra |
| --- | ---: | --- |
| CREATE, LIST, COMPLETE, EDIT, DELETE, DEADLINE, DUE | 4/action | Mỗi action có hai khung câu; mỗi khung có hai ID hoặc payload khác nhau. |
| Clarify | 6 | Thiếu `content`, `id`, `when` hoặc `scope`. |
| No-action | 6 | Trích dẫn, phủ định, bộ lọc/reminder chưa hỗ trợ và câu kể. |

DELETE ở đây chỉ chấm **proposal**. Ứng dụng vẫn phải hỏi xác nhận trước khi xóa; evaluator không được gọi policy, tool, session hoặc SQLite. Với CREATE/EDIT, `content` phải là đoạn nguyên văn trong prompt. Với DEADLINE, `when` phải nguyên văn và parser Python nhận được. Với các action theo ID, ID phải đúng số người dùng nêu. Câu no-action/clarify không được gọi tool; review ngữ nghĩa phản hồi vẫn cần thực hiện sau inference.

Mỗi cặp trong cùng action chia một `family` để nhìn được độ nhạy với ID/payload trên một khung. Các khung đã được viết riêng thay vì thay dữ liệu trong 40 câu train v4. Audit sàng lọc với 640 câu train v4, development v1/v2, diagnostic cặp và final holdout v2: không trùng nguyên văn, không trùng sau khi mask ID/payload và không có cặp nào đạt similarity ≥0,85. Đây là kiểm tra bằng chuỗi, **không chứng minh độc lập ngữ nghĩa**. Final holdout v2 không được inference trong bước này và đã lộ cho policy layer trước đó, nên không dùng để tuyên bố end-to-end mù.

Chạy audit offline:

```sh
.venv/bin/python -m scripts.build_model_proposal_development_v3
.venv/bin/python -m scripts.audit_model_proposal_development_v3
.venv/bin/python -m scripts.audit_sft_candidate_v4_preflight
```

Mốc [preflight v4](sft_candidate_v4_offline_preflight.json) khóa dataset train, sidecar duyệt local, development này, base revision, baseline v3 step 300 và contract proposal-only. Thiết kế giữ hai lượt qua 640 ví dụ, effective batch 4 nên là 320 optimizer step, bắt đầu từ base thay vì tiếp tục adapter v3. Sidecar duyệt chi tiết nằm ngoài Git. [Runner v4](../modal_sft_candidate_v4.py) và [execution plan](sft_candidate_v4_execution_plan.json) đã được viết và kiểm tra offline. Ngày 03/11/2026, chủ dự án đã duyệt riêng train và development trên Modal theo mức dự toán 2,60 USD; final holdout vẫn đóng. Base/v3-step-300/v4 dùng **cùng** renderer, prompt v13, routing `all` và decoding greedy.

## Kiểm tra token và tài nguyên trước GPU

Tokenizer của đúng Qwen revision đã khóa được nạp từ cache local bằng Transformers 4.57.1. Cả 640/640 ví dụ có assistant-only loss mask hợp lệ; tổng token được supervise là 16.050. Độ dài toàn bản ghi: nhỏ nhất 1.296, trung vị 1.318, dài nhất 1.368 token; không ví dụ nào vượt 2.048 token. Vì vậy không phải cắt mất câu trả lời đích để đưa vào training. Kiểm tra lại bằng:

```sh
.venv/bin/python -m scripts.inspect_sft_tokens --dataset evals/sft_candidate_v4_draft.json --case v4_deadline_02 --all-cases --summary-only --tokenizer /home/minhduc5a12/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
```

Theo [bảng giá Modal](https://modal.com/pricing) kiểm tra ngày 03/11/2026, T4 là 0,000164 USD/giây, CPU 0,0000131 USD/core/giây và RAM 0,00000222 USD/GiB/giây. Thiết kế giữ 1 T4, 2 CPU core, 16 GiB RAM, timeout train 8.400 giây và development 1.800 giây. Nếu cả hai dùng hết timeout, phép tính tài nguyên là `(8.400 + 1.800) × (0,000164 + 2 × 0,0000131 + 16 × 0,00000222) = 2,302344 USD`. V3 đã hoàn tất 300 step trong khoảng 5.987 giây; 8.400 giây là timeout có dự phòng cho v4 320 step, không phải dự báo runtime chính xác. Mức **2,60 USD** trong preflight chỉ là ngân sách *đề xuất*, chưa được duyệt; chi phí thực tế còn có thời gian khởi động/idle và các khoản ngoài phép tính này. Chưa có runner GPU hay cơ chế thực thi giới hạn chi phí, nên không coi số này là hard cap.

Runner v4 chỉ đóng gói source và dataset cần cho từng stage; không đóng gói web review, sidecar duyệt local hoặc final holdout. Train lưu checkpoint mỗi 25 step và report cùng adapter trên Modal volume. Development giữ phản hồi thô từng case, ghi partial sau mỗi case; [report audit](../scripts/audit_sft_candidate_v4_report.py) kiểm tra lại hash adapter, thứ tự 40 case và tính lại scoring từ phản hồi thô sau khi tải về. Kết quả lời đáp không gọi tool còn phải được review ngữ nghĩa; điểm automatic chưa thay thế bước này.

Kiểm tra offline trước khi cân nhắc chạy GPU:

```sh
.venv/bin/python -m scripts.audit_sft_candidate_v4_execution
.venv/bin/python -m unittest tests.test_sft_candidate_v4_execution tests.test_sft_candidate_v4_report
```

Mức 2,60 USD là ngân sách được duyệt cho train và development, không phải hard cap do Modal thực thi. Runner giữ timeout 8.400 và 1.800 giây, không tự retry. Final holdout v2 vẫn đóng.

Lần train đầu `sft-candidate-v4-train-20261103a` đã nạp model và chạy vài optimizer step, sau đó CLI mất DNS tới Modal. App dừng lúc 23:23 ngày 03/11/2026; state trên volume ghi `InputCancellation` lúc 23:22:10, chưa lưu checkpoint 25 step và chưa có adapter. Đây là lần hỏng do hạ tầng, không dùng làm kết quả model. Không xóa artifact của lần `a`.

Lần `sft-candidate-v4-train-20261103b` dùng `--detach` nhưng launcher vẫn gọi `Function.remote()`. Việc gửi `Ctrl+C` cho local entrypoint lúc 06:39:49 ngày 04/11/2026 đã phát cancellation tới input đồng bộ; run dừng ở step 6/320, chưa có checkpoint hoặc adapter. Runner sau đó chuyển sang `Function.spawn()` theo contract fire-and-forget của Modal; run ID hợp lệ tiếp theo dùng hậu tố `20261104c`. Audit từ chối nếu launcher quay lại `.remote()`.

Run bất đồng bộ `sft-candidate-v4-train-20261104c` chạy tới giới hạn timeout 8.400 giây và lưu checkpoint hoàn chỉnh ở step 275/320. State kết thúc lúc 14:58:23 ngày 04/11/2026 với `InputCancellation`; không có adapter cuối. Run `sft-candidate-v4-train-20261106d` resume từ checkpoint 275, khóa SHA-256 trainer state `002ae377...8b337b`, giữ optimizer/scheduler/RNG và chỉ chạy 45 step còn lại. Timeout train giảm còn 1.800 giây; development giảm còn 800 giây để giới hạn thêm chi phí sau các lần hỏng hạ tầng.

## Kết quả train và development v4

Run resume `sft-candidate-v4-train-20261106d` hoàn thành 320/320 step trong 1.037,8 giây. Report train có SHA-256 `12a93c961cb66e4cd14adc9c023afc6a60f1ce09d977dbaafe4a09069a0bf189`; adapter directory có SHA-256 `8fba7ddd77753f78246b9652fb9c3544aa6f3a5ed3e9da12dd13eaec6195f131`. State hash trước và sau phần resume khác nhau, adapter nạp lại khớp state hash sau train. Training loss do Trainer báo cho phần resume là `0,007961`; số này chỉ xác nhận quá trình tối ưu, không phải độ chính xác ứng dụng.

Development `sft-candidate-v4-development-20261106d` dùng đúng 40 ca, prompt v13, routing `all`, greedy decoding và cùng renderer. Raw report Git-ignore có SHA-256 `ae4a7df181223ec1cd16f3952a0cb7df848abdb94100a270fa4f6e06c92261eb`; audit tính lại điểm từ từng phản hồi thô đạt.

| Model | Automatic | Action | Clarify | No-action | Unexpected call | Semantic control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 37/40 | 26/28 | 5/6 | 6/6 | 1 | 8/12 |
| v3 step 300 | 40/40 | 28/28 | 6/6 | 6/6 | 0 | 12/12 |
| v4 step 320 | 40/40 | 28/28 | 6/6 | 6/6 | 0 | 12/12 |

Base sai hai ca EDIT và gọi DEADLINE khi người dùng chưa cung cấp thời gian. Review ngữ nghĩa còn phát hiện ba reply base đưa lựa chọn ngoài contract hoặc xử lý sai thứ tự xác nhận. [Semantic review](model_proposal_development_v3_semantic_review.json) đọc toàn bộ 12 control của mỗi model và không tìm thấy lỗi reply ở v3/v4; đây là review của Codex, chưa phải review người độc lập.

Median latency sinh một proposal trên Modal T4 là 3,22 giây cho base, 4,19 giây cho v3 và 4,19 giây cho v4; p95 lần lượt là 5,44, 6,22 và 6,14 giây. Cả ba nhận cùng 51.891 prompt token. Đây là latency HF/NF4 trên Modal, không đại diện cho GGUF trên laptop.

Kết quả cho thấy v4 giữ được mức 40/40 của v3 trên diagnostic mới; chưa cho thấy v4 tốt hơn v3. Bộ 40 ca đã được dùng để chọn và đọc kết quả nên là diagnostic, không phải final holdout. Final holdout v2 vẫn chưa được gửi tới model.
