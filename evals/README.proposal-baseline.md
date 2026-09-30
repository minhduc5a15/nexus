# Baseline development proposal — protocol 30/09/2026

Checkpoint trước inference: `202cc79`. Development/holdout không thay nhãn.
Lượt chạy này chỉ dùng development 80 ca và nhóm lặp 14 ca; không gửi holdout tới model.
Không thực thi tool, policy authorization hoặc SQLite. Prompt/schema giữ nguyên.

## Khối lượng và thứ tự đã chốt

- Development: v13/all 80 ca, sau đó v13/classified 80 ca, theo thứ tự dataset.
- [Nhóm lặp](model_proposal_repeats_v1.json): 14 ca, ba vòng mỗi mode (84 lượt).
- Vòng 1: all → classified; vòng 2: classified → all; vòng 3: all → classified.
- Tổng **244 lượt chấm** và **8 cold probe riêng**, không retry ca sai.

Nhóm lặp chứa 12 ca development cùng `regression_edit` và `create_duplicate` lấy
từ diagnostic lịch sử. ID và thứ tự cụ thể nằm trong JSON; gold đã chốt trước run.
Duplicate CREATE chỉ dùng input `Thêm việc mua sữa.` và proposal create_task tương ứng.
Database có task trùng của thí nghiệm lịch sử **không** được gửi cho model. Không
được từ kết quả proposal này suy ra hệ thống xử lý lưu trùng đúng hoặc model biết
những task đã lưu. Muốn kiểm tra điều đó cần evaluator ứng dụng riêng.

## Cold, warm và backend

NVIDIA driver không hoạt động ở preflight, nên toàn bộ phép so sánh dùng CPU:
llama.cpp build 10809 / commit 5266f24da, GPU layers 0, threads 6, context 4096,
parallel 1; Qwen3-4B-Instruct-2507 Q4_K_M. Không so trực tiếp với latency GPU cũ.
Temperature 0, seed 42, top_p 0.8, top_k 20, min_p 0, max_tokens 256,
enable_thinking false. Timeout transport 300 giây được chốt trước inference vì
CPU xử lý prompt chậm; không thay sampling hoặc request để giúp model đạt.

Mỗi block khởi động server mới. Request inference đầu luôn là ca `list_direct`;
lưu trong file `*-cold.json` và loại khỏi điểm chất lượng/metrics warm. Tiếp theo
chạy chuỗi được chấm liên tục, không xóa cache giữa các ca. Warm nghĩa là server
đang chạy, không bảo đảm schema mới có cache hit. Nhóm lặp là chuỗi action hỗn hợp
cố định để thấy chi phí đổi schema; development vẫn theo thứ tự dataset đã khóa.

Có bốn cold probe mỗi mode, không đủ để ước lượng p95 production. Ba vòng lặp
cho tín hiệu ổn định, không chứng minh temperature 0 luôn tái lập tuyệt đối. Ba
vòng là số lẻ nên thứ tự đầu tiên được luân phiên nhưng không cân bằng hoàn hảo.
Tải máy/nhiệt độ chưa được cô lập; load average được ghi sau mỗi request.

## Evidence và review

Protocol, hash, runner, raw request/response từng ca, server props/chat template,
log, model hash và source hash được giữ dưới
`evals/results/development-baseline-20260930/` (Git ignore). `protocol.json` và
`protocol.sha256` đã được tạo trước inference. Runner tự dừng server giữa block,
không ghi đè hoặc âm thầm tiếp tục run cũ; lỗi generation được lưu rồi dừng để
review hạ tầng. Có thể xem tiến độ từ `run-state.json` hoặc `run.log`.

Chấm tự động và review lời trả lời tách biệt theo
[hướng dẫn review](README.proposal-data.md). Chỉ tạo sidecar cho report đã hoàn tất
để hash không đổi giữa chừng. Mỗi nhận xét là review của assistant, chưa có người
review độc lập. Không dùng lỗi development để sửa prompt hoặc nới policy trong
lượt baseline này.

## Kết quả đã chạy

Run hoàn tất đủ 244/244 lượt, không có generation error. Proposal exact trên
development là **65/80 (81,25%)** với `all` và **62/80 (77,50%)** với
`classified`. `classified` giúp hai ca EDIT từ fail thành pass, nhưng làm năm ca
đang pass thành fail; vì vậy chưa đủ điều kiện rollout và mặc định vẫn là v9/all.

Ba vòng lặp cho kết quả giống nhau ở mọi ca: `all` đạt 10/14 mỗi vòng,
`classified` đạt 11/14 mỗi vòng. Routing sửa ổn định `regression_edit`, nhưng cả
hai mode đều trượt ổn định ở `edit_literal`, `bare_statement` và
`ambiguous_today`. Đây là tín hiệu lặp lại trên nhóm nhỏ, không phải ước lượng
generalization.

Trên 42 lượt lặp mỗi mode, median warm là 5,085 giây (`all`) và 4,861 giây
(`classified`); prompt token median giảm từ 1.287 xuống 599. Development bị gián
đoạn giữa hai mode nên số latency của hai block 80 ca không được dùng làm bằng
chứng nhân quả. Nhóm lặp sau recovery phù hợp hơn để mô tả hiệu năng, nhưng tải
máy và nhiệt độ vẫn chưa được cô lập.

Runner ban đầu dừng sau 14 ca `development-classified`. Partial report được đóng
dấu `infra-interrupted-*`, không trộn vào điểm cuối; toàn bộ block classified
được chạy lại từ server mới. Vì vậy 14 prompt đầu của block này đã được inference
hai lần, dù chỉ run khôi phục được chấm. Sai lệch protocol và hash của partial
được giữ trong `INTERRUPTION.md` cùng `interrupted.sha256`.

Review lời trả lời của assistant (không phải review độc lập của con người) đạt
70/80 ở `all` và 69/80 ở `classified`. Các lỗi chính là hỏi lại dữ liệu đã có,
không hỏi lại khi model đã gọi tool sai, và ví dụ thời gian nằm ngoài grammar.
Điểm review này không thay đổi điểm proposal.

Smoke ứng dụng mặc định v9/all sau baseline đạt 7/7 case, 9/9 lượt và không có
mutation ngoài yêu cầu. Đây chỉ là regression nhỏ của ứng dụng hiện hành; nó
không bù cho 15–18 lỗi proposal trên development v13.

Báo cáo diễn giải đầy đủ nằm ở
`reports/model-proposal-development-baseline-2026-09-30.md` (thư mục local bị
Git ignore). Final holdout 120 ca vẫn chưa được gửi tới model.

## Chạy lại trên Modal T4 và thử Q8_0

Ngày 30/09/2026, cùng development/repeats đã khóa được chạy lại bằng
`modal_benchmark.py` trên Tesla T4 15 GB. llama.cpp vẫn ở commit
`5266f24da75dc449bd56cbed7addb9c8e4a6a73e`; prompt v13, routing, sampling,
thứ tự ca và hash dataset giữ nguyên. Hai model được khóa theo revision và
SHA-256: Q4_K_M `3605803b...c67e597`, Q8_0 `391c1e41...c4ef068`.
Final holdout vẫn chưa được gửi tới model.

| Backend/model | v13/all | v13/classified | Median warm all | Median warm classified |
|---|---:|---:|---:|---:|
| CPU Q4_K_M lịch sử | 65/80 | 62/80 | 5,330 s | 5,215 s |
| Modal T4 Q4_K_M | 66/80 | 63/80 | 0,398 s | 0,509 s |
| Modal T4 Q8_0 | 67/80 | 63/80 | 0,561 s | 0,705 s |

Cả Q4 và Q8 hoàn tất 244/244 lượt, không có generation error. Ba vòng lặp
đều giữ nguyên `all=10/14`, `classified=11/14`. Smoke ứng dụng v9/all đạt
7/7 case, 9/9 lượt với cả hai quantization và không có mutation ngoài yêu cầu.
Q8 chỉ tăng một ca development ở `all`, không tăng `classified`, trong khi
median chậm hơn Q4 khoảng 39–41%. Bộ development đã được xem nên chênh lệch
này là diagnostic, chưa phải bằng chứng generalization hay lý do đổi model mặc định.

Artifact raw, metadata, server log và smoke report nằm ở
`evals/results/modal-gpu-20260930-downloaded/` (Git ignore). Báo cáo diễn giải
nằm ở `reports/modal-qwen3-4b-quantization-baseline-2026-09-30.md` (Git ignore).

## Audit lỗi trước SFT

Audit tracked tại `model_proposal_error_audit_v1.json` bao phủ đúng 19 case từng
fail ở ít nhất một trong bốn cấu hình Modal, tổng cộng 61 failure observation.
Mỗi failure có đúng một primary cause, raw observation gắn với SHA-256 của report
và quyết định dùng cho SFT, regression guard hoặc sửa kiến trúc.

Kết quả: 52/61 failure observation thuộc proposal model, 9/61 liên quan routing;
14 case là mục tiêu SFT, ba case là regression guard và hai case (`dev_061`,
`dev_064`) chỉ thuộc classifier expose sai tool. 19/19 nhãn được đối chiếu với
contract hiện hành; đây là review của assistant, chưa phải review độc lập của
người thứ hai. Prompt, policy, routing và dataset không bị sửa trong audit.

Chạy lại kiểm tra bằng:

```sh
./scripts/demo/25_model_error_audit.sh
# hoặc
.venv/bin/python -m scripts.audit_model_errors
```

Lệnh chỉ đọc development report cục bộ và không gọi model, tool hay SQLite.
Final holdout vẫn chưa được mở.
