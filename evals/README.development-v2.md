# Development v2: kiểm tra hồi quy DEADLINE của SFT v3

Đây là một mốc **offline** trước khi sửa dữ liệu training. Không có inference trên final holdout v2, không train thêm và không đổi model đang triển khai. Bộ `model_proposal_development_v2.json` là development diagnostic: dùng để so sánh các candidate, không được gọi là phép đo mù. Nhãn do Codex soạn từ contract sản phẩm; chủ dự án chưa duyệt thủ công.

## Bằng chứng đã có

Trong development v1, base đúng 8/8 ca DEADLINE, adapter step-300 đúng 5/8. Cả ba lỗi `deadline_relative`, `dev_057`, `dev_058` là **không gọi tool** dù câu đều có ID và cụm thời gian hợp lệ. Chúng lần lượt yêu cầu `8 giờ 30 phút tối ngày mai`, `7 giờ sáng hôm nay` và `18:15 ngày mai`. Step-300 vẫn đạt 77/80 toàn bộ và không có gọi tool ngoài yêu cầu trong bộ cũ. Đây là hồi quy ở một action, chưa phải bằng chứng rằng model mất khả năng xử lý mọi deadline tương đối.

Trong 600 mẫu training v3 có 60 proposal DEADLINE và 18 ca DEADLINE cần hỏi lại. Sáu mươi proposal chỉ dùng 20 giá trị `when` khác nhau. Riêng 32 ca được kế thừa từ candidate v2 lặp 4 giá trị `when`, mỗi giá trị 8 lần, trong các khung câu khác nhau. Việc lặp này **có thể** làm model học khung câu tốt hơn khả năng nhận diện tổ hợp ID + thời gian mới; dữ liệu hiện tại không xác định được nguyên nhân nhân quả. Các ca hỏi lại khi thiếu ID hoặc thiếu giờ cũng có thể góp phần khiến model hỏi quá mức, nhưng chưa có ablation chứng minh.

## Phép đo tiếp theo

Development v2 có 80 ca: 56 yêu cầu hành động (8 ca cho mỗi action CREATE, LIST, COMPLETE, EDIT, DELETE, DEADLINE, DUE QUERY), 12 ca hỏi lại và 12 ca không hành động. Trong 8 ca DEADLINE có 6 cụm thời gian tương đối và 2 cụm tuyệt đối. Các action khác giữ vai trò đối chứng: một lần sửa DEADLINE chỉ có ích nếu không làm chúng đi xuống hoặc tạo tool call sai trong nhóm control.

Mỗi ca có action, arguments và missing slots độc lập với classifier/policy. Audit kiểm tra schema, cân bằng, content/when nguyên văn, grammar thời gian và trùng lặp. Sau khi che ID và payload, không có ca trùng chính xác hoặc cặp nào đạt độ tương tự chuỗi 0,85 với training v3, development v1 hoặc final holdout v2. Đây là **sàng lọc từ vựng**, không chứng minh độc lập ngữ nghĩa. Final holdout v2 chưa được đưa vào model. Một phần builder của holdout đã được người soạn đọc khi kiểm tra cấu trúc dữ liệu; vì vậy không thể tuyên bố người soạn chưa từng xem nội dung holdout.

Chạy lại audit:

```sh
.venv/bin/python -m scripts.build_model_proposal_development_v2
.venv/bin/python -m scripts.audit_model_proposal_development_v2 --verify-manifest
.venv/bin/python -m unittest tests.test_model_proposal_development_v2
```

Manifest khóa hash dataset, builder, training v3, development v1, final holdout v2 và bản review development v3. Raw report development v3 trong `evals/results/` được đối chiếu hash nếu còn trên máy; bản review tracked đủ để audit chạy lại trong checkout sạch.

## Kết quả chạy base và step-300 trên v2

Run `dev2-base-step300-20261021b` dùng Qwen3-4B-Instruct-2507 revision đã khóa, prompt v13, routing all, NF4 base, adapter step-300, greedy decoding, seed 42 và tối đa 128 token. Raw report nằm ở `evals/results/dev2-base-step300-20261021b/sft-development-v2/development.json` (Git ignore), SHA-256 `558f794af0e5b5517973107fbd2fc5fde0cdfaa3ad54d22825f0446a224b48c8`. Runner chỉ tạo proposal; không chạy policy, tool, session hoặc SQLite. Lần thử `...20261021a` dừng ở lỗi import của runner trước khi nạp model, không có inference case nào.

| Phép đo | Base | Step-300 |
| --- | ---: | ---: |
| Proposal tự động đúng | 67/80 | 78/80 |
| CREATE | 6/8 | 8/8 |
| LIST | 8/8 | 6/8 |
| COMPLETE | 6/8 | 8/8 |
| EDIT | 8/8 | 8/8 |
| DELETE | 8/8 | 8/8 |
| DEADLINE | 8/8 | 8/8 |
| DUE QUERY | 6/8 | 8/8 |
| Gọi tool ngoài yêu cầu ở 24 ca control | 7 | 0 |
| Ca control sai ngữ nghĩa, kể cả tool call sai | 13/24 | 4/24 |

Trên Modal T4, median thời gian sinh mỗi ca là 3,16 giây cho base và 4,14 giây cho step-300; p95 lần lượt 5,17 và 5,97 giây. Tổng prompt token của hai bên giống nhau (103.706), nhưng step-300 sinh ít token hơn (1.828 so với 2.398). Đây là một run chẩn đoán trên HF/NF4, không phải phép đo latency triển khai GGUF trên laptop.

Hai lỗi LIST của step-300 là `dev2_list_05` và `dev2_list_06`: người dùng yêu cầu danh sách gồm cả mục đã xong; model không gọi `list_tasks` và nói sai rằng NEXUS chỉ xem task chưa hoàn thành. `list_tasks` trong ứng dụng thực sự trả toàn bộ task. Đây là hồi quy chức năng và lỗi mô tả năng lực, dù không gây mutation.

Step-300 đúng cả sáu ca DEADLINE tương đối mới. Vì vậy ba lỗi DEADLINE cũ **không** cho thấy lỗi tổng quát với mọi thời gian tương đối. Giả thuyết hợp lý hơn là độ nhạy với cách đặt câu và vị trí ID/thời gian; điều này vẫn cần phép kiểm tra cặp câu có kiểm soát trước khi quyết định dữ liệu train v4.

Review ngữ nghĩa do **Codex**, không phải người review độc lập, thực hiện trên toàn bộ 24 ca control cho mỗi cấu hình. Trong bốn lỗi còn lại của step-300: một câu tự suy task chưa được tạo khi chỉ thiếu ID; một câu đòi lại thời gian đã nêu trong câu phủ định; một câu gợi ý LIST từ lời kể trích dẫn; một câu gợi ý CREATE từ ghi chú không yêu cầu hành động. [Quyết định từng ca](model_proposal_development_v2_semantic_review.json) gắn hash raw report. Kết quả này là diagnostic, không phải điểm final holdout.

Kiểm tra lại report và review:

```sh
.venv/bin/python -m scripts.audit_model_proposal_development_v2_report --semantic-review
.venv/bin/python -m unittest tests.test_model_proposal_development_v2_report
```

## Cổng trước training v4

1. Hoàn tất review độc lập của nhãn v2 nếu muốn dùng bộ này làm cổng chọn candidate. Nhãn hiện do Codex soạn và audit; inference đã diễn ra, nên mọi sửa nhãn sau đây phải có lý do riêng và không được âm thầm đổi đáp án để tăng điểm.
2. Kiểm tra có kiểm soát các cặp framing: giữ nguyên ID và `when`, chỉ đổi cách nói giữa ba câu DEADLINE cũ bị miss và các khung mới được gọi đúng; đồng thời kiểm tra LIST “gồm cả việc đã xong” với vài cách nói khác. Đây là diagnostic đã lộ, không phải final holdout.
3. Chỉ thiết kế dataset train v4 sau khi phân loại các lỗi ở bước 2. Nếu lỗi chỉ do vài khung câu, thêm mẫu train đa dạng nhưng giữ đối chứng của sáu action khác và control; nếu phát hiện lỗi render/integration, sửa đường triển khai trước. Không quyết định trước số ví dụ hoặc tỷ lệ tăng.
4. Candidate mới phải so với step-300 theo từng action, lời trả lời control và safety. Development v2 đã được xem để thiết kế training nên không còn là bằng chứng mù. Final holdout v2 vẫn đóng theo protocol riêng.

Hiện chưa thể khẳng định thêm dữ liệu DEADLINE sẽ giải quyết vấn đề. Kết quả v2 còn cho thấy cần bảo vệ contract LIST trong mọi lần cải thiện model.
