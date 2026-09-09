# NEXUS

Phiên bản đầu: thêm việc và xem danh sách qua CLI, lưu bằng SQLite.
Mỗi dòng có nội dung tạo một việc; nội dung trùng vẫn tạo việc riêng.
CLI chưa dùng LLM hoặc xử lý lịch nhắc. Demo và đánh giá Gemini nằm riêng bên dưới.

Môi trường đã kiểm tra: Python 3.12.3, SQLite 3.45.1. Chỉ dùng thư viện chuẩn;
câu SQL `RETURNING` yêu cầu SQLite >= 3.35.

## Sử dụng

Chạy từ thư mục dự án:

```sh
python3 -m nexus.cli.main add "mua sữa, gọi mẹ và học Python"
python3 -m nexus.cli.main list
```

Dòng ví dụ trên tạo đúng một việc. Để nhập nhiều việc:

```sh
python3 -m nexus.cli.main add
```

Nhập hoặc dán mỗi việc trên một dòng. Trên Linux, kết thúc bằng Ctrl+D ở đầu
dòng mới. Chương trình đọc hết đầu vào rồi mới lưu. Dòng trống hoặc chỉ có
khoảng trắng được bỏ qua; nội dung các dòng còn lại được giữ nguyên.

Cũng có thể đọc từ file UTF-8:

```sh
python3 -m nexus.cli.main add < tasks.txt
```

Mặc định dữ liệu nằm trong `nexus.db` cạnh `cli.py`, không đổi theo thư mục
đang chạy. Có thể chỉ định file khác, đặt `--db` trước lệnh con:

```sh
python3 -m nexus.cli.main --db /tmp/nexus-demo.db add "học Python"
python3 -m nexus.cli.main --db /tmp/nexus-demo.db list
```

Thư mục cha của database phải tồn tại. Mỗi việc được commit riêng: nếu lỗi
giữa chừng, các việc đã lưu trước đó vẫn còn. Xem danh sách trước khi thử lại
để tránh thêm trùng ngoài ý muốn.

## Kiểm tra

```sh
PYTHONPATH=src python3 -B -m unittest discover -s tests -v
```

Test dùng database tạm; các test CLI chạy thêm và xem trong những tiến trình riêng.

## Demo Gemini tool calling

`gemini_demo.py` thử một yêu cầu cố định “Thêm việc mua sữa vào danh sách giúp tôi.”
với `gemini-2.5-flash`. Dùng database tạm được dọn khi kết thúc, không ghi vào
`nexus.db`. Demo gửi prompt và schema tool tới Google, rồi gửi kết quả tool để
model trả lời; tối đa hai yêu cầu generateContent, không tự retry.

Nếu đã đặt biến môi trường `GEMINI_API_KEY`:

```sh
python3 gemini_demo.py
```

Hoặc nhập key ẩn, chỉ giữ trong môi trường của tiến trình demo:

```sh
python3 -c 'import getpass, os; from gemini_demo import main; os.environ["GEMINI_API_KEY"] = getpass.getpass("Gemini API key: "); raise SystemExit(main())'
```

Các test trong `tests/test_gemini_demo.py` dùng phản hồi giả lập, không gọi mạng
hoặc dùng key. Demo thật cần mạng và quyền/quota generateContent của key.

## Đánh giá Gemini trên 5 tình huống

`evals/basic_tasks.json` định nghĩa prompt, dữ liệu ban đầu, tool call và dữ liệu
cuối mong đợi. Mỗi ca dùng database tạm riêng. Khi đã đặt `GEMINI_API_KEY`:

```sh
python3 eval_gemini.py --output /tmp/nexus-eval-run-1.json
```

File output phải chưa tồn tại. Bộ chạy lưu kết quả sau từng ca và dừng khi có
lỗi API/thực thi. Response có finishReason không hợp lệ được ghi thành lỗi
model, rồi tiếp tục ca độc lập kế tiếp. Có thể chọn những ca còn lại:

```sh
python3 eval_gemini.py --output /tmp/nexus-eval-run-2.json --case greeting --case missing_content
```

`pass` chỉ chấm tự động tool và dữ liệu, chưa chấm ngữ nghĩa câu trả lời.
`fail` là lượt hoàn tất nhưng sai kỳ vọng; `error` là lượt không hoàn tất.
Cả khi lỗi, report vẫn ghi các call quan sát được và trạng thái database cuối.
Các test local của bộ chấm dùng phản hồi giả lập, không gọi API.

## So sánh prompt

`gemini_demo.py` giữ prompt gốc `v1` và bản thử nghiệm `v2` làm rõ quy tắc
một dòng/một task. Mặc định vẫn là `v1`. Chỉ system instruction thay đổi;
model, tool schema và code thực thi giống nhau.

```sh
python3 eval_gemini.py --cases evals/prompt_comparison_cases.json --prompt-version v2 --request-interval 15 --output /tmp/nexus-prompt-v2.json
```

Dataset so sánh gồm 5 ca cũ và 2 câu mới, được viết trước khi chạy thử.
Khoảng nghỉ giúp giảm tần suất gọi API, không bảo đảm đủ mọi quota và không
retry request thất bại. `elapsed_seconds` bao gồm cả thời gian nghỉ này,
không dùng để so tốc độ model giữa các report có khoảng nghỉ khác nhau.
Report lưu nguyên system prompt, phiên bản prompt và metadata response
(finishReason, số candidate, blockReason, modelVersion).
