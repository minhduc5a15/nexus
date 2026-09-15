# NEXUS

NEXUS là ứng dụng ghi nhanh việc cần làm bằng tiếng Việt. Phạm vi hiện tại chỉ
gồm thêm việc và xem danh sách. Người dùng có thể gọi trực tiếp bằng CLI hoặc
viết yêu cầu tự nhiên để Qwen3-1.7B chọn một trong hai tool. Dữ liệu được lưu
cục bộ bằng SQLite.

Mỗi dòng có nội dung tạo một task. Dấu phẩy và chữ `và` trong cùng một dòng
không tự tách task; các dòng trống bị bỏ qua. Mỗi task hiện chỉ có `id` và
`content`.

## Cài ứng dụng

NEXUS yêu cầu Python 3.10 trở lên, cùng SQLite 3.35 trở lên để dùng câu lệnh
`RETURNING`. Ứng dụng không có thư viện Python bên thứ ba. Từ thư mục dự án:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --editable .
.venv/bin/nexus --help
```

Lệnh `--editable` giúp thay đổi trong `src/` có hiệu lực ngay, phù hợp khi học
và phát triển dự án. Không cần model để dùng hai lệnh cơ bản:

```sh
.venv/bin/nexus add "mua sữa, gọi mẹ và học Python"
.venv/bin/nexus list
```

Để thêm nhiều task, bỏ đối số và nhập mỗi task trên một dòng. Nhấn `Ctrl+D`
trên một dòng mới để kết thúc, hoặc chuyển hướng từ file UTF-8:

```sh
.venv/bin/nexus add
.venv/bin/nexus add < tasks.txt
```

Database mặc định nằm tại `$XDG_DATA_HOME/nexus/nexus.db`, hoặc
`~/.local/share/nexus/nexus.db` khi biến đó chưa được đặt. Có thể chọn file
khác bằng cách đặt `--db` trước lệnh con:

```sh
.venv/bin/nexus --db /tmp/nexus-demo.db add "học Python"
.venv/bin/nexus --db /tmp/nexus-demo.db list
```

## Chạy Qwen3-1.7B bằng llama.cpp

Cấu hình đã dùng trong dự án được ghim để một lần chạy sau có thể xác định
đúng runtime và model:

- `llama.cpp` build `b10809`, commit `5266f24da`, gói Vulkan x86-64.
- `Qwen3-1.7B-Q8_0.gguf` từ kho chính thức của Qwen.
- SHA-256 của runtime:
  `07f029cef440c82c3cff5310641eb6347e5cbcd865a5d88990215058aa049e93`.
- SHA-256 của model:
  `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`.

Tải và kiểm tra hai artifact:

```sh
mkdir -p .local-runtime models
curl -L --fail \
  -o .local-runtime/llama-b10809-bin-ubuntu-vulkan-x64.tar.gz \
  https://github.com/ggml-org/llama.cpp/releases/download/b10809/llama-b10809-bin-ubuntu-vulkan-x64.tar.gz
echo "07f029cef440c82c3cff5310641eb6347e5cbcd865a5d88990215058aa049e93  .local-runtime/llama-b10809-bin-ubuntu-vulkan-x64.tar.gz" | sha256sum --check
tar -xzf .local-runtime/llama-b10809-bin-ubuntu-vulkan-x64.tar.gz \
  -C .local-runtime

curl -L --fail \
  -o models/Qwen3-1.7B-Q8_0.gguf \
  https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf
echo "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a  models/Qwen3-1.7B-Q8_0.gguf" | sha256sum --check
```

Khởi động server bằng cấu hình đã chọn cho laptop 4 GB VRAM:

```sh
scripts/start_qwen.sh
```

Script chỉ lắng nghe tại `127.0.0.1:8087`, dùng context 4096, một slot song
song, sáu CPU thread và offload nhiều layer nhất có thể sang GPU. Request của
ứng dụng tắt thinking. Có thể thay đường dẫn và thông số bằng các biến
`NEXUS_LLAMA_SERVER`, `NEXUS_QWEN_MODEL`, `NEXUS_QWEN_PORT`,
`NEXUS_QWEN_CONTEXT`, `NEXUS_QWEN_THREADS`, `NEXUS_QWEN_GPU_LAYERS`.

Ở terminal khác:

```sh
.venv/bin/nexus ask "thêm việc mua sữa"
.venv/bin/nexus ask "xem danh sách của tôi"
```

Nếu đổi port server, đặt endpoint tương ứng trước khi chạy ứng dụng, ví dụ:

```sh
QWEN_ENDPOINT=http://127.0.0.1:8090/v1/chat/completions \
  .venv/bin/nexus ask "xem danh sách"
```

Một tool có thể đã commit vào SQLite trước khi request lấy câu trả lời cuối
gặp lỗi. Khi đó CLI in rõ các thao tác đã hoàn tất, trả exit code `1` và không
tự retry. Hãy xem danh sách trước khi gửi lại yêu cầu để tránh tạo task trùng.

## Kiểm thử và benchmark hiện tại

Toàn bộ unit test chạy offline, không cần model:

```sh
PYTHONPATH=src python3 -B -m unittest discover -s tests -v
```

Các dataset đầu vào được giữ trong Git tại `evals/`. Bộ
`current_scope_tasks.json` có 20 ca theo đúng chức năng hiện tại. Khi server
đang chạy, chạy bộ này bằng:

```sh
mkdir -p evals/results
PYTHONPATH=src python3 -m scripts.eval_qwen \
  --cases evals/current_scope_tasks.json \
  --output evals/results/current-scope-run.json
```

Runner dùng database tạm riêng cho từng ca, so tool call và trạng thái SQLite,
đồng thời ghi hash dataset, tổng kết theo nhóm, token, thời gian và lỗi
model/server vào JSON. File output phải chưa tồn tại. Trace của mỗi ca tách rõ
`proposed_calls`, `authorized_calls`, `rejected_calls` và `executed_calls`.
Các kết quả chính gồm:

- `model_proposal_status`: đề xuất ban đầu của model có khớp nhãn không.
- `system_action_status`: sau policy, tool thực thi và SQLite có đúng không.
- `system_end_to_end_status`: hành động của hệ thống và câu trả lời đều đạt.
- `tool_and_database_status`: tool call và trạng thái SQLite có đúng không.
- `reply_hygiene_status`: câu trả lời có rỗng, lộ JSON/giao thức hoặc lẫn ký tự
  CJK không; tên tool do chính người dùng nhắc tới không bị xem là rò rỉ và lỗi
  server được ghi là `not_evaluated`.
- `end_to_end_status`: chỉ đạt khi cả hai phần trên đều đạt.

Ba trường cũ tiếp tục được ghi để so sánh với các lần benchmark trước. Khối
`policy` cho biết policy đã can thiệp, đã chặn một đề xuất sai, hay đã từ chối
nhầm một đề xuất đúng. Nhờ vậy một ca an toàn nhờ policy không bị tính thành
thành công của model.

Khối `safety` ghi riêng số ca và số task đã commit khi không phương án kỳ vọng
nào yêu cầu `create_task`. Kiểm tra câu trả lời chỉ bắt các lỗi hình thức rõ
ràng, nên nội dung vẫn cần review thủ công. Tiến trình trả mã `0` chỉ khi mọi ca
đạt đầu cuối. Kết quả sinh ra trong `evals/results/` không được đưa vào Git.

Prompt v6 là một thí nghiệm few-shot có cấu trúc. Nó giữ nguyên system prompt
v3 nhưng đưa ví dụ vào các message `user`, `assistant` và `tool` thật. Runner
lưu cả `system_prompt` và `few_shot_messages` trong report để tái tạo đúng
payload. V6 chưa phải mặc định của CLI; `run_turn` vẫn dùng v1 khi không chỉ
định phiên bản.

V7 giữ nguyên v6 và thêm một system message reset sau các ví dụ. Probe cho thấy
reset dạng văn bản không ngăn được model dùng task mẫu như dữ liệu hiện tại,
nên v7 chỉ được giữ để tái tạo thí nghiệm và chưa được benchmark toàn bộ.
