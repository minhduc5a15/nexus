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

Mỗi lượt `ask` gọi model một lần và cho phép tối đa một tool call. Một call
`create_task` có thể chứa nhiều dòng. Sau khi tool thành công, formatter Python
tạo câu trả lời từ dữ liệu thật. Một tool có thể đã commit vào SQLite trước khi
formatter gặp lỗi. Khi đó CLI in rõ các thao tác đã hoàn tất, trả exit code `1` và không
tự retry. Hãy xem danh sách trước khi gửi lại yêu cầu để tránh tạo task trùng.

## Kiểm thử và benchmark hiện tại

Toàn bộ unit test chạy offline, không cần model:

```sh
PYTHONPATH=src python3 -B -m unittest discover -s tests -v
```

Các dataset đầu vào được giữ trong Git tại `evals/`. Bộ
`current_scope_tasks_v2.json` có 20 ca và áp dụng contract một tool call mỗi
lượt. Bộ `current_scope_tasks.json` cùng các holdout cũ giữ nguyên để đọc lại
lịch sử; một số ca cũ chấp nhận cả trace nhiều call. Khi server
đang chạy, chạy bộ này bằng:

```sh
mkdir -p evals/results
PYTHONPATH=src python3 -m scripts.eval_qwen \
  --cases evals/current_scope_tasks_v2.json \
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
nhầm một đề xuất đúng. `recovered_model_failure` cho biết policy còn giúp toàn
bộ hành động hệ thống trở về đúng hay không. Nhờ vậy một ca an toàn nhờ policy
không bị tính thành thành công của model.

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
nên chưa thể xem reset là giải pháp tổng quát. Trong lượt benchmark dataset v2
ngày 2026-09-16, v7 đạt 19/20 hành động và 18/20 đầu cuối tự động; v4 đạt
20/20 hành động và 17/20 đầu cuối. Đây là tập phát triển có câu trùng ví dụ.

V8 là thí nghiệm từ v4: chỉ thay hai ví dụ không gọi tool bằng phần hành động
và câu trả lời mẫu riêng biệt. Các quy tắc và hai ví dụ CREATE giữ nguyên.
Mục tiêu là tránh model đọc nguyên chỉ dẫn “không gọi tool; hỏi người dùng…”.
Benchmark đối chứng ngày 2026-09-17 cho v8 đạt 19/20 hành động và 19/20 đầu
cuối, hygiene 20/20. Hai phản hồi mục tiêu đã đúng, nhưng `list_colloquial`
không gọi LIST và còn lặp nhãn “Câu trả lời cho người dùng”. V8 chưa đạt tiêu
chí giữ 20/20 hành động của v4; CLI vẫn mặc định v1. Để so sánh từ terminal
có server đang chạy:

```sh
./scripts/demo/05_benchmark.sh --prompt-version v4 --temperature 0
./scripts/demo/05_benchmark.sh --prompt-version v8 --temperature 0
./scripts/demo/05_benchmark.sh --prompt-version v7 --temperature 0
```

Chạy từng lệnh kể cả khi lệnh trước trả mã 1 do ca trượt. Mỗi lệnh tự tạo
output riêng, không ghi đè báo cáo cũ.

## Policy: cấp quyền và kiểm tra nguồn nội dung

Policy create nhận diện lời dẫn từ các nhóm động từ, đại từ lịch sự và phần
chỉ task/nhiều dòng. Sau đó nó kiểm tra content do model đề xuất là một span
nguyên văn trong prompt và không bỏ lại nội dung cần lưu.

- Có `:` hoặc xuống dòng ngay sau lời dẫn: toàn bộ phần sau là dữ liệu, chỉ bỏ
  khoảng trắng bao quanh cả khối; giữ dấu câu và xuống dòng bên trong.
- Câu tự nhiên: phần sau content có thể là “vào danh sách”, “giúp tôi/mình/tui”,
  “nhé”, “với” và dấu kết câu. Dấu `:` trong `8:00` hoặc `C++: vector` thuộc nội dung.
- `content_not_grounded`: proposal không xuất hiện nguyên văn trong prompt.
- `content_boundary_mismatch`: proposal có xuất hiện nhưng cắt thiếu hoặc chọn
  sai vị trí. Policy không tự sửa proposal.

Ví dụ “Thêm việc mua sữa vào danh sách giúp tôi.” cho phép `mua sữa`; “Thêm việc
mua sữa và gọi mẹ” không cho phép chỉ lấy `mua sữa`. “Thêm việc: sửa xe.” phải
được giữ nguyên dấu chấm. “Thêm việc: đừng quên gọi mẹ.” là nội dung task hợp lệ.

LIST nhận cả yêu cầu xem và câu hỏi về task đã lưu, nhưng không cấp quyền từ
câu ví dụ hoặc lời kể chỉ nhắc đến danh sách. Grammar vẫn hữu hạn; chưa bảo đảm
hiểu mọi cách diễn đạt tiếng Việt. Các trường hợp không chứng minh được quyền
hoặc ranh giới nội dung vẫn bị từ chối.

## Demo có thể chạy trực tiếp

Các script tại [`scripts/demo/`](scripts/demo/README.md) lần lượt minh họa CLI,
policy, agent offline, Qwen thật, benchmark và cách đọc báo cáo. Bắt đầu bằng:

```sh
./scripts/demo/01_cli.sh
./scripts/demo/02_policy.sh
./scripts/demo/03_agent_offline.sh
```

Demo tự dùng database riêng trong `evals/results/demos/`. Hướng dẫn chạy AI và
chọn ca benchmark nằm trong README của thư mục demo.
