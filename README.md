# NEXUS

NEXUS là ứng dụng ghi nhanh việc cần làm bằng tiếng Việt. Phạm vi hiện tại gồm
thêm việc, xem danh sách, hoàn thành, sửa, xóa, đặt thời hạn theo ID và truy vấn
việc theo hạn. Người dùng có thể gọi trực tiếp bằng CLI hoặc viết yêu cầu tự nhiên
để Qwen3-4B-Instruct-2507 chọn một trong bảy tool. Dữ liệu được lưu cục bộ bằng SQLite.

Mỗi dòng có nội dung tạo một task. Dấu phẩy và chữ `và` trong cùng một dòng
không tự tách task; các dòng trống bị bỏ qua. Mỗi task có `id`, `content`,
`completed` và `due_at`. `due_at` là Unix timestamp theo giây hoặc `NULL`.

## Cài ứng dụng

NEXUS yêu cầu Python 3.10 trở lên, cùng SQLite 3.35 trở lên để dùng câu lệnh
`RETURNING`. Ứng dụng không có thư viện Python bên thứ ba. Từ thư mục dự án:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --editable .
.venv/bin/nexus --help
```

Lệnh `--editable` giúp thay đổi trong `src/` có hiệu lực ngay, phù hợp khi học
và phát triển dự án. Không cần model để dùng các lệnh trực tiếp:

```sh
.venv/bin/nexus add "mua sữa, gọi mẹ và học Python"
.venv/bin/nexus list
.venv/bin/nexus complete 1
.venv/bin/nexus edit 1 "mua sữa không đường"
.venv/bin/nexus deadline 1 "8 giờ sáng mai"
.venv/bin/nexus due today
.venv/bin/nexus due tomorrow
.venv/bin/nexus due overdue
.venv/bin/nexus delete 1
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

## Chạy Qwen3-4B-Instruct-2507 bằng llama.cpp

Cấu hình mặc định đã dùng và kiểm chứng trên laptop 4 GB VRAM:

- `llama.cpp` build `b10809`, commit `5266f24da`, gói Vulkan x86-64.
- `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` từ bản chuyển đổi GGUF của Unsloth.
- SHA-256 của runtime:
  `07f029cef440c82c3cff5310641eb6347e5cbcd865a5d88990215058aa049e93`.
- SHA-256 của model:
  `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597`.

Cài đúng runtime nếu checkout chưa có:

```sh
mkdir -p .local-runtime
curl -L --fail \
  -o .local-runtime/llama-b10809-bin-ubuntu-vulkan-x64.tar.gz \
  https://github.com/ggml-org/llama.cpp/releases/download/b10809/llama-b10809-bin-ubuntu-vulkan-x64.tar.gz
echo "07f029cef440c82c3cff5310641eb6347e5cbcd865a5d88990215058aa049e93  .local-runtime/llama-b10809-bin-ubuntu-vulkan-x64.tar.gz" | sha256sum --check
tar -xzf .local-runtime/llama-b10809-bin-ubuntu-vulkan-x64.tar.gz \
  -C .local-runtime
```

Tải model bằng script có resume và kiểm tra hash:

```sh
./scripts/download_qwen4.sh
```

Khởi động server:

```sh
./scripts/start_qwen.sh
```

Script chỉ lắng nghe tại `127.0.0.1:8087`, dùng context 4096, một slot song
song, sáu CPU thread và offload toàn bộ layer sang GPU. Lần đo hiện tại dùng
khoảng 3.037 MiB trong 4.096 MiB VRAM. Request của ứng dụng tắt thinking.
Có thể thay đường dẫn và thông số bằng `NEXUS_LLAMA_SERVER`,
`NEXUS_QWEN_MODEL`, `NEXUS_MODEL_ID`, `NEXUS_QWEN_PORT`,
`NEXUS_QWEN_CONTEXT`, `NEXUS_QWEN_THREADS` và `NEXUS_QWEN_GPU_LAYERS`.
Đặt `NEXUS_LLAMA_LOG_PROMPTS_DIR` nếu cần lưu prompt đã render để chẩn đoán.

Model 1.7B trước đây vẫn dùng được làm baseline nếu file cũ còn trong `models/`:

```sh
./scripts/start_qwen_1_7b.sh
```

Ở terminal khác:

```sh
.venv/bin/nexus ask "thêm việc mua sữa"
.venv/bin/nexus ask "xem danh sách của tôi"
.venv/bin/nexus chat
```

`ask` và `chat` mặc định gửi model ID `qwen3-4b-instruct-2507-q4_k_m`; có thể
ghi đè bằng `--model`. `chat` giữ một `AgentSession` trong suốt tiến trình. Mỗi
dòng nhập là một tin
nhắn; gõ `/exit` hoặc nhấn `Ctrl+D` để thoát. Ví dụ:

```text
NEXUS > Thêm việc
NEXUS: Bạn muốn thêm việc gì?
NEXUS > mua sữa
NEXUS: Đã thêm [1] mua sữa
NEXUS > Xem danh sách
NEXUS: Danh sách hiện có 1 việc:
[1] [ ] mua sữa
NEXUS > Hoàn thành việc 1
NEXUS: Đã hoàn thành [1] mua sữa
NEXUS > Sửa việc 1 thành mua sữa không đường
NEXUS: Đã sửa [1] thành: mua sữa không đường
NEXUS > Đặt hạn việc 1 lúc 8 giờ sáng mai
NEXUS: Đã đặt hạn [1] vào 27/09/2026 08:00: mua sữa không đường
NEXUS > Xóa việc 1
NEXUS: Bạn có chắc muốn xóa [1] [x] mua sữa không đường?
NEXUS > có
NEXUS: Đã xóa [1] mua sữa không đường
```

Để xem đầy đủ luồng xử lý của từng lượt:

```sh
.venv/bin/nexus chat --trace
.venv/bin/nexus chat --prompt-version v13 --tool-routing classified --trace
```

`--tool-routing all` gửi đủ bảy tool như trước. Chế độ thử nghiệm
`classified` dùng classifier hữu hạn để chỉ gửi tool liên quan; với câu chưa
nhận diện được, nó tự fallback về đủ bảy tool. Routing chỉ giới hạn schema model
nhìn thấy. Policy vẫn kiểm tra proposal và giữ quyền cho phép side effect.

Trace được in trên stderr, tách khỏi câu trả lời người dùng trên stdout. Mỗi
khối chứa input, state session trước/sau, request/response model, nguồn và
trạng thái runtime, proposal, kiểm tra contract một tool call, call được cấp
quyền hoặc từ chối, tool đã thực thi, SQLite trước/sau, reply và lỗi. Request
model chứa system prompt và tool schema thật; chỉ bật trace khi cần chẩn đoán.

Nếu đổi port server, đặt endpoint tương ứng trước khi chạy ứng dụng, ví dụ:

```sh
QWEN_ENDPOINT=http://127.0.0.1:8090/v1/chat/completions \
  .venv/bin/nexus ask "xem danh sách"
```

Mỗi lượt `ask` gọi model một lần và cho phép tối đa một tool call. Prompt mặc
định vẫn là v9; v1–v13 được giữ để tái tạo các thí nghiệm. V10 thêm EDIT, v11
thêm DELETE có xác nhận, v12 thêm DEADLINE theo ID và v13 thêm truy vấn theo hạn.
V13 cùng classified routing đạt live smoke chuẩn 20/20 lượt, nhưng chưa được
chọn làm mặc định vì median latency tăng quá cổng 10% đã chốt. Một call
`create_task` có thể chứa nhiều dòng và lưu chúng trong cùng một transaction.
Sau khi tool thành công, formatter Python tạo câu trả lời từ dữ liệu thật.
Nếu model không gọi tool, ứng dụng chỉ báo chưa thực hiện thao tác; lời model
được giữ trong trace chẩn đoán, không dùng làm lời xác nhận cho người dùng.
`run_turn` trả `status` (`no_tool`, `rejected`, `needs_clarification`, `needs_confirmation`,
`not_found`, `executed`)
cùng trace các call. Khi policy từ chối, ứng dụng báo rõ thao tác chưa được hỗ
trợ hoặc nội dung không khớp lời người dùng; các lỗi định dạng và lượt không có
tool vẫn dùng câu báo không thực hiện thao tác. Một tool có thể đã commit vào
SQLite trước khi formatter gặp lỗi. Khi đó CLI in rõ các thao tác đã hoàn tất,
trả exit code `1` và không tự retry. Hãy xem danh sách trước khi gửi lại yêu
cầu để tránh tạo task trùng.

Nếu ứng dụng cần nhận nhiều tin nhắn trong cùng một cuộc trò chuyện, dùng
`AgentSession` riêng cho từng cuộc trò chuyện. State machine có mười trạng thái
trong bộ nhớ: `idle`, `awaiting_create_content`, `awaiting_complete_id`,
`awaiting_edit_id`, `awaiting_edit_content`, `awaiting_delete_id`,
`awaiting_delete_confirmation`, `awaiting_deadline_id`,
`awaiting_deadline_text` và `awaiting_deadline_scope`. Hai trạng thái EDIT và
hai trạng thái đặt deadline giữ phần dữ liệu đã có trong khi hỏi phần còn thiếu. Ví dụ khi đã khởi động llama.cpp:

```python
from nexus.agent.session import AgentSession
from nexus.storage.sqlite_db import initialize_database

database_path = "/tmp/nexus-session-demo.db"
initialize_database(database_path)
session = AgentSession(database_path)
print(session.run_turn("Thêm việc")["reply"])
print(session.run_turn("mua sữa\ngọi mẹ")["reply"])
```

Lượt đầu chuyển `idle → awaiting_create_content` mà không gọi model; lượt thứ
hai dùng quyền CREATE từ lượt trước, lưu mỗi dòng thành một task trong cùng
transaction rồi chuyển về `idle` và trả `source="session_continuation"`. Mỗi kết
quả hoàn tất chứa `session_state_before` và `session_state_after`. Nội dung trả
lời được dùng nguyên văn, không được model viết lại.

Khi chờ ID, lượt sau chỉ nhận `3`, `#3`, `việc 3` hoặc `task 3`; không tìm task
theo nội dung. EDIT thiếu ID hoặc content hỏi lại đúng phần còn thiếu. Tin nhắn
trống giữ nguyên trạng thái chờ; `thôi` hoặc `hủy` hủy yêu cầu và về `idle`.
Một yêu cầu CREATE/LIST/COMPLETE/EDIT/DELETE/DEADLINE/DUE QUERY mới thay thế yêu cầu đang chờ. DELETE
trong agent luôn hiển thị snapshot task và chờ một xác nhận hẹp; `nexus delete ID` là lệnh cấu trúc nên xóa trực tiếp. Nếu task đổi giữa hai lượt, NEXUS cập
nhật snapshot và hỏi lại; nếu task đã biến mất, NEXUS không xóa gì.

DEADLINE chỉ nhận một ID và một cụm thời gian nguyên văn. Parser Python dùng múi
giờ cố định `Asia/Ho_Chi_Minh` và chỉ hiểu các dạng `27/09/2026 08:00`,
`08:00 ngày 27/09/2026`, `08:00 hôm nay/mai/ngày mai`, `8 giờ sáng mai` và
`8 giờ 30 phút tối ngày mai`. Model không được truyền timestamp. Khi thiếu ID
nhưng đã có thời gian, session giữ cả cụm thời gian và mốc tham chiếu của lượt
đầu; khi thiếu thời gian, session giữ ID. Chưa hỗ trợ xóa deadline.

Truy vấn deadline chỉ trả task chưa hoàn thành và có `due_at`. Ba scope hữu hạn
là `today`, `tomorrow` và `overdue`; CLI dùng `nexus due SCOPE`, còn agent dùng
`list_tasks_by_deadline`. `today` gồm cả task đã trễ trong ngày, vì vậy một task
có thể xuất hiện đồng thời ở `today` và `overdue`. Kết quả được sắp theo
`due_at`, rồi ID. Câu “Xem việc theo hạn” chuyển session sang
`awaiting_deadline_scope`; lượt sau chỉ nhận “hôm nay”, “ngày mai”, “mai”,
“quá hạn” hoặc “trễ hạn”.

Nếu SQLite lỗi trước commit, session vẫn chờ để người dùng thử lại. Nếu định dạng phản hồi
lỗi sau commit, session đã về `idle` để tránh tự thực thi lại. Session không lưu trạng thái qua lần khởi động lại, không tự giữ lịch sử model và không được dùng chung
cho nhiều người. `nexus chat` tạo đúng một
session cho tiến trình CLI; `nexus ask` và hàm `run_turn` vẫn là các lượt độc
lập. CLI chat hiện coi mỗi dòng là một tin nhắn, nên chưa nhập được một tin nhắn
nhiều dòng trực tiếp; API `AgentSession` vẫn hỗ trợ content nhiều dòng. Bộ nhận
diện lệnh mới có grammar hữu hạn, nên câu mơ hồ giữa lệnh và nội dung task vẫn
cần người dùng diễn đạt rõ hơn.

SQLite dùng schema version 3 và khóa chính `INTEGER PRIMARY KEY AUTOINCREMENT`.
Cột `due_at INTEGER NULL` lưu Unix timestamp theo giây. Database legacy chỉ có
`id/content`, version 1 có thêm `completed`, còn version 2 đã dùng
`AUTOINCREMENT`; tất cả được migration trong một transaction, giữ nguyên ID,
content và trạng thái, đồng thời đặt deadline cũ thành `NULL`. ID đã xóa không
được cấp lại. Ứng dụng dừng với lỗi rõ khi
gặp schema lạ hoặc database có version mới hơn.

## Kiểm thử và benchmark hiện tại

Toàn bộ unit test chạy offline, không cần model:

```sh
PYTHONPATH=src python3 -B -m unittest discover -s tests -v
```

Runtime hội thoại có evaluator riêng. Bản scripted dùng proposal cố định nên
không cần server và đo trực tiếp state, tool, SQLite, reply cùng ghi ngoài yêu
cầu:

```sh
./scripts/demo/09_session_eval.sh
```

Dataset [session_conversations_v1.json](evals/session_conversations_v1.json)
gồm 13 cuộc hội thoại và 32 lượt. Nó bao phủ bổ sung content, nhiều dòng, hủy,
tin nhắn trống, lệnh mới thay thế trạng thái chờ, batch hai call, rollback,
lỗi formatter và hai session độc lập. Khi đang chờ CREATE, tin nhắn tự do kế
tiếp được xem là content theo quyền từ lượt trước; người dùng dùng `thôi`, `hủy`
hoặc một lệnh mới rõ ràng để thay thế yêu cầu đang chờ.

Sau khi chạy llama.cpp, smoke suite Qwen thật dùng bảy ca nhỏ theo contract hiện tại. `session_smoke_v1.json` được giữ làm lịch sử; script dùng v2 vì v1 từng kỳ vọng sai rằng câu yêu cầu tự nhiên có dấu chấm phải bị từ chối:

```sh
./scripts/demo/10_session_smoke.sh
```

Completion có hai bộ riêng. Bộ scripted kiểm tra cả proposal sai ID bị policy
chặn; bộ live đo Qwen v9 trên CREATE → COMPLETE → LIST, hoàn thành lại, ID không
tồn tại, hỏi ID qua hai lượt và yêu cầu theo nội dung:

```sh
./scripts/demo/11_completion_eval.sh
./scripts/demo/12_completion_smoke.sh
```

EDIT cũng có bộ scripted và smoke riêng. Bộ scripted kiểm tra ID/content chính
xác, continuation, giữ trạng thái completion và các proposal sai bị chặn. Bộ
smoke dùng prompt v10. Qwen 4B đạt 9/10 lượt và không có update ngoài yêu
cầu; ca không tồn tại trả lời bằng text thay vì gọi tool, nên prompt mặc định vẫn là v9:

```sh
./scripts/demo/13_edit_eval.sh
./scripts/demo/14_edit_smoke.sh
```

DELETE có bộ scripted 13 case/27 lượt kiểm tra xác nhận, hủy, thiếu/nhiều ID,
snapshot stale, lỗi trước/sau commit và proposal sai. Bộ này đạt toàn bộ. Smoke
Qwen 4B v11 đạt 7/8 case, 14/16 lượt và không có mutation ngoài yêu cầu;
toàn bộ luồng DELETE đạt, còn regression EDIT không gọi tool. Theo quy tắc rollout,
prompt mặc định vẫn là v9 và v11 chỉ dùng qua `--prompt-version v11`:

```sh
./scripts/demo/15_delete_eval.sh
./scripts/demo/16_delete_smoke.sh
.venv/bin/nexus chat --prompt-version v11 --trace
```

DEADLINE có bộ scripted 14 case/29 lượt, đạt toàn bộ và không có mutation ngoài
yêu cầu. Live smoke Qwen 4B v12 đạt 8/9 case, 14/15 lượt; cả ba case deadline đều
đạt và không có mutation ngoài yêu cầu. Ca regression EDIT không gọi tool, nên
SQLite không đổi. Theo tiêu chí rollout,
v9 vẫn là mặc định và không tiếp tục chỉnh prompt theo ca này:

```sh
./scripts/demo/17_deadline_eval.sh
./scripts/demo/18_deadline_smoke.sh
.venv/bin/nexus chat --prompt-version v12 --trace
```


Truy vấn deadline có bộ scripted 9 case/18 lượt, đạt toàn bộ và không thay đổi
database. Live smoke Qwen 4B v13 đạt 12/13 case, 19/20 lượt; cả ba luồng query mới
đạt. CREATE trước đây trượt với model 1.7B nay đã đạt; regression EDIT không gọi
tool. Không có mutation ngoài yêu cầu. Vì smoke chưa đạt toàn
bộ, prompt mặc định vẫn là v9:

```sh
./scripts/demo/19_deadline_query_eval.sh
./scripts/demo/20_deadline_query_smoke.sh
.venv/bin/nexus chat --prompt-version v13 --trace
```

Các evaluator ghi report mới vào `evals/results/demos/`; thư mục này bị Git
ignore. Exit code `1` nghĩa là ít nhất một ca không đạt contract, không nhất
thiết là lỗi chương trình.

Classified tool routing hiện là tính năng thử nghiệm. Trên diagnostic Qwen 4B,
mode `all` đạt 13/14 case, 20/21 lượt; `classified` đạt 14/14 case, 21/21 lượt.
EDIT và duplicate CREATE đều đạt 3/3 lần lặp, còn smoke v13 chuẩn đạt 13/13 case,
20/20 lượt. Median prompt token giảm từ 1.286 xuống 576,5, nhưng median latency
tăng từ 1,020 lên 1,414 giây do các schema thay đổi làm prompt cache của
llama.cpp tái sử dụng kém hơn. Holdout ghép cặp đạt bằng nhau 7/8 case, 8/9 lượt;
cả hai cùng trượt một cách nói CREATE mới và không có mutation ngoài yêu cầu.
Vì cổng latency không đạt, mặc định vẫn là prompt v9 và routing `all`:

```sh
./scripts/demo/21_tool_routing_eval.sh
./scripts/demo/22_tool_routing_diagnostic.sh
.venv/bin/nexus chat --prompt-version v13 --tool-routing classified --trace
```

Không dùng holdout để chỉnh prompt hoặc classifier trong cùng giai đoạn này.

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

Bộ `evals/generalization_holdout_v1.json` có 92 ca mới, được gắn nhãn và
kiểm tra trùng câu trước lần chạy model đầu tiên. Nó kiểm tra cách diễn đạt
CREATE/LIST khác tập 20 ca, nội dung nguyên văn, một dòng/nhiều dòng, lời kể,
phủ định và thao tác chưa hỗ trợ. Lượt đo đầu chạy v1/v4/v8 với temperature 0;
xem `notes/24-generalization-holdout-v1-2026-09-17.md` và
`evals/results/generalization-holdout-2026-09-17/run-01/comparison.md`.
Kết quả này được dùng để chẩn đoán, không dùng chỉnh prompt/policy theo từng câu;
đây là dữ liệu tự soạn, chưa thay thế đánh giá từ cách nói thực tế của người dùng.
Trong lượt đo, chỉ số `unrequested_write` bằng 0 nhưng có ca CREATE lưu cả lời
lịch sự vào nội dung task; phải xem thêm `tasks_match` và `database_after`.

Sau bản sửa policy, replay đúng response cũ trên holdout v1 đưa false rejection
v1/v4/v8 từ 6/11/7 về 0 và không còn ghi thừa lời lịch sự. Bộ mới
`evals/generalization_holdout_v2.json` có 56 ca được chốt trước khi chạy model:
v4 đạt 35 proposal đúng, 34 hành động đúng, còn 4 false rejection và không
ghi task sai. Xem `notes/25-policy-coverage-boundary-holdout-v2.md` và artifact
trong `evals/results/policy-generalization-2026-09-20/` (cả hai chỉ lưu cục bộ).

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
./scripts/demo/08_session.sh
```

Demo tự dùng database riêng trong `evals/results/demos/`. Để tự nhập yêu cầu
và thấy từng bước với Qwen thật, chạy `scripts/start_qwen.sh` ở một terminal,
rồi `./scripts/demo/04_ai.sh --interactive` ở terminal khác. Hướng dẫn chi tiết
và cách chọn ca benchmark nằm trong README của thư mục demo.
