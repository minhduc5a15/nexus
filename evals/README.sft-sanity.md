# SFT sanity QLoRA v1

Lát cắt này khóa dữ liệu, kiểm tra labels rồi chạy một QLoRA sanity thật trên
Modal T4. Nó không gọi tool, policy, SQLite, development hoặc final holdout.

Dataset `sft_sanity_v1.json` gồm 16 mẫu:

- 8 tool call, phủ cả bảy action; CREATE có hai mẫu để kiểm tra dấu câu và dòng trống.
- 4 câu cần hỏi lại vì thiếu dữ liệu hoặc có nhiều lựa chọn.
- 4 câu không được hành động: câu trần thuật, phủ định DELETE, reminder lặp và
  hoàn tác completion chưa hỗ trợ.

Mỗi case chỉ lưu hai message user/assistant để dễ review. Khi đọc dataset,
`materialize_case()` thêm nguyên system prompt v13 và bảy tool schema hiện tại.
Validator từ chối tool sai, nhiều call, argument không grounded, call trong ca
clarification/no-action, mất cân bằng và prompt trùng hoặc quá gần development /
final holdout.

Chạy audit offline:

```sh
./scripts/demo/27_sft_sanity_data.sh
```

Để render bằng tokenizer Qwen thật:

```sh
.venv/bin/pip install -r requirements-sft-inspection.txt
./scripts/demo/27_sft_sanity_data.sh --tokenize
```

Inspector tokenizes phần prompt kết thúc ở generation prefix, rồi tokenizes toàn
bộ conversation. Nó yêu cầu dãy đầu là prefix byte-for-byte của dãy đầy đủ,
đặt label `-100` cho toàn bộ prefix và chỉ giữ token assistant làm labels. Cách
này khiến system prompt, tool schema và user message không chịu loss.

Kết quả kiểm tra bằng `transformers 4.57.1`, tokenizer revision
`cdbee75f17c01a7cc42f958dc650907174af0554`:

- Cả 16/16 record có assistant suffix hợp lệ và ít nhất một supervised token.
- Record `sft_create_delimited` có 1.300 prompt token và 28 supervised token.
- Dấu chấm của `đối chiếu số dư cuối quý.` nằm trong assistant target.
- Record dài nhất có 1.343 token, không vượt giới hạn chuẩn bị 2.048 token.

## Protocol training

`sft_sanity_training_manifest_v1.json` khóa toàn bộ thí nghiệm:

- Qwen3-4B-Instruct-2507 tại revision đã nêu trên.
- Base load bằng NF4 + double quantization, compute FP16.
- LoRA rank 16, alpha 32, target `all-linear`; 33.030.144 phần tử adapter ở FP32.
- TRL 0.27.0, PEFT 0.18.0, Transformers 4.57.1 và PyTorch 2.8.0.
- Batch 1, learning rate 2e-4, 64 optimizer step, seed 42.
- Trainer AMP tắt. T4 vẫn dùng FP16 bên trong linear 4-bit; việc tắt AMP tránh
  GradScaler cố unscale gradient BF16 do dtype gốc của checkpoint.

Chạy preflight hoàn toàn offline:

```sh
./scripts/demo/28_sft_qlora_modal.sh
```

Chạy probe và dùng đúng run đó làm gate cho overfit:

```sh
./scripts/demo/28_sft_qlora_modal.sh probe my-sft-probe
./scripts/demo/28_sft_qlora_modal.sh overfit my-sft-overfit my-sft-probe
```

Probe làm đúng một backward và optimizer step. Nó từ chối tiếp tục nếu base chưa
đóng băng, có parameter trainable ngoài LoRA, adapter không có gradient, base có
gradient, hash adapter không đổi hoặc artifact không lưu được. Overfit chỉ nhận
một probe hoàn tất cùng model, dataset và cấu hình; sau training nó giải phóng
model, nạp lại base cùng adapter đã lưu rồi so hash trọng số.

## Kết quả run 2026-10-06

Run hợp lệ:

- Probe: `sft-sanity-probe-20261006d`, 7/7 gate đạt.
- Overfit: `sft-sanity-overfit-20261006c`, 5/5 gate đạt.
- Train 64 step trong 378,93 giây; toàn function mất khoảng 464 giây gồm đo
  trước/sau, save và reload.
- Teacher-forced mean loss: 1,4670 → 0,01256.
- Teacher-forced token accuracy: 82,37% → 99,45%.
- Exact teacher-forced sequence: 6/16 → 14/16.
- Peak CUDA allocated: 6.525.612.544 byte; reserved: 8.218.738.688 byte.
- Adapter safetensors: 66.127.776 byte.
- Hash state adapter sau train và sau reload cùng là
  `a58ff5b6e2a0240a10aeeed61c4d70d63783a5aa1a2115b16b1cf14a3e1ec4a9`.

Sau khi tải hai report vào `evals/results/`, có thể kiểm tra lại chúng độc lập:

```sh
PYTHONPATH=src .venv/bin/python -m scripts.sft_sanity_protocol \
  --probe-report evals/results/sft-sanity-probe-20261006d-probe.json \
  --probe-run-id sft-sanity-probe-20261006d \
  --overfit-report evals/results/sft-sanity-overfit-20261006c-report.json \
  --overfit-run-id sft-sanity-overfit-20261006c \
  --approved-probe-run-id sft-sanity-probe-20261006d
```

Hai ca chưa exact là `sft_no_action_bare_statement` và
`sft_no_action_recurring_reminder`. Đây là kết quả teacher forcing trên chính 16
mẫu training. Nó chứng minh pipeline loss → gradient → optimizer → save/load hoạt
động; nó chưa chứng minh model sinh đúng tự do hoặc tổng quát hóa sang câu mới.

## Free generation sau khi reload

Run `sft-sanity-generation-20261008b` sinh greedy trên cùng 16 mẫu, cùng thứ tự,
trước bằng base model rồi bằng adapter đã reload:

| Chỉ số | Base | Adapter |
| --- | ---: | ---: |
| Proposal đúng | 14/16 | 16/16 |
| Exact assistant token sequence | 6/16 | 14/16 |
| Tool-call đúng | 6/8 | 8/8 |
| Clarification không gọi tool | 4/4 | 4/4 |
| No-action không gọi tool | 4/4 | 4/4 |
| Thời gian sinh 16 lượt | 62,58 giây | 63,07 giây |
| Output token | 586 | 342 |

Hai lỗi base đều nằm trong arguments: đổi `đóng` thành `dóng` ở content nhiều
dòng, và bỏ dấu chấm cuối content EDIT. Adapter sửa được cả hai. Peak CUDA của
run so sánh là 4.022.468.096 byte allocated và 4.282.384.384 byte reserved.

`16/16 proposal` không đồng nghĩa mọi reply đều tốt. Review thủ công tám phản
hồi không gọi tool cho kết quả base 8/8, adapter 7/8. Ở
`sft_no_action_bare_statement`, adapter không tạo tool call nhưng nói
`Tôi sẽ lưu … thành một task`, ngụ ý hành động không được yêu cầu. Ca reminder
còn lại khác target nhưng câu `Tôi chưa hỗ trợ nhắc lặp lại.` là đúng. Review có
hash report và rubric trong `sft_generation_review_v1.json`.

Đây vẫn là training-set memorization. Kết quả đủ để xác nhận adapter tác động đến
free generation, nhưng chưa được dùng làm bằng chứng generalization hay rollout.

Chạy lại stage này:

```sh
./scripts/demo/28_sft_qlora_modal.sh generation \
  my-sft-generation sft-sanity-overfit-20261006c
```

Sau khi tải report về, kiểm tra độc lập:

```sh
PYTHONPATH=src .venv/bin/python -m scripts.sft_generation_protocol \
  --report evals/results/sft-sanity-generation-20261008b-report.json \
  --run-id sft-sanity-generation-20261008b
```

Hai lỗi chạy bị giữ lại trong Modal volume như bằng chứng chẩn đoán: lần đầu gọi
audit leakage bên trong image không chứa eval; lần sau Trainer AMP cố unscale
gradient BF16 trên T4. Cách sửa cuối cùng không đưa holdout vào image và không
đổi GPU: leakage audit nằm ở preflight, adapter FP32 và Trainer AMP tắt.

## Bộ 100 mẫu pilot

`sft_pilot_v1.json` giữ nguyên 16 mẫu sanity và thêm 84 mẫu được gán nhãn trực
tiếp trong repository. Đây là dữ liệu training candidate; không mẫu nào được
coi là bằng chứng đánh giá model.

| Nhóm | Số mẫu |
| --- | ---: |
| Tool call | 56 |
| Clarification | 22 |
| No action | 22 |

56 tool call được chia đều: mỗi action CREATE, LIST, COMPLETE, EDIT, DELETE,
DEADLINE và DUE có tám mẫu. Auditor buộc mỗi record chỉ có tối đa một call,
arguments đúng schema, `content`/`when` xuất hiện nguyên văn và ID nằm trong lời
người dùng. Với non-call, mọi clarification phải là câu hỏi và câu trả lời không
được tuyên bố hoặc hứa thực hiện mutation.

Dataset được dựng lại hoàn toàn từ `scripts/build_sft_pilot_v1.py`. Manifest
khóa hash dataset, 16 mẫu nguồn, builder, prompt v13, bảy tool schema và hai tập
đánh giá. Review liệt kê đủ 100 ID theo đúng thứ tự; cả manifest lẫn review đều
đặt `training_authorized=false` để mốc này dừng trước GPU training.

Leakage screen so prompt sau NFKC/casefold, che gold `content`/`when` và chữ số,
rồi dùng SequenceMatcher với development và final holdout. Kết quả hiện tại:
không exact match, không masked match, không cặp nào đạt ngưỡng 0,92; cặp gần
nhất là 0,915254. Đây là bộ lọc từ vựng, không phải bằng chứng hai tập độc lập về
ngữ nghĩa.

Tokenizer Qwen revision đã khóa xác minh exact prefix cho cả 100/100 record:

- Ngắn nhất: 1.296 token.
- Trung vị: 1.315,5 token.
- Dài nhất: 1.362 token, dưới giới hạn 2.048.
- Chỉ assistant suffix nhận label; toàn bộ system/user prefix mang `-100`.

Chạy audit offline hoặc kiểm tra tokenizer:

```sh
./scripts/demo/29_sft_pilot_data.sh
./scripts/demo/29_sft_pilot_data.sh tokens
```

Policy grammar hiện tại cố ý không được dùng làm đáp án chuẩn cho nhãn model.
Vì vậy một proposal có thể đúng theo ý nghĩa/contract nhưng vẫn bị policy hữu
hạn từ chối; khi đánh giá ứng dụng, trường hợp đó phải được ghi là policy
coverage thay vì sửa gold label để model học né policy.

## Protocol training pilot

`sft_pilot_training_manifest_v1.json` khóa thí nghiệm đầu tiên trên 100 mẫu:

- QLoRA NF4, LoRA rank 16/alpha 32, chỉ adapter FP32 được học.
- 200 optimizer step, batch 1, learning rate `1e-4`, warmup 10 step, seed 42.
- Stage `train` không chứa development hoặc final holdout trong image.
- Stage `development` chỉ chạy sau một train report hoàn tất, dùng 80 ca
  diagnostic đã lộ và sinh greedy bằng base/candidate trong cùng runtime.
- Không proposal nào được thực thi; policy, tool và SQLite không tham gia.
- Hai timeout GPU cộng lại tối đa 4.800 giây, tương đương trần 0,7872 USD theo
  đơn giá T4 0,000164 USD/giây trong manifest.

Preflight offline:

```sh
./scripts/demo/30_sft_pilot_modal.sh
```

Chạy train rồi development bằng hai run ID riêng:

```sh
./scripts/demo/30_sft_pilot_modal.sh train my-pilot-train
./scripts/demo/30_sft_pilot_modal.sh development my-pilot-dev my-pilot-train
```

Automatic scoring của development chỉ kết luận model có gọi đúng tool và truyền
đúng arguments hay không. 24 phản hồi clarification/no-action vẫn phải được đọc
thủ công trước khi chọn candidate, vì “không gọi tool” chưa đủ chứng minh câu trả
lời phù hợp. Final holdout tiếp tục đóng cho tới khi candidate và cấu hình được
chốt từ development.

## Kết quả pilot 100 mẫu ngày 2026-10-11

Train run `sft-pilot-train-20261011a` hoàn tất 200/200 optimizer step trên T4:

| Teacher-forced trên 100 mẫu train | Trước | Sau |
| --- | ---: | ---: |
| Mean loss | 1,20650 | 0,05107 |
| Token accuracy | 85,38% | 98,36% |
| Exact sequence | 46/100 | 77/100 |

Training loss là 0,33787; 2.251 assistant token chịu loss. Run mất 1.170,09
giây, peak CUDA allocated 6.578.489.856 byte và reserved 9.869.197.312 byte.
Adapter save/reload có cùng state hash; bảy gate hạ tầng đều đạt. Đây vẫn chỉ
là số đo teacher forcing trên chính dữ liệu train.

Development run hợp lệ `sft-pilot-development-20261011b` sinh greedy trên 80 ca
diagnostic đã lộ, không thực thi proposal:

| Proposal exact | Base | Adapter |
| --- | ---: | ---: |
| Toàn bộ | 66/80 | 72/80 |
| Tool-call | 48/56 | 53/56 |
| Clarification | 10/13 | 12/13 |
| No-action | 8/11 | 7/11 |

CREATE tăng 3/8 → 6/8 và EDIT tăng 5/8 → 7/8; năm action còn lại giữ 8/8.
Số lỗi arguments giảm từ 7 xuống 2. Tuy nhiên adapter tạo bốn call khi chưa được
ủy quyền, gồm các regression đáng kể: biến “bỏ hoàn thành” thành COMPLETE, biến
“xóa deadline” thành đặt deadline `none`, và biến “tuần sau” thành `tomorrow`.
Nó còn tự CREATE từ câu trần thuật, tự chọn truy vấn hạn `today`, làm mất dòng
trống trong content nhiều dòng và bỏ dấu chấm ở EDIT.

Review thủ công đủ 24 phản hồi non-call cho kết quả base 15/24 và adapter 18/24.
Adapter trả lời mơ hồ ở yêu cầu xóa theo content, khiến người dùng có thể hiểu
nhầm rằng xóa theo content được hỗ trợ. Vì các lỗi mới có thể dẫn đến sai action,
candidate **không đủ điều kiện rollout** và final holdout vẫn đóng. Không dùng
72/80 như bằng chứng generalization: đây là development đã dùng để chẩn đoán.

Development run `sft-pilot-development-20261011a` đã sinh xong nhưng không ghi
report do một lỗi biểu diễn gate: trường `final_holdout_inference=false` bị đặt
vào tập yêu cầu mọi giá trị đều `true`. Bản sửa đổi gate thành
`final_holdout_not_used=true`; model, adapter và dữ liệu không đổi trước lần chạy
lại. Regression test khóa cực tính này.

Sau khi tải artifact về đúng các đường dẫn trong demo, xác minh độc lập toàn bộ
hash, summary tự động và quyết định review bằng:

```sh
./scripts/demo/30_sft_pilot_modal.sh audit
```

Report GPU nằm trong `evals/results/` và được Git ignore. File review có track
hash report, adapter, runner, raw output và quyết định của từng ca non-call.

## Candidate data v2 sau pilot

Pilot cho thấy pipeline học được nhưng tạo regression ở ranh giới authorization.
`sft_candidate_v2.json` giữ nguyên 100 mẫu pilot và thêm 300 mẫu, tổng cộng:

| Behavior | Số ca |
| --- | ---: |
| Tool call | 280, đúng 40 ca cho mỗi action |
| Clarification | 60 |
| No action | 60 |

Phần bổ sung tập trung vào thao tác đảo chưa hỗ trợ (`uncomplete`, gỡ deadline),
ngữ pháp thời gian ngoài phạm vi, câu trần thuật/mơ hồ, thiếu hoặc nhiều ID và
ranh giới nguyên văn của content/when. Bộ dữ liệu có 256 family; mỗi family có
tối đa bốn biến thể. Audit kiểm tra schema, grounding, cân bằng, reply không tự
nhận side effect và similarity với development/final holdout ở ngưỡng 0,92.

Chạy checkpoint dữ liệu bằng:

```sh
./scripts/demo/31_sft_candidate_data.sh
```

Manifest và review vẫn đặt `training_authorized=false`; final holdout chưa được
gửi tới model. Gate tiếp theo là kiểm tra tokenizer/assistant-only mask của đủ
400 mẫu, sau đó mới khóa protocol, ngân sách và cấu hình train riêng.

Tokenizer Qwen đúng revision đã render đủ 400 mẫu bằng Transformers 4.57.1:

| Token | Min | Median | Max |
| --- | ---: | ---: | ---: |
| Tổng mỗi record | 1.296 | 1.316 | 1.362 |
| Prompt đã mask | 1.286 | 1.294 | 1.315 |
| Assistant chịu loss | 9 | 23 | 47 |

Một epoch có tổng 9.834 assistant token chịu loss. Exact-prefix mask đạt ở tất
cả record và không record nào vượt giới hạn 2.048 token.

Protocol candidate v2 giữ nguyên rank 16/alpha 32 và learning rate `1e-4`, dùng
batch vật lý 1, gradient accumulation 4 và hai lượt qua 400 mẫu. Như vậy có 800
microbatch nhưng chỉ 200 optimizer step. Adapter được giữ ở step 100 (cuối epoch
1) và step 200 (cuối epoch 2); cả hai sẽ được so với base trên development.

Hai function T4 có timeout tối đa 6.600 + 1.800 = 8.400 giây. Với đơn giá T4
`0,000164 USD/giây` kiểm tra ngày 12/10/2026, trần GPU là `1,3776 USD`; phí thực
tế tính theo thời gian dùng. Ngân sách xin duyệt là tối đa `1,50 USD` để chừa
`0,1224 USD` cho CPU và memory được tính riêng; protocol không tự retry. Manifest
đã được người dùng duyệt ngày 12/10/2026. Manifest ghi
`training_authorized=true`; local entrypoint và remote function vẫn kiểm tra cờ
này trước khi gọi GPU. Preflight offline:

```sh
./scripts/demo/32_sft_candidate_modal.sh
```

## Kết quả candidate v2 ngày 2026-10-12–13

Train run `sft-candidate-v2-train-20261012a` hoàn tất 200/200 optimizer step
trên T4 trong 4.651,59 giây. Training loss cuối run là 0,16495. Probe cố định
40 mẫu đo trên chính dữ liệu train cho thấy:

| Teacher-forced probe | Base trước train | Adapter step 200 |
| --- | ---: | ---: |
| Mean loss | 0,86249 | 0,03112 |
| Token accuracy | 89,99% | 99,07% |
| Exact sequence | 21/40 | 34/40 |

Run chỉ cập nhật 33.030.144 tham số LoRA FP32, tương đương 1,475% tổng số tham
số được report; base quantized bị đóng băng. Peak CUDA allocated là
6.644.648.448 byte và reserved là 10.750.001.152 byte. Adapter step 200 giữ
cùng state hash sau save/reload; toàn bộ chín gate train đạt.

Development run `sft-candidate-v2-development-20261013a` sinh greedy trên 80 ca
diagnostic đã lộ, không gọi policy, tool hoặc SQLite:

| Proposal exact | Base | Step 100 | Step 200 |
| --- | ---: | ---: | ---: |
| Toàn bộ | 66/80 | 70/80 | **74/80** |
| Tool-call | 48/56 | 53/56 | 51/56 |
| Clarification | 10/13 | 8/13 | **12/13** |
| No-action | 8/11 | 9/11 | **11/11** |
| Tool call ngoài yêu cầu | 6 | 7 | **1** |

Review thủ công đủ 24 phản hồi non-call cho kết quả base 15/24, step 100 17/24
và step 200 19/24. Step 100 bị loại vì tăng tool call ngoài yêu cầu từ sáu lên
bảy. Step 200 được chọn làm candidate cho phép đánh giá tiếp vì có điểm exact
cao nhất, phản hồi non-call tốt nhất và giảm tool call ngoài yêu cầu xuống một.
Quyết định này **không cho phép rollout** và vẫn giữ final holdout đóng.

Sáu lỗi automatic còn lại của step 200 gồm ba CREATE bị bỏ sót, một EDIT làm
mất dấu câu, một DEADLINE trích sai ranh giới thời gian và một câu mơ hồ vẫn tự
gọi truy vấn `today`. Review ngữ nghĩa còn phát hiện bốn câu trả lời dù không gọi
tool nhưng hỏi sai phần thiếu hoặc mô tả khả năng không chính xác. Vì vậy 74/80
chỉ là kết quả chọn checkpoint trên development, chưa phải bằng chứng
generalization hay điều kiện đưa model vào ứng dụng.

Thời gian train cộng với thời gian wall-clock development xấp xỉ 5.672,35 giây.
Theo đơn giá T4 đã khóa `0,000164 USD/giây`, phần GPU ước tính khoảng 0,930 USD;
CPU và memory do Modal tính riêng. Con số này thấp hơn trần tổng 1,50 USD đã được
duyệt, nhưng không thay thế hóa đơn thực tế của Modal.

Sau khi artifact đã được tải về đúng đường dẫn mặc định, xác minh độc lập bằng:

```sh
./scripts/demo/32_sft_candidate_modal.sh audit
```

Validator buộc report và review khớp hash, raw output, automatic summary,
adapter state và quyết định của từng ca non-call. Nó cũng buộc review giữ
`rollout_eligible=false` và `open_final_holdout=false`.
