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
