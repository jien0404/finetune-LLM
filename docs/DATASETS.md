# Datasets — instruction tiếng Việt

Mục tiêu: instruction-following tổng quát. Mix định nghĩa trong `configs/data/instruction_mix_v1.yaml`,
xử lý bởi `src/data_prep.py` → JSONL (`messages` format) trong `data/<mix_name>/`.

## Nguồn trong mix v1

| Tên | HF path (cần verify) | Mô tả | weight |
|---|---|---|---|
| bactrian-x-vi | `MBZUAI/Bactrian-X` (config `vi`) | Instruction dịch Alpaca/Dolly + response GPT-3.5, 52 ngôn ngữ; subset vi ~67K | 1.0 |
| vi-alpaca | `bkai-foundation-models/vi-alpaca` | Alpaca tiếng Việt (~25K), format instruction/input/output | 1.0 |
| 5cd-ai-cot-vi | `5CD-AI/Vietnamese-cot` | Chain-of-thought / reasoning tiếng Việt (cắt 20K) | 0.5 |

> **Lưu ý:** HF repo id ở trên cần xác nhận khi chạy thật (tên có thể đổi). Nếu một nguồn fail,
> `data_prep.py` sẽ báo lỗi tải — cập nhật path trong YAML, ghi vào RESEARCH_LOG. Có thể bổ sung
> dữ liệu task có nhãn (ViQuAD, ViNLI) chuyển thành instruction để tăng đa dạng (mix v2).

## Pipeline làm sạch (`data_prep.py`)
1. Chuẩn hoá Unicode **NFC** (đồng nhất dấu tiếng Việt).
2. Loại record rỗng (thiếu instruction hoặc output).
3. Lọc độ dài output: `[min_chars_output, max_chars_output]`.
4. **Lọc độc hại**: blacklist + heuristic (`is_toxic`). *Tối thiểu* — nên nâng cấp bằng classifier
   ViHSD cho production (ghi chú trong code).
5. **Khử trùng** theo hash(instruction+output).
6. Chuyển sang `messages` (system/user/assistant) qua template `vi_instruct_v1`.
7. Cắt **train / val (10%)** + **devset cố định** (200–300 mẫu) cho sanity định tính.

## Output
```
data/<mix_name>/
├── train.jsonl      # mỗi dòng: {instruction, input, output, messages, _source}
├── val.jsonl
├── devset.jsonl     # subset cố định để so base vs tuned bằng infer.py
└── summary.json     # n_total/train/val/dev, avg độ dài
```

## Verify không cần mạng/GPU
```bash
python src/data_prep.py --dry-run --out data/_dryrun   # sinh mẫu giả, kiểm tra logic
```
Khi chạy thật (máy L40S, có mạng):
```bash
python src/data_prep.py --mix configs/data/instruction_mix_v1.yaml
```

## Đánh giá — xem `EVAL.md`
VMLU là benchmark chính (không nằm trong tập train). Giữ devset tách biệt để review tay.
