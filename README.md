# Finetune LLM tiếng Việt — Qwen3 / Gemma3 (+ bản mới nhất)

Chương trình thực nghiệm 2 tuần để khám phá finetune LLM instruction-following tiếng Việt,
ưu tiên **Qwen3 & Gemma 3** (và các bản mới: Qwen3.5/3.6, Gemma 4), đánh giá chính bằng **VMLU**.

> **Mô hình làm việc:** code được viết trên máy **không-GPU**, đẩy qua GitHub, chạy trên máy
> **1× L40S (46GB VRAM)**. Mọi thứ điều khiển qua **YAML** (`configs/exp/*.yaml`), không sửa code
> để đổi model/hyperparam.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `configs/` | `base.yaml` (default) + `exp/*.yaml` (mỗi experiment) + `data/` (dataset mix) |
| `src/` | `data_prep.py`, `train.py`, `eval_vmlu.py`, `eval_general.py`, `infer.py`, `utils/` |
| `scripts/` | `setup_env.sh`, `check_wandb.py`, `smoke_test.sh`, `run_experiment.sh` |
| `docs/` | `RESEARCH_LOG.md` (nhật ký sống), `MODELS.md`, `DATASETS.md`, `EVAL.md` + survey gốc |
| `results/` | `LEADERBOARD.md` (bảng so sánh), `runs/<exp_id>/`, `plots/` |

## Quickstart

### Trên máy dev (không GPU) — verify logic trước khi push
```bash
bash scripts/setup_env.sh dev            # cài tối thiểu (CPU)
python scripts/check_wandb.py            # kiểm tra kết nối W&B (PASS/FALLBACK)
python src/data_prep.py --dry-run        # kiểm tra tải + làm sạch dữ liệu (mẫu nhỏ)
```

### Trên máy L40S
```bash
bash scripts/setup_env.sh                # cài full (GPU)
python scripts/check_wandb.py            # chốt backend tracking
bash scripts/smoke_test.sh               # 5 step + eval mini, KHÔNG tốn GPU lâu
bash scripts/run_experiment.sh configs/exp/qwen3-4b-lora.yaml   # train + eval + ghi leaderboard
CUDA_VISIBLE_DEVICES=1 bash scripts/run_experiment.sh configs/exp/qwen3-4b-lora.yaml
```

Mỗi run tạo `results/runs/<exp_id>/` (config snapshot, metrics, sample generations) và thêm 1 dòng
vào `results/LEADERBOARD.md`.

## Theo dõi & so sánh
- **`results/LEADERBOARD.md`** — bảng số liệu tự cập nhật mỗi run.
- **`docs/RESEARCH_LOG.md`** — nhật ký nghiên cứu theo ngày (quyết định, phát hiện, nguồn).
- **W&B** (nếu kết nối được) hoặc **TensorBoard** (`runs/tb/<exp_id>`).

Xem `docs/` cho chi tiết model/dataset/eval. Kế hoạch tổng thể: file plan đã được duyệt.
