# Nhật ký nghiên cứu — Finetune LLM tiếng Việt

> Append theo ngày. Mỗi mục: **quyết định / phát hiện / lỗi gặp + cách fix / nguồn tham khảo**.
> Đây là tài liệu *sống*, đi song song với `results/LEADERBOARD.md`.

---

## 2026-06-24 — Ngày 0: Khởi động & cập nhật bối cảnh

**Bối cảnh.** Có survey gốc (`deep-research-report-tuneLLM.md`) nhưng số liệu cũ (Gemma 7B, LLaMA-2,
Phi-3). Mục tiêu: chương trình thực nghiệm 2 tuần, instruction-following tổng quát tiếng Việt, eval
bằng VMLU. Ưu tiên Qwen3/Gemma3 + bản mới nhất. Hạ tầng: code ở máy không-GPU → GitHub → L40S 46GB.

**Cập nhật lineup model (06/2026) — đã tra cứu:**
- **Qwen3** (05/2025): 4B-Instruct-2507, 8B, 14B, 32B, 30B-A3B (MoE). Qwen3-4B-Instruct-2507 đứng đầu
  nhiều bake-off finetune cộng đồng; họ Qwen3 chiếm 4/6 vị trí đầu.
- **Qwen3.5** (02/2026): 0.8B, 4B, 35B-A3B, 397B-A17B.
- **Qwen3.6** (04/2026): 27B dense, 35B-A3B — **thiên về coding** → ưu tiên thấp cho instruction tổng quát.
- **Gemma 3** (2025): 4B, 12B, 27B.
- **Gemma 4** (04/2026): E4B, 12B, 26B-A4B (MoE), 31B dense — **Apache-2.0**, day-one HF/TRL.
- **SEA-LION v3** (Gemma/Llama 9B-IT): đã tune sẵn tiếng Việt/SEA → dùng làm baseline mốc.

**Fit trên L40S 46GB (quy tắc Unsloth):** LoRA bf16 ≈ 2.5×params GB; QLoRA 4-bit ≈ 0.7×params GB.
→ ≤12B: LoRA bf16 thoải mái. 27–35B: cần QLoRA 4-bit.

**Quyết định công cụ.** Unsloth + TRL `SFTTrainer` (fallback TRL+PEFT thuần nếu Unsloth chưa hỗ trợ model
mới). vLLM + lm-eval-harness cho eval. Tracking: check W&B trước, fallback TensorBoard.

**Nguồn:**
- Qwen3 Technical Report — arxiv.org/pdf/2505.09388
- distil labs SLM fine-tuning bake-off — distillabs.ai/blog/we-benchmarked-12-small-language-models...
- Unsloth Qwen fine-tune docs — unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune
- VMLU leaderboard — vmlu.ai/leaderboard ; ACL 2025 VMLU toolkit — aclanthology.org/2025.acl-long.563
- Gemma 4 — blog.google/innovation-and-ai/technology/developers-tools/gemma-4/

**TODO tiếp:** dựng scaffold (xong Ngày 1) → data pipeline (Ngày 2) → train+smoke (Ngày 3).

---

## 2026-06-24 — Ngày 1–5: Dựng pipeline (viết trên máy không-GPU, verify trên CPU)

Đã hoàn thiện toàn bộ phần code/scaffold có thể làm mà không cần GPU. Mọi thứ điều khiển qua YAML.

**Đã làm & đã verify trên CPU:**
- **Config**: `base.yaml` + `exp/*.yaml`, merge sâu (override thắng). Verify load 8 config → exp_id đúng.
- **Tracking**: `utils/tracking.py` + `scripts/check_wandb.py` — cổng quyết định backend. Test: máy này chưa
  cài wandb → in `FALLBACK` đúng như thiết kế (sẽ dùng TensorBoard).
- **Data**: `data_prep.py` (NFC, dedup, lọc rỗng/độ dài/độc hại → `messages` → split train/val/devset).
  Verify `--dry-run` (không cần mạng): 150 mẫu giả → train/val/devset + summary.json OK.
- **Train**: `train.py` (Unsloth → fallback TRL+PEFT). `--check` validate config+data trên CPU OK. Import
  nặng (torch/unsloth) hoãn vào trong hàm → script parse được trên máy không-GPU.
- **Eval**: `eval_vmlu.py` (chấm bằng log-prob A/B/C/D, accuracy tổng + 4 nhóm, subset phân tầng),
  `eval_general.py` (wrapper lm-eval), `infer.py` (so base vs tuned trên devset).
- **Tự động hoá**: `report.py` + `run_experiment.sh` (data→eval base→train→eval tuned→infer→ghi leaderboard).
  Verify `report.py` append đúng 1 dòng vào `LEADERBOARD.md` với metrics giả.
- **Configs sẵn**: Qwen3-4B/8B, Gemma3-4B/12B, Gemma4-12B, Qwen3.5-4B (LoRA bf16), Gemma4-31B (QLoRA), _smoke.
- **Docs**: README, MODELS, DATASETS, EVAL.

**Quyết định kỹ thuật đáng ghi:**
- Chấm VMLU bằng **log-prob của token đáp án** thay vì parse text → ổn định, không phụ thuộc model có chịu
  trả lời đúng format hay không.
- Hoãn import torch/unsloth vào trong hàm để **toàn bộ pipeline validate được trên máy không-GPU** trước khi push.
- Repo id của model **mới nhất** (Qwen3.5/Gemma4) đặt theo dự đoán → **phải verify trên HF khi chạy thật**
  (đã ghi chú trong từng YAML + MODELS.md).

**Việc cần làm trên máy L40S (chưa làm được ở đây vì không GPU):**
1. `bash scripts/setup_env.sh` → `python scripts/check_wandb.py` → `bash scripts/smoke_test.sh`.
2. Verify repo id model + repo dataset HF thật (Bactrian-X vi, vi-alpaca, 5CD-AI; VMLU mirror).
3. Chạy baseline Qwen3-4B + Gemma3-4B, đối chiếu **VMLU base** với leaderboard công bố (sanity eval).

**Chưa push GitHub:** repo mới `git init` + commit local; **chưa có remote**. Cần người dùng tạo repo GitHub
rồi `git remote add origin ... && git push -u origin main`.

<!-- Thêm mục mới phía dưới theo ngày -->
