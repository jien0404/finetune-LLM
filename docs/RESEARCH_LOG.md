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

<!-- Thêm mục mới phía dưới theo ngày -->
