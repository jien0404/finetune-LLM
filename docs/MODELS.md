# Model — bảng cập nhật & lý do chọn (06/2026)

Ưu tiên: **Qwen3 & Gemma 3** (phổ biến, benchmark tốt) + thử **bản mới nhất** (Qwen3.5/3.6, Gemma 4).
Phần cứng mục tiêu: **1× L40S, 46GB VRAM**.

## Quy tắc fit VRAM (tham chiếu Unsloth)
- LoRA **bf16** ≈ `2.5 × số_tỉ_params` GB → ví dụ 4B≈10GB, 8B≈22GB, 12B≈30GB, 14B≈36GB.
- QLoRA **4-bit** ≈ `0.7 × số_tỉ_params` GB → 27B≈19GB, 32B≈22GB (weights); cộng activation/optimizer.
- ⇒ **≤12–14B**: LoRA bf16 thoải mái. **27–35B**: dùng QLoRA 4-bit.

## Danh sách model & vai trò

| HF repo id | Họ | Size | Cách chạy (L40S) | Vai trò | Ưu tiên |
|---|---|---|---|---|---|
| `Qwen/Qwen3-4B-Instruct-2507` | Qwen3 | 4B | LoRA bf16 | **Baseline chính #1** (đứng đầu bake-off) | Cao |
| `Qwen/Qwen3-8B` | Qwen3 | 8B | LoRA bf16 | Scale size | Cao |
| `Qwen/Qwen3-14B` | Qwen3 | 14B | LoRA bf16 (sát) / QLoRA | So lớn hơn | TB |
| `google/gemma-3-4b-it` | Gemma 3 | 4B | LoRA bf16 | **Baseline chính #2** | Cao |
| `google/gemma-3-12b-it` | Gemma 3 | 12B | LoRA bf16 | Scale size | Cao |
| `google/gemma-3-27b-it` | Gemma 3 | 27B | QLoRA 4-bit | Trần chất lượng Gemma3 | TB |
| `google/gemma-4-12b-it` | Gemma 4 | 12B | LoRA bf16 | Bản mới vs Gemma3-12B | Cao |
| `Qwen/Qwen3.5-4B` | Qwen3.5 | 4B | LoRA bf16 | Thế hệ mới vs Qwen3-4B | Cao |
| `google/gemma-4-31b` | Gemma 4 | 31B | QLoRA 4-bit | Trần chất lượng (Apache-2.0) | TB |
| `Qwen/Qwen3-32B` | Qwen3 | 32B | QLoRA 4-bit | Trần chất lượng Qwen3 | TB |
| `Qwen/Qwen3.6-27B` | Qwen3.6 | 27B | QLoRA 4-bit | Coding-focused → instruction phụ | Thấp |
| SEA-LION v3 (9B-IT) | tham chiếu | 9B | LoRA bf16 / eval-only | Baseline đã-tune-tiếng-Việt | Tùy chọn |

> **Lưu ý repo id**: tên chính xác cần xác nhận trên HuggingFace lúc chạy (một số có hậu tố `-it`,
> `-Instruct`, hoặc collection riêng). Cập nhật vào đây khi đã verify trên máy L40S.

## Rủi ro tương thích
- Model rất mới (Qwen3.5/3.6, Gemma 4) có thể cần `transformers` bản mới + Unsloth bản mới. Nếu Unsloth
  chưa hỗ trợ → đặt `model.use_unsloth: false` trong YAML để fallback TRL+PEFT thuần. Ghi lại vào RESEARCH_LOG.
- MoE (Qwen3-30B-A3B, Qwen3.5-35B-A3B, Gemma-4-26B-A4B): LoRA bf16 cần giữ toàn bộ params 16-bit → quá lớn;
  **bắt buộc QLoRA 4-bit**.
