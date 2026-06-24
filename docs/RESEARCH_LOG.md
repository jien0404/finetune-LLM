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

## 2026-06-24 — Ngày 1 (chạy thật trên L40S): gỡ loạt lỗi môi trường

Smoke test trên L40S (Python 3.11, torch 2.6.0+cu124, transformers 4.57, TRL 0.21) phát hiện và fix:
1. **Python 3.8** trên máy → transformers ≥4.57 không cài được. Fix: venv `uv --python 3.11`.
2. **W&B check sai** (ping HEAD root → 404 dù đã login). Fix: dùng `wandb.Api().viewer`. → PASS (user logged in).
3. **`configs/data/*.yaml` không có trên máy** vì `.gitignore` `data/` nuốt luôn `configs/data/`. Fix: neo `/data/`.
4. **torchao crash** (cần torch≥2.11 nhưng vLLM ghim torch 2.6.0) làm transformers import fail. Fix: gỡ torchao (ta dùng bitsandbytes).
5. **TRL 0.21 đổi API**: `SFTConfig.max_seq_length`→`max_length`; `SFTTrainer(tokenizer=)`→`processing_class=`.
6. **Unsloth ↔ TRL 0.21 xung đột**: Unsloth trả logits rỗng khi train (tiết kiệm VRAM) → TRL metric
   `entropy_from_logits` reshape tensor 0 phần tử → crash. **Quyết định: tạm tắt Unsloth** (`use_unsloth: false`),
   dùng transformers+PEFT thuần — ổn định, trên L40S 46GB vẫn đủ VRAM. Việc tương lai: thử cặp version
   Unsloth/TRL tương thích để bật lại (lợi ích ~2x tốc độ, ~70% VRAM) cho model lớn.

**Bài học:** stack HF mid-2026 (transformers 5.x sắp tới, TRL 0.21) đổi API nhanh — pin version vào
`requirements.txt` khi đã có cấu hình chạy được trên L40S.

## 2026-06-24 — Ngày 1: SMOKE TEST PASS (pipeline end-to-end xanh) ✅

Sau 8 lần gỡ lỗi môi trường/API, smoke test chạy hết: data → train → eval.

**Kết quả smoke (Qwen3-0.6B, 5 step, data tí hon — chỉ để verify):**
- Train: loss 1.63 → 0.043, eval_loss 0.024, **VRAM peak 1.11GB**, train_time 8s, adapter lưu OK.
- Eval VMLU (proxy `ura-hcmut/vmlu_vi` valid, 744 câu có nhãn, subset 20): `vmlu_avg=25.0` =
  **đúng mức ngẫu nhiên 4-choice** → xác nhận eval nối đúng (không phải tín hiệu chất lượng).
- W&B log online OK (project finetune-llm-vi).

**Các fix API/môi trường lần lượt (stack HF mid-2026):**
7. TRL 0.21 đổi `max_seq_length`→`max_length`, `tokenizer=`→`processing_class=`.
8. **Bỏ hẳn TRL SFTTrainer** (bug entropy_from_logits với logits rỗng) → dùng `transformers.Trainer`
   + completion-only masking tự viết (mask prompt=-100). Ổn định, không phụ thuộc API TRL.
9. `apply_chat_template(tokenize=True)` trả Encoding → pyarrow lỗi; chuyển sang lấy string rồi tokenize.
10. VMLU: repo đoán không tồn tại / gated (anhdungitvn 401) → đọc thẳng `ura-hcmut/vmlu_vi` valid.jsonl
    qua JSON loader (bypass script lỗi cột). **Lưu ý:** test set đầy đủ không có nhãn → số chính thức nộp vmlu.ai.
11. `apply_chat_template(return_tensors=pt)` trả BatchEncoding → `model(**inputs)` thay vì `model(inputs)`.

**Hạn chế đã biết (xử lý sau):**
- Proxy VMLU valid (744 câu) không có trường `subject` → điểm theo nhóm dồn hết vào `other`. Bộ test đầy
  đủ có subject; map nhóm sẽ hoạt động khi chấm trên đó. Đủ dùng để iterate cục bộ.
- Unsloth đang tắt (xung đột TRL) — nhưng giờ đã bỏ TRL, có thể thử bật lại Unsloth để tăng tốc/giảm VRAM
  cho model lớn (việc tương lai, cần kiểm tra Unsloth tương thích `transformers.Trainer`).

**Sẵn sàng chạy thật:** `bash scripts/run_experiment.sh configs/exp/qwen3-4b-lora.yaml`.

## 2026-06-24 — Ngày 1: Nghiên cứu VRAM — Qwen3-4B LoRA tốn 36GB (kỳ vọng ~18GB)

**Câu hỏi:** Vì sao Qwen3-4B LoRA (weights bf16 chỉ 8GB) lại ngốn 36GB VRAM?

**Nguyên nhân (đã xác minh qua tài liệu):** **cross-entropy trên vocab khổng lồ** (Qwen3 = 151.936,
Gemma = 256k). Khi tính loss, transformers materialize tensor logits `[batch, seq, vocab]`, **upcast fp32**
cho CE, và giữ gradient của nó:
```
logits bf16   8×1024×152k×2B ≈ 2.4GB
logits fp32 (.float() trong CE) ≈ 4.7GB
grad d_logits fp32             ≈ 4.7GB   -> "đầu loss" spike ~12-15GB
+ weights 8GB + activation + CUDA ctx + reserved của caching allocator -> nvidia-smi 36GB
```
Tài liệu (Towards Data Science / Liger) mô tả đúng case: **"84% reduction, from 36GB to 5GB"**. Đây là
hiện tượng kinh điển của model vocab lớn, không phải bug. Lưu ý: `nvidia-smi` hiển thị memory **reserved**
(caching allocator giữ), cao hơn `torch.cuda.max_memory_allocated()` (peak thật, đang ghi vào metric).

**Giải pháp đã áp dụng: Liger Kernel** (`use_liger_kernel: true`). Fused linear cross-entropy tính loss theo
chunk, **không materialize full logits** → giảm 60-80% VRAM phần loss + nhanh ~20%. Hỗ trợ Qwen3
(`apply_liger_kernel_to_qwen3`) và Gemma3. Cần `pip install liger-kernel` (transformers ≥4.52). Đã thêm flag
+ guard (tự bỏ qua nếu chưa cài). **VRAM kỳ vọng sau Liger: ~15-18GB** (đúng ước tính ban đầu).

**Nguồn:**
- Liger-Kernel — github.com/linkedin/Liger-Kernel ; docs linkedin.github.io/Liger-Kernel
- "Cutting LLM Memory by 84%: A Deep Dive into Fused Kernels" — towardsdatascience.com
- PyTorch blog: torchtune + torch.compile & Liger Kernel

**Hệ quả:** sau khi bật Liger, có thể cân nhắc tắt gradient checkpointing hoặc tăng batch để nhanh hơn nữa
(việc tối ưu tiếp, đo VRAM thực rồi quyết).

## 2026-06-24 — Phân tích kết quả Qwen3-4B + xoay hướng (base model + data uy tín)

**Kết quả (proxy VMLU 744 câu):**
- Qwen3-4B-Instruct: base 63.71 → tuned 62.1 (**−1.61**). VRAM 14.57GB (Liger ✓), 5h18m.
- Qwen3.5-4B: base **24.06** (≈ ngẫu nhiên).

**Phân tích:**
1. **−1.61 nằm trong nhiễu** (SE trên 744 câu ≈ ±1.8%) → coi như đi ngang, KHÔNG phải regression thật.
   Kiến thức được giữ (không catastrophic forgetting).
2. **Vì sao SFT không cải thiện:** đang tune `Qwen3-4B-Instruct` (model ĐÃ instruction-tune chất lượng cao)
   bằng data **dịch máy** (Bactrian-X + vi-alpaca) → không thể vượt, chỉ giữ/giảm nhẹ trên benchmark kiến thức.
3. **Qwen3.5-4B = 24%:** `Qwen/Qwen3.5-4B` là model **BASE** (chưa instruct) → 0-shot MCQA ra ngẫu nhiên.
   Đây là model mà SFT sẽ tạo cú nhảy lớn (base→tuned).
4. **Lệch tác vụ/thước đo:** VMLU đo KIẾN THỨC, không đo tuân lệnh (thứ SFT cải thiện) → VMLU gần như không
   bao giờ phản ánh giá trị SFT khi tune model đã-instruct.

**Quyết định (theo người dùng):**
- **Finetune từ BASE models** (Qwen3-4B-Base — đã xác nhận có chat_template) → base→tuned delta rõ.
- **Data uy tín hơn (mix v2):** **Aya Dataset** (`CohereForAI/aya_dataset`, lọc `language=="Vietnamese"`,
  **8.676 cặp do người bản xứ viết**, ACL 2024 Cohere) + **bkai vi-alpaca** (lab HUST). **BỎ Bactrian-X** (dịch máy).
- Thêm hỗ trợ `filter` trong data_prep để lọc subset tiếng Việt của Aya.

**Tham khảo data uy tín cho LLM tiếng Việt (research-grade):**
- Aya Dataset/Collection (Cohere, human-annotated) — acl 2024.
- bkai-foundation-models/* (HUST), Vistral, SeaLLM, PhoGPT (mô tả data trong paper).
- Đánh giá: VMLU (Zalo), VLUE (UIT), VLSP-LLM 2023 — đây là *benchmark eval*, khác data *train*.

**Nguồn:** Aya — aclanthology.org/2024.acl-long.620 ; Vistral — huggingface.co/Viet-Mistral/Vistral-7B-Chat ;
awesome-vietnamese-nlp — github.com/vndee/awsome-vietnamese-nlp

<!-- Thêm mục mới phía dưới theo ngày -->
