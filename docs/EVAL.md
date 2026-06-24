# Đánh giá (Evaluation)

Giữ **nhất quán xuyên suốt** để leaderboard công bằng giữa các model/cấu hình.

## 1. VMLU — benchmark chính
- **Là gì:** 10.880 câu trắc nghiệm 4 lựa chọn, 58 chủ đề / 4 nhóm (STEM, Social Science, Humanities, Other),
  từ tiểu học → chuyên nghiệp. Leaderboard chuẩn cộng đồng VN.
- **Cách chấm (`src/eval_vmlu.py`):** với mỗi câu, tính log-prob token đáp án "A"/"B"/"C"/"D", chọn cao nhất,
  so ground-truth. Robust hơn parse text tự do. Báo **accuracy tổng + theo 4 nhóm**.
- **Subset:** mặc định 2000 câu **phân tầng theo nhóm** để lặp nhanh; run cuối/đối thủ chính chạy full
  (`--subset` để override, hoặc `eval.vmlu.subset: null`).
- **Lưu ý nhãn (quan trọng):** test set chính thức VMLU (10.880 câu) **không công khai đáp án** — phải nộp
  prediction lên https://vmlu.ai/submit để lấy số trên leaderboard. Vì vậy:
  - **Dev/sanity cục bộ:** `eval_vmlu.py` tự tải tập **có nhãn public** `ura-hcmut/vmlu_vi` (split `valid.jsonl`,
    ~100 câu) để chấm accuracy nhanh khi lặp. Đây là **proxy nhỏ**, không đại diện đủ 58 chủ đề.
  - **Số chính thức:** chạy trên test set đầy đủ (unlabeled) để sinh `preds.json`/submission rồi nộp vmlu.ai.
  - Có dataset đầy đủ có nhãn (vd `anhdungitvn/vmlu_v1.5`) nhưng **gated** — cần `huggingface-cli login` + xin quyền.
  - Hoặc tự tải file dev có nhãn rồi truyền `--data-path file.jsonl`.

```bash
# baseline (model gốc) — chạy 1 lần cho mỗi base model
python src/eval_vmlu.py --config configs/exp/qwen3-4b-lora.yaml --base-only
# sau finetune
python src/eval_vmlu.py --config configs/exp/qwen3-4b-lora.yaml --adapter results/runs/<exp_id>
```

**Sanity:** accuracy của **base model** phải xấp xỉ số công bố trên VMLU leaderboard → xác nhận eval đúng
trước khi tin số của model đã tune. Ghi lại đối chiếu vào RESEARCH_LOG.

## 2. Benchmark phụ — lm-evaluation-harness (`src/eval_general.py`)
Bật ở run cuối để bắt regression (vd `m_mmlu_vi` nếu khả dụng, hoặc bộ MCQA tiếng Việt khác). Khai báo trong
config `eval.aux.tasks`. **Verify task tồn tại trong lm-eval** khi chạy; nếu không có task tiếng Việt phù hợp,
ghi chú và bỏ qua — VMLU vẫn là thước đo chính.

## 3. Định tính (`src/infer.py`)
So **base vs tuned** trên devset (200–300 mẫu cố định), lưu `samples.jsonl` để review tay: kiểm tra trôi
chảy tiếng Việt, bám instruction, hallucination, trả lời ngoài scope.
```bash
python src/infer.py --config ... --adapter results/runs/<exp_id> --n 20 --compare-base
```

## Metric ghi vào leaderboard
`vmlu_avg`, `vmlu_stem`, `vmlu_social`, `vmlu_humanities`, `vmlu_other` (accuracy %), `aux_bench` (nếu bật),
kèm `vram_peak_gb`, `train_time`. So với **dòng base** của cùng model để biết finetune có cải thiện không.
