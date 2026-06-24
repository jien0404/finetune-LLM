1. VRAM thực của Qwen3-4B + Liger đã giảm xuống 15GB rồi
2. Kết quả ban đầu của leaderboard:
# Leaderboard nội bộ — Finetune LLM tiếng Việt

Mỗi run được `run_experiment.sh` tự động thêm 1 dòng. VMLU = accuracy (%). Sắp xếp tay theo `vmlu_avg` khi tổng kết.

Cột `config` = tóm tắt hyperparam (rank/lr/epoch). `data_mix` = tên file mix. `aux_bench` = benchmark phụ (nếu bật).

| exp_id | model | method | config | data_mix | vmlu_avg | vmlu_stem | vmlu_social | vmlu_humanities | vmlu_other | aux_bench | vram_peak_gb | train_time | date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-4b_lora-r16_lr2e4_ep2_BASE | Qwen/Qwen3-4B-Instruct-2507 | lora (base) | - | instruction_mix_v1 | 63.71 | - | - | - | 63.71 | - | - | - | 2026-06-24 |
| qwen3-4b_lora-r16_lr2e4_ep1_BASE | Qwen/Qwen3-4B-Instruct-2507 | lora (base) | - | instruction_mix_v1 | 63.71 | - | - | - | 63.71 | - | - | - | 2026-06-24 |
| gemma3-4b_lora-r16_lr2e4_ep1_BASE | google/gemma-3-4b-it | lora (base) | - | instruction_mix_v1 | 52.42 | - | - | - | 52.42 | - | - | - | 2026-06-24 |