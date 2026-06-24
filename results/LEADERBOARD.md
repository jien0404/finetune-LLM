# Leaderboard nội bộ — Finetune LLM tiếng Việt

Mỗi run được `run_experiment.sh` tự động thêm 1 dòng. VMLU = accuracy (%). Sắp xếp tay theo `vmlu_avg` khi tổng kết.

Cột `config` = tóm tắt hyperparam (rank/lr/epoch). `data_mix` = tên file mix. `aux_bench` = benchmark phụ (nếu bật).

| exp_id | model | method | config | data_mix | vmlu_avg | vmlu_stem | vmlu_social | vmlu_humanities | vmlu_other | aux_bench | vram_peak_gb | train_time | date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
