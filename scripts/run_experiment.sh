#!/usr/bin/env bash
# =============================================================================
# Chạy 1 experiment đầu-cuối: (data nếu thiếu) -> [eval base lần đầu] -> train ->
# eval tuned -> ghi leaderboard. Một lệnh.
#
#   bash scripts/run_experiment.sh configs/exp/qwen3-4b-lora.yaml
#   EVAL_BASE=1 bash scripts/run_experiment.sh configs/exp/qwen3-4b-lora.yaml   # ép eval base
#   VMLU_SUBSET=full bash scripts/run_experiment.sh configs/exp/...             # eval full VMLU
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${1:?Cần đường dẫn config, vd configs/exp/qwen3-4b-lora.yaml}"
EVAL_BASE="${EVAL_BASE:-auto}"      # auto: eval base nếu chưa có vmlu_base.json
SUBSET_ARG=""
if [ "${VMLU_SUBSET:-}" = "full" ]; then SUBSET_ARG="--subset 0"; fi  # 0 -> hiểu là full (xem ghi chú)

EXP_ID="$(python - "$CONFIG" <<'PY'
import sys; sys.path.insert(0, "src")
from utils.config import load_config
print(load_config(sys.argv[1])["_meta"]["exp_id"])
PY
)"
RUN_DIR="results/runs/${EXP_ID}"
echo "==> Experiment: ${EXP_ID}"

# 0) Dữ liệu
DATA_DIR="data/$(python - "$CONFIG" <<'PY'
import sys, yaml; sys.path.insert(0,"src")
from utils.config import load_config
cfg=load_config(sys.argv[1])
print(yaml.safe_load(open(cfg["data"]["mix_config"]))["name"])
PY
)"
if [ ! -f "${DATA_DIR}/train.jsonl" ]; then
  echo "==> Chưa có dữ liệu, chạy data_prep..."
  python src/data_prep.py --mix "$(python - "$CONFIG" <<'PY'
import sys; sys.path.insert(0,"src")
from utils.config import load_config
print(load_config(sys.argv[1])["data"]["mix_config"])
PY
)"
fi

# 1) Eval base (1 lần / base model) để có mốc so sánh
if [ "$EVAL_BASE" = "1" ] || { [ "$EVAL_BASE" = "auto" ] && [ ! -f "${RUN_DIR}/vmlu_base.json" ]; }; then
  echo "==> Eval BASE (mốc so sánh)..."
  python src/eval_vmlu.py --config "$CONFIG" --base-only $SUBSET_ARG
  python src/report.py --config "$CONFIG" --tag base
fi

# 2) Train
echo "==> Train..."
python src/train.py --config "$CONFIG"

# 3) Eval tuned
echo "==> Eval TUNED..."
python src/eval_vmlu.py --config "$CONFIG" --adapter "$RUN_DIR" $SUBSET_ARG

# 4) Sinh mẫu định tính (so base vs tuned, 20 mẫu)
echo "==> Sinh mẫu định tính..."
python src/infer.py --config "$CONFIG" --adapter "$RUN_DIR" --n 20 --compare-base || \
  echo "!! infer lỗi (không chặn) — bỏ qua."

# 5) Ghi leaderboard
python src/report.py --config "$CONFIG" --tag tuned

echo ""
echo "==> XONG ${EXP_ID}. Xem results/LEADERBOARD.md và ${RUN_DIR}/"
