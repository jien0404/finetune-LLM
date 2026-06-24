#!/usr/bin/env bash
# =============================================================================
# Smoke test trên máy L40S: verify TOÀN BỘ pipeline (data -> train -> eval) chạy
# hết không lỗi, KHÔNG tốn nhiều GPU. Chạy: bash scripts/smoke_test.sh
# Phải PASS trước khi chạy experiment thật.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== [1/4] Data prep (dry-run, không cần mạng) ==="
python src/data_prep.py --dry-run --out data/instruction_mix_v1 --devset-size 10

echo "=== [2/4] Train check (validate config + data, không nạp model) ==="
python src/train.py --config configs/exp/_smoke.yaml --check

echo "=== [3/4] Train 5 step trên model nhỏ ==="
python src/train.py --config configs/exp/_smoke.yaml --max-steps 5

echo "=== [4/4] Eval mini (VMLU subset 20 câu) ==="
python src/eval_vmlu.py --config configs/exp/_smoke.yaml --adapter results/runs/_smoke --subset 20 || {
  echo "!! Eval lỗi — kiểm tra eval_vmlu.py / kết nối tải VMLU."; exit 1;
}

echo ""
echo "=== SMOKE TEST PASS ==="
echo "Kiểm tra: loss có giảm? eval ra số? results/runs/_smoke/ có adapter?"
echo "Nếu OK -> chạy thật: bash scripts/run_experiment.sh configs/exp/qwen3-4b-lora.yaml"
