#!/usr/bin/env bash
# =============================================================================
# Cài môi trường trên máy L40S (CUDA 12.x). Chạy: bash scripts/setup_env.sh
# Máy dev không-GPU: bỏ qua phần torch/unsloth/vllm, chỉ cài "dev minimal".
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Python: $(python --version 2>&1)"
echo "==> Repo:   $REPO_ROOT"

# 1) Virtualenv (khuyến nghị)
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

MODE="${1:-gpu}"   # "gpu" (mặc định) hoặc "dev" (CPU-only, máy không GPU)

if [ "$MODE" = "dev" ]; then
  echo "==> Cài DEV (CPU-only): chỉ đủ để verify data_prep/prompt/check_wandb."
  pip install datasets pyyaml pandas tqdm wandb transformers
  echo "==> Xong DEV. Lưu ý: train/eval cần GPU, chạy trên máy L40S."
  exit 0
fi

echo "==> Kiểm tra GPU:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
  echo "!! Không thấy nvidia-smi. Bạn có chắc đang ở máy L40S? Dùng 'dev' nếu máy không GPU."
fi

# 2) torch khớp CUDA — chỉnh index-url theo CUDA của máy (cu121/cu124...).
echo "==> Cài PyTorch (CUDA 12.4 wheel mặc định; sửa nếu CUDA khác):"
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3) Phần còn lại
echo "==> Cài requirements.txt:"
pip install -r requirements.txt

echo "==> Phiên bản chính:"
python - <<'PY'
import importlib
for m in ["torch", "transformers", "trl", "peft", "datasets", "bitsandbytes"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:14s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:14s} LỖI: {e}")
try:
    import torch
    print(f"  CUDA available: {torch.cuda.is_available()} | device: "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
except Exception:
    pass
PY

echo "==> Hoàn tất. Bước tiếp: python scripts/check_wandb.py && bash scripts/smoke_test.sh"
