#!/usr/bin/env python3
"""Kiểm tra kết nối tới Weights & Biases TRƯỚC khi train.

Chạy: python scripts/check_wandb.py
- Exit 0 + in "PASS"     => W&B gọi được, pipeline sẽ log online.
- Exit 1 + in "FALLBACK" => sẽ tự fallback sang TensorBoard.

Đây là cổng quyết định backend tracking (yêu cầu của người dùng).
"""
import sys
from pathlib import Path

# cho phép chạy từ repo root mà không cần cài package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.tracking import check_wandb_connectivity  # noqa: E402


def main() -> int:
    ok, msg = check_wandb_connectivity()
    status = "PASS" if ok else "FALLBACK"
    print(f"[check_wandb] {status}: {msg}")
    if not ok:
        print(
            "[check_wandb] Gợi ý: `wandb login` hoặc đặt WANDB_API_KEY. "
            "Nếu máy bị chặn mạng, cứ để fallback TensorBoard — không chặn tiến độ."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
