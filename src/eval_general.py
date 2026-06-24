#!/usr/bin/env python3
"""Benchmark phụ qua lm-evaluation-harness (bắt regression, bật ở run cuối).

Wrapper mỏng gọi lm-eval với model HF + adapter PEFT. Task tiếng Việt/đa ngôn ngữ
khai báo trong config (`eval.aux.tasks`) — verify task khả dụng trong lm-eval khi chạy.

Dùng:
  python src/eval_general.py --config configs/exp/qwen3-4b-lora.yaml \
        --adapter results/runs/<exp_id> --tasks m_mmlu_vi
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import load_config  # noqa: E402
from utils.io import ensure_dir  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tasks", default=None, help="csv task lm-eval; mặc định lấy từ config")
    ap.add_argument("--limit", type=int, default=None, help="giới hạn số mẫu mỗi task")
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp_id = cfg["_meta"]["exp_id"]
    tasks = args.tasks or ",".join(cfg.get("eval", {}).get("aux", {}).get("tasks", []))
    if not tasks:
        print("[eval_general] Không có task nào (eval.aux.tasks rỗng). Bỏ qua.")
        return

    model_args = f"pretrained={cfg['model']['name']},dtype=bfloat16"
    if args.adapter:
        model_args += f",peft={args.adapter}"

    out_dir = ensure_dir(REPO_ROOT / "results" / "runs" / exp_id / "lm_eval")
    cmd = [
        "lm_eval", "--model", "hf",
        "--model_args", model_args,
        "--tasks", tasks,
        "--batch_size", "auto",
        "--output_path", str(out_dir),
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    print("[eval_general] $", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[eval_general] Kết quả trong {out_dir}/. Tổng hợp số vào leaderboard cột aux_bench.")


if __name__ == "__main__":
    main()
