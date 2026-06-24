#!/usr/bin/env python3
"""Gộp kết quả 1 run vào results/LEADERBOARD.md (gọi bởi run_experiment.sh).

Đọc train_metrics.json + vmlu_tuned.json (hoặc vmlu_base.json) trong run dir,
ghép thành 1 dòng leaderboard.

Dùng:
  python src/report.py --config configs/exp/qwen3-4b-lora.yaml --tag tuned
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import load_config  # noqa: E402
from utils.io import append_leaderboard_row, load_json  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="tuned", choices=["tuned", "base"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    exp_id = cfg["_meta"]["exp_id"]
    run_dir = REPO_ROOT / "results" / "runs" / exp_id

    train_m = {}
    tm_path = run_dir / "train_metrics.json"
    if tm_path.exists():
        train_m = load_json(tm_path)

    vmlu = {}
    vmlu_path = run_dir / f"vmlu_{args.tag}.json"
    if vmlu_path.exists():
        vmlu = load_json(vmlu_path)

    lora = cfg["lora"]
    train = cfg["train"]
    config_summary = f"r{lora['r']}/lr{train['learning_rate']}/ep{train['num_train_epochs']}"
    import yaml

    mix = yaml.safe_load(open(REPO_ROOT / cfg["data"]["mix_config"], encoding="utf-8"))

    row = {
        "exp_id": exp_id + ("" if args.tag == "tuned" else "_BASE"),
        "model": cfg["model"]["name"],
        "method": cfg["method"] + (" (base)" if args.tag == "base" else ""),
        "config": config_summary if args.tag == "tuned" else "-",
        "data_mix": mix.get("name", "-"),
        "vmlu_avg": vmlu.get("vmlu_avg", "-"),
        "vmlu_stem": vmlu.get("vmlu_stem", "-"),
        "vmlu_social": vmlu.get("vmlu_social_science", vmlu.get("vmlu_social", "-")),
        "vmlu_humanities": vmlu.get("vmlu_humanities", "-"),
        "vmlu_other": vmlu.get("vmlu_other", "-"),
        "aux_bench": "-",
        "vram_peak_gb": train_m.get("vram_peak_gb", "-"),
        "train_time": train_m.get("train_time", "-"),
        "date": str(date.today()),
    }
    append_leaderboard_row(row)
    print(f"[report] Đã thêm dòng leaderboard cho {row['exp_id']} (vmlu_avg={row['vmlu_avg']}).")


if __name__ == "__main__":
    main()
