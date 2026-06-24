"""Đọc/ghi JSON và cập nhật leaderboard Markdown."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
LEADERBOARD = REPO_ROOT / "results" / "LEADERBOARD.md"

# Thứ tự cột của bảng leaderboard. Phải khớp header trong results/LEADERBOARD.md.
LEADERBOARD_COLUMNS: List[str] = [
    "exp_id",
    "model",
    "method",
    "config",
    "data_mix",
    "vmlu_avg",
    "vmlu_stem",
    "vmlu_social",
    "vmlu_humanities",
    "vmlu_other",
    "aux_bench",
    "vram_peak_gb",
    "train_time",
    "date",
]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_leaderboard_row(row: Dict[str, Any], path: str | Path = LEADERBOARD) -> None:
    """Thêm 1 dòng vào bảng leaderboard Markdown (tạo file + header nếu chưa có).

    `row` là dict, thiếu cột nào thì điền '-'. Mỗi run gọi hàm này 1 lần.
    """
    path = Path(path)
    ensure_dir(path.parent)

    if not path.exists():
        header = "# Leaderboard nội bộ — Finetune LLM tiếng Việt\n\n"
        header += (
            "Mỗi run được `run_experiment.sh` tự động thêm 1 dòng. "
            "VMLU = accuracy (%). Sắp xếp tay theo `vmlu_avg` khi tổng kết.\n\n"
        )
        header += "| " + " | ".join(LEADERBOARD_COLUMNS) + " |\n"
        header += "| " + " | ".join(["---"] * len(LEADERBOARD_COLUMNS)) + " |\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)

    cells = [str(row.get(col, "-")) for col in LEADERBOARD_COLUMNS]
    line = "| " + " | ".join(cells) + " |\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
