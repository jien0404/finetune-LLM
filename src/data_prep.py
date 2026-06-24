#!/usr/bin/env python3
"""Chuẩn bị dữ liệu instruction tiếng Việt cho SFT.

Đọc `configs/data/<mix>.yaml` -> tải từng nguồn HF -> chuẩn hoá cột về
(instruction, input, output) -> làm sạch (Unicode NFC, dedup, lọc rỗng/quá dài,
lọc độc hại) -> chuyển sang `messages` -> cắt train/val + dev-set cố định ->
lưu ra JSONL.

Dùng:
  python src/data_prep.py --mix configs/data/instruction_mix_v1.yaml --out data/instruction_mix_v1
  python src/data_prep.py --dry-run        # không tải mạng, tạo mẫu giả để verify logic (chạy được trên CPU)

Chạy được trên máy KHÔNG-GPU. `--dry-run` để kiểm tra pipeline trước khi push.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt_templates import build_messages  # noqa: E402
from utils.io import ensure_dir  # noqa: E402
from utils.seeding import set_seed  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# Blacklist tối thiểu cho lọc độc hại heuristic. Thực tế nên thay/bổ sung bằng
# classifier (vd ViHSD) — xem RESEARCH_LOG. Giữ ngắn gọn ở đây.
TOXIC_BLACKLIST = {
    "đm", "đmm", "vcl", "vl", "địt", "lồn", "cặc", "đụ", "đéo",
}


def normalize_unicode(text: str, form: str = "NFC") -> str:
    return unicodedata.normalize(form, text or "")


def is_toxic(text: str) -> bool:
    low = (text or "").lower()
    tokens = set(low.replace(",", " ").replace(".", " ").split())
    return bool(tokens & TOXIC_BLACKLIST)


def _hash_example(instruction: str, output: str) -> str:
    return hashlib.sha1(f"{instruction}\x00{output}".encode("utf-8")).hexdigest()


def clean_record(
    rec: Dict[str, str], cleaning: Dict, system_prompt: str, template: str
) -> Optional[Dict]:
    """Trả về record đã chuẩn hoá (kèm `messages`) hoặc None nếu bị loại."""
    instruction = normalize_unicode(rec.get("instruction", ""), cleaning.get("unicode_normalize", "NFC"))
    input_text = normalize_unicode(rec.get("input", ""), cleaning.get("unicode_normalize", "NFC"))
    output = normalize_unicode(rec.get("output", ""), cleaning.get("unicode_normalize", "NFC"))

    instruction, input_text, output = instruction.strip(), input_text.strip(), output.strip()

    if cleaning.get("drop_empty", True) and (not instruction or not output):
        return None
    n_out = len(output)
    if n_out < cleaning.get("min_chars_output", 1):
        return None
    if n_out > cleaning.get("max_chars_output", 10**9):
        return None
    if cleaning.get("toxicity_filter", True) and (is_toxic(instruction) or is_toxic(output)):
        return None

    messages = build_messages(template, instruction, input_text, output, system_prompt)
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "messages": messages,
    }


def _map_columns(row: Dict, colmap: Dict[str, str]) -> Dict[str, str]:
    return {
        "instruction": row.get(colmap.get("instruction", "instruction"), "") or "",
        "input": row.get(colmap.get("input", "input"), "") or "",
        "output": row.get(colmap.get("output", "output"), "") or "",
    }


def load_source_rows(src: Dict, dry_run: bool) -> Iterable[Dict]:
    """Trả về iterable các row đã map cột. dry_run -> sinh mẫu giả, không tải mạng."""
    colmap = src.get("columns", {})
    if dry_run:
        n = 50
        for i in range(n):
            yield {
                "instruction": f"[{src['name']}] Câu hỏi mẫu số {i}: Thủ đô Việt Nam là gì?",
                "input": "" if i % 2 else "Bối cảnh: địa lý Việt Nam.",
                "output": f"Đây là câu trả lời mẫu số {i}: Thủ đô của Việt Nam là Hà Nội.",
            }
        return

    from datasets import load_dataset

    split = src.get("split", "train")
    if src.get("data_files"):
        # Trỏ thẳng file dữ liệu (json/jsonl/.gz/parquet) -> bỏ qua dataset script.
        # Cần khi datasets 3.x từ chối repo có script (vd Bactrian-X.py).
        df = src["data_files"]
        fmt = src.get("data_format", "json")
        ds = load_dataset(fmt, data_files=df, split=split)
    elif src.get("hf_config"):
        ds = load_dataset(src["hf_path"], src["hf_config"], split=split)
    else:
        ds = load_dataset(src["hf_path"], split=split)
    # Lọc theo cột (vd Aya: chỉ giữ language == "Vietnamese")
    flt = src.get("filter")
    if flt:
        for col, val in flt.items():
            ds = ds.filter(lambda r, c=col, v=val: r.get(c) == v)

    max_samples = src.get("max_samples")
    for i, row in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        yield _map_columns(row, colmap)


def build_dataset(mix_cfg: Dict, dry_run: bool) -> List[Dict]:
    cleaning = mix_cfg.get("cleaning", {})
    template = mix_cfg.get("prompt_template", "vi_instruct_v1")
    system_prompt = mix_cfg.get("system_prompt", "")
    dedup = cleaning.get("dedup", True)

    seen = set()
    out: List[Dict] = []
    stats = {}
    for src in mix_cfg["sources"]:
        kept = 0
        total = 0
        for row in load_source_rows(src, dry_run):
            total += 1
            rec = clean_record(row, cleaning, system_prompt, template)
            if rec is None:
                continue
            if dedup:
                h = _hash_example(rec["instruction"], rec["output"])
                if h in seen:
                    continue
                seen.add(h)
            rec["_source"] = src["name"]
            out.append(rec)
            kept += 1
        stats[src["name"]] = {"total": total, "kept": kept}
    print("[data_prep] Thống kê theo nguồn:")
    for name, s in stats.items():
        print(f"  - {name:18s} total={s['total']:>7} kept={s['kept']:>7}")
    return out


def split_and_save(records: List[Dict], out_dir: Path, val_fraction: float, devset_size: int, seed: int):
    import random

    rng = random.Random(seed)
    rng.shuffle(records)

    n = len(records)
    n_val = max(1, int(n * val_fraction)) if n > 1 else 0
    val = records[:n_val]
    train = records[n_val:]
    dev = train[: min(devset_size, len(train))]  # dev-set cố định để sanity định tính

    ensure_dir(out_dir)
    _write_jsonl(train, out_dir / "train.jsonl")
    _write_jsonl(val, out_dir / "val.jsonl")
    _write_jsonl(dev, out_dir / "devset.jsonl")

    # thống kê độ dài
    def avg_len(recs):
        if not recs:
            return 0
        return round(sum(len(r["output"]) for r in recs) / len(recs), 1)

    summary = {
        "n_total": n,
        "n_train": len(train),
        "n_val": len(val),
        "n_dev": len(dev),
        "avg_output_chars_train": avg_len(train),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[data_prep] {summary}")
    print(f"[data_prep] Đã lưu vào {out_dir}/ (train/val/devset.jsonl + summary.json)")


def _write_jsonl(records: List[Dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="configs/data/instruction_mix_v1.yaml")
    ap.add_argument("--out", default=None, help="thư mục output; mặc định data/<mix_name>")
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--devset-size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="không tải mạng, dùng mẫu giả để verify (CPU OK)")
    args = ap.parse_args()

    set_seed(args.seed)
    mix_path = Path(args.mix)
    if not mix_path.is_absolute():
        mix_path = REPO_ROOT / mix_path
    with open(mix_path, "r", encoding="utf-8") as f:
        mix_cfg = yaml.safe_load(f)

    out_dir = Path(args.out) if args.out else REPO_ROOT / "data" / mix_cfg.get("name", "mix")
    if not Path(out_dir).is_absolute():
        out_dir = REPO_ROOT / out_dir

    print(f"[data_prep] Mix: {mix_cfg.get('name')} | dry_run={args.dry_run}")
    records = build_dataset(mix_cfg, args.dry_run)
    if not records:
        print("[data_prep] CẢNH BÁO: 0 record sau khi lọc. Kiểm tra cấu hình nguồn/cột.")
    split_and_save(records, out_dir, args.val_fraction, args.devset_size, args.seed)


if __name__ == "__main__":
    main()
