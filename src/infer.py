#!/usr/bin/env python3
"""Sinh câu trả lời để review tay — so base vs tuned trên devset.

Dùng:
  python src/infer.py --config configs/exp/qwen3-4b-lora.yaml \
        --adapter results/runs/<exp_id> --n 20 --compare-base

Lưu results/runs/<exp_id>/samples.jsonl: mỗi dòng {prompt, base_output?, tuned_output}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import load_config  # noqa: E402
from utils.io import ensure_dir  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_devset(cfg: Dict, n: int) -> List[Dict]:
    import yaml

    mix_path = cfg["data"]["mix_config"]
    p = Path(mix_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    mix = yaml.safe_load(open(p, encoding="utf-8"))
    dev_file = REPO_ROOT / "data" / mix.get("name", "mix") / "devset.jsonl"
    rows = [json.loads(l) for l in open(dev_file, encoding="utf-8")]
    return rows[:n]


def generate(model, tokenizer, messages, max_new_tokens=512) -> str:
    import torch

    user_messages = [m for m in messages if m["role"] != "assistant"]
    inputs = tokenizer.apply_chat_template(
        user_messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--compare-base", action="store_true", help="sinh thêm output model gốc để so")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    cfg = load_config(args.config)
    model_name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    base.eval()

    devset = load_devset(cfg, args.n)
    samples = []
    # tuned = base + adapter
    tuned = PeftModel.from_pretrained(base, args.adapter)
    tuned.eval()

    for row in devset:
        prompt = row["messages"][1]["content"] if len(row["messages"]) > 1 else row.get("instruction", "")
        rec = {"prompt": prompt, "reference": row.get("output", "")}
        rec["tuned_output"] = generate(tuned, tokenizer, row["messages"], args.max_new_tokens)
        samples.append(rec)

    if args.compare_base:
        # tách adapter để chạy base thuần
        base_plain = tuned.get_base_model()
        with tuned.disable_adapter():
            for rec, row in zip(samples, devset):
                rec["base_output"] = generate(base_plain, tokenizer, row["messages"], args.max_new_tokens)

    exp_id = cfg["_meta"]["exp_id"]
    out = ensure_dir(REPO_ROOT / "results" / "runs" / exp_id) / "samples.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[infer] Đã lưu {len(samples)} mẫu vào {out}")


if __name__ == "__main__":
    main()
