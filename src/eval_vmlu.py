#!/usr/bin/env python3
"""Đánh giá VMLU (trắc nghiệm 4 lựa chọn tiếng Việt).

Cách chấm: với mỗi câu hỏi, tính log-prob của model cho từng đáp án "A"/"B"/"C"/"D"
(robust hơn parse text tự do), chọn đáp án có log-prob cao nhất, so với ground-truth.
Báo accuracy tổng + theo 4 nhóm chủ đề (STEM / Social Science / Humanities / Other).

Dùng:
  python src/eval_vmlu.py --config configs/exp/qwen3-4b-lora.yaml --adapter results/runs/<exp_id>
  python src/eval_vmlu.py --config ... --base-only        # eval model gốc (baseline)
  python src/eval_vmlu.py --config ... --subset 2000       # lặp nhanh

Dữ liệu VMLU: thử load_dataset("Zalo-AI/VMLU"/mirror); nếu offline, truyền --data-path
tới file JSON/JSONL có các trường: question, choices/A..D, answer, category/subject.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import load_config  # noqa: E402
from utils.io import ensure_dir, save_json  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
LETTERS = ["A", "B", "C", "D"]

# Ánh xạ subject -> nhóm. VMLU có 58 subject/4 nhóm. Bảng rút gọn; bổ sung khi chạy
# thật từ metadata VMLU. Subject không khớp -> "other".
GROUP_KEYWORDS = {
    "stem": ["math", "toán", "physics", "lý", "chemistry", "hóa", "biology", "sinh",
              "informatics", "tin học", "engineering", "kỹ thuật"],
    "social_science": ["economics", "kinh tế", "geography", "địa", "politics", "chính trị",
                         "law", "luật", "education", "giáo dục", "business", "quản"],
    "humanities": ["history", "sử", "literature", "văn", "philosophy", "triết",
                    "religion", "tôn giáo", "language", "ngôn ngữ"],
}


def map_group(subject: str) -> str:
    s = (subject or "").lower()
    for group, kws in GROUP_KEYWORDS.items():
        if any(k in s for k in kws):
            return group
    return "other"


def load_vmlu(data_path: Optional[str], subset: Optional[int]) -> List[Dict]:
    if data_path:
        rows = _read_local(Path(data_path))
    else:
        from datasets import load_dataset

        rows = None
        errors = []

        # (1) File JSONL CÓ NHÃN, đọc trực tiếp qua JSON loader (bypass script dataset lỗi).
        # ura-hcmut/vmlu_vi: split valid.jsonl có 'answer' (nhỏ, ~100 câu - đủ để dev/sanity).
        raw_sources = [
            ("ura-hcmut/vmlu_vi", "valid.jsonl"),
        ]
        for repo, fname in raw_sources:
            url = f"https://huggingface.co/datasets/{repo}/resolve/main/{fname}"
            try:
                ds = load_dataset("json", data_files=url, split="train")
                rows = [dict(r) for r in ds]
                print(f"[eval] VMLU (labeled): {repo}/{fname} n={len(rows)}")
                break
            except Exception as e:  # noqa: BLE001
                errors.append(f"{repo}/{fname}: {e}")

        # (2) Dataset đầy đủ (có thể gated - cần `huggingface-cli login`); tự chọn split có nhãn.
        if rows is None:
            for repo in ["anhdungitvn/vmlu_v1.5", "Zalo-AI/VMLU"]:
                try:
                    dd = load_dataset(repo)
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{repo}: {e}")
                    continue
                chosen = next(
                    (s for s in ["dev", "validation", "val", "train", "test"]
                     if s in dd and len(dd[s]) > 0
                     and (_normalize_row(dict(dd[s][0])) or {}).get("answer_idx") is not None),
                    list(dd.keys())[0],
                )
                rows = [dict(r) for r in dd[chosen]]
                print(f"[eval] VMLU: repo={repo} split={chosen} n={len(rows)}")
                break

        if rows is None:
            raise RuntimeError(
                "Không tải được VMLU có nhãn từ HF. Cách xử lý:\n"
                "  - Tải dev/valid set có nhãn rồi truyền --data-path file.jsonl, HOẶC\n"
                "  - `huggingface-cli login` nếu dùng dataset gated.\n"
                "Lỗi từng nguồn:\n  - " + "\n  - ".join(errors)
            )
    rows = [_normalize_row(r) for r in rows]
    rows = [r for r in rows if r is not None]
    if subset:
        rows = _stratified_subset(rows, subset)
    return rows


def _read_local(path: Path) -> List[Dict]:
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in open(path, encoding="utf-8")]
    data = json.load(open(path, encoding="utf-8"))
    return data if isinstance(data, list) else data.get("data", [])


def _normalize_row(r: Dict) -> Optional[Dict]:
    """Đưa về schema: {question, choices:[4], answer_idx, group}."""
    question = r.get("question") or r.get("Question") or ""
    # choices: hoặc list, hoặc các trường A/B/C/D
    if isinstance(r.get("choices"), list):
        choices = r["choices"]
    else:
        choices = [r.get(L) or r.get(L.lower()) for L in LETTERS]
    choices = [c for c in choices if c]
    if len(choices) < 2 or not question:
        return None

    ans = r.get("answer") or r.get("Answer") or r.get("label")
    if isinstance(ans, str) and ans.strip().upper() in LETTERS:
        answer_idx = LETTERS.index(ans.strip().upper())
    elif isinstance(ans, int):
        answer_idx = ans
    else:
        answer_idx = None  # test set không có nhãn -> chỉ sinh dự đoán

    subject = r.get("subject") or r.get("category") or r.get("class") or ""
    return {
        "question": question,
        "choices": choices,
        "answer_idx": answer_idx,
        "group": map_group(subject),
        "subject": subject,
    }


def _stratified_subset(rows: List[Dict], n: int) -> List[Dict]:
    """Lấy mẫu phân tầng theo group để subset đại diện."""
    import random

    rng = random.Random(42)
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group"]].append(r)
    per = max(1, n // max(1, len(by_group)))
    out = []
    for g, items in by_group.items():
        rng.shuffle(items)
        out.extend(items[:per])
    rng.shuffle(out)
    return out[:n]


def format_prompt(row: Dict) -> str:
    lines = [row["question"].strip(), ""]
    for i, c in enumerate(row["choices"][:4]):
        lines.append(f"{LETTERS[i]}. {c}")
    lines.append("")
    lines.append("Đáp án đúng (chỉ trả lời bằng một chữ cái A, B, C hoặc D):")
    return "\n".join(lines)


def evaluate(cfg: Dict, adapter: Optional[str], base_only: bool, data_path: Optional[str],
             subset: Optional[int]) -> Dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if adapter and not base_only:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        print(f"[eval] Đã gắn adapter: {adapter}")
    model.eval()

    rows = load_vmlu(data_path, subset)
    print(f"[eval] Số câu: {len(rows)}")

    # token id của " A"/" B"... (kèm leading space để khớp tokenizer)
    letter_ids = []
    for L in LETTERS:
        ids = tokenizer.encode(L, add_special_tokens=False)
        letter_ids.append(ids[0])

    correct = 0
    counted = 0
    by_group = defaultdict(lambda: [0, 0])  # group -> [correct, total]
    preds = []

    for row in rows:
        messages = [
            {"role": "system", "content": "Bạn là trợ lý làm bài trắc nghiệm tiếng Việt."},
            {"role": "user", "content": format_prompt(row)},
        ]
        enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        # return_dict=True -> BatchEncoding (dict). Một số version trả tensor -> bọc lại.
        if hasattr(enc, "items"):
            inputs = {k: v.to(model.device) for k, v in enc.items()}
        else:
            inputs = {"input_ids": enc.to(model.device)}
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]  # logits cho token kế tiếp
        choice_logits = torch.tensor([logits[i].item() for i in letter_ids])
        pred_idx = int(choice_logits.argmax())
        preds.append({"pred": LETTERS[pred_idx], "subject": row["subject"]})

        if row["answer_idx"] is not None:
            counted += 1
            ok = pred_idx == row["answer_idx"]
            correct += int(ok)
            g = row["group"]
            by_group[g][1] += 1
            by_group[g][0] += int(ok)

    result = {
        "model": model_name,
        "adapter": None if base_only else adapter,
        "n_questions": len(rows),
        "n_scored": counted,
    }
    if counted:
        result["vmlu_avg"] = round(100 * correct / counted, 2)
        for g, (c, t) in by_group.items():
            result[f"vmlu_{g}"] = round(100 * c / t, 2) if t else None
    else:
        result["note"] = "Test set không có nhãn — chỉ sinh dự đoán (preds.json)."
    print(f"[eval] {json.dumps({k: v for k, v in result.items() if k.startswith('vmlu') or k.startswith('n_')}, ensure_ascii=False)}")
    return result, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default=None, help="thư mục adapter LoRA (results/runs/<exp_id>)")
    ap.add_argument("--base-only", action="store_true", help="eval model gốc, bỏ adapter")
    ap.add_argument("--data-path", default=None, help="file VMLU local nếu offline")
    ap.add_argument("--subset", type=int, default=None, help="số câu (phân tầng); override config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    subset = args.subset if args.subset is not None else cfg.get("eval", {}).get("vmlu", {}).get("subset")
    result, preds = evaluate(cfg, args.adapter, args.base_only, args.data_path, subset)

    exp_id = cfg["_meta"]["exp_id"]
    out_dir = ensure_dir(REPO_ROOT / "results" / "runs" / exp_id)
    tag = "base" if args.base_only else "tuned"
    save_json(result, out_dir / f"vmlu_{tag}.json")
    save_json(preds, out_dir / f"vmlu_{tag}_preds.json")
    print(f"[eval] Đã lưu {out_dir}/vmlu_{tag}.json")


if __name__ == "__main__":
    main()
