#!/usr/bin/env python3
"""Finetune LLM tiếng Việt từ config YAML (Unsloth + TRL SFTTrainer).

Đọc config -> nạp model (Unsloth FastLanguageModel, hoặc fallback TRL+PEFT thuần)
-> gắn LoRA -> SFT trên `messages` (áp chat template của model) -> lưu adapter +
metrics vào results/runs/<exp_id>/.

Dùng (trên máy L40S):
  python src/train.py --config configs/exp/qwen3-4b-lora.yaml
  python src/train.py --config configs/exp/qwen3-4b-lora.yaml --max-steps 5   # smoke

Chế độ kiểm tra (CPU, không nạp model):
  python src/train.py --config configs/exp/qwen3-4b-lora.yaml --check
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import load_config  # noqa: E402
from utils.io import ensure_dir, save_json  # noqa: E402
from utils.seeding import set_seed  # noqa: E402
from utils.tracking import init_tracking, finish_tracking  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_data_dir(cfg: Dict) -> Path:
    """Suy ra thư mục data đã chuẩn bị từ mix_config."""
    import yaml

    mix_path = cfg["data"]["mix_config"]
    p = Path(mix_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        mix = yaml.safe_load(f)
    return REPO_ROOT / "data" / mix.get("name", "mix")


def check_config(cfg: Dict) -> None:
    """Validate config + sự tồn tại dữ liệu mà KHÔNG nạp model (chạy được trên CPU)."""
    exp_id = cfg["_meta"]["exp_id"]
    print(f"[check] exp_id = {exp_id}")
    print(f"[check] model  = {cfg['model']['name']}  (4bit={cfg['model']['load_in_4bit']})")
    print(f"[check] method = {cfg['method']}  lora.r={cfg['lora']['r']}")
    eff_bs = cfg["train"]["per_device_train_batch_size"] * cfg["train"]["gradient_accumulation_steps"]
    print(f"[check] effective batch = {eff_bs}")
    data_dir = resolve_data_dir(cfg)
    train_file = data_dir / "train.jsonl"
    if train_file.exists():
        n = sum(1 for _ in open(train_file, encoding="utf-8"))
        print(f"[check] data OK: {train_file} ({n} mẫu)")
    else:
        print(f"[check] CẢNH BÁO: chưa có {train_file}. Chạy src/data_prep.py trước.")
    print("[check] Config hợp lệ.")


def load_dataset_for_sft(cfg: Dict):
    from datasets import load_dataset

    data_dir = resolve_data_dir(cfg)
    files = {
        "train": str(data_dir / "train.jsonl"),
        "validation": str(data_dir / "val.jsonl"),
    }
    ds = load_dataset("json", data_files=files)
    max_train = cfg["data"].get("max_train_samples")
    if max_train:
        ds["train"] = ds["train"].select(range(min(max_train, len(ds["train"]))))
    return ds


def load_model_and_tokenizer(cfg: Dict):
    """Nạp model + tokenizer + gắn LoRA. Ưu tiên Unsloth, fallback TRL+PEFT."""
    m = cfg["model"]
    lora = cfg["lora"]
    use_unsloth = m.get("use_unsloth", True)

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=m["name"],
                max_seq_length=m["max_seq_length"],
                load_in_4bit=m["load_in_4bit"],
                dtype=None,  # tự chọn theo GPU (bf16 trên L40S)
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora["r"],
                lora_alpha=lora["lora_alpha"],
                lora_dropout=lora["lora_dropout"],
                target_modules=lora["target_modules"],
                bias=lora["bias"],
                use_gradient_checkpointing=lora["use_gradient_checkpointing"],
                random_state=cfg["experiment"]["seed"],
            )
            print("[train] Dùng Unsloth FastLanguageModel.")
            return model, tokenizer
        except Exception as e:  # noqa: BLE001
            print(f"[train] Unsloth không dùng được ({e}). Fallback TRL+PEFT thuần.")

    # Fallback: transformers + PEFT
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    quant_kwargs = {}
    if m["load_in_4bit"]:
        from transformers import BitsAndBytesConfig

        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(m["name"])
    model = AutoModelForCausalLM.from_pretrained(
        m["name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **quant_kwargs,
    )
    peft_cfg = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["lora_alpha"],
        lora_dropout=lora["lora_dropout"],
        target_modules=lora["target_modules"],
        bias=lora["bias"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def train(cfg: Dict, max_steps: int | None = None) -> Dict:
    import torch
    from trl import SFTConfig, SFTTrainer
    from transformers import EarlyStoppingCallback

    set_seed(cfg["experiment"]["seed"])
    exp_id = cfg["_meta"]["exp_id"]
    run_dir = ensure_dir(REPO_ROOT / cfg["train"]["output_dir"] / exp_id)
    save_json(cfg, run_dir / "config_snapshot.json")

    tracking = init_tracking(cfg)
    model, tokenizer = load_model_and_tokenizer(cfg)
    ds = load_dataset_for_sft(cfg)

    t = cfg["train"]
    sft_args = SFTConfig(
        output_dir=str(run_dir),
        num_train_epochs=t["num_train_epochs"] if not max_steps else 1,
        max_steps=max_steps if max_steps else -1,
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        optim=t["optim"],
        logging_steps=t["logging_steps"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        bf16=True,
        packing=t.get("packing", True),
        max_seq_length=cfg["model"]["max_seq_length"],
        load_best_model_at_end=True,
        metric_for_best_model=t.get("report_metric", "eval_loss"),
        report_to=tracking["report_to"],
        seed=cfg["experiment"]["seed"],
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    callbacks = []
    if t.get("early_stopping_patience"):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=t["early_stopping_patience"]))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        callbacks=callbacks,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0

    # lưu adapter
    trainer.save_model(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))

    vram_peak = (
        round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0
    )
    eval_metrics = trainer.evaluate()
    metrics = {
        "exp_id": exp_id,
        "model": cfg["model"]["name"],
        "method": cfg["method"],
        "train_runtime_sec": round(elapsed, 1),
        "train_time": _fmt_time(elapsed),
        "vram_peak_gb": vram_peak,
        "train_loss": round(train_result.training_loss, 4),
        "eval_loss": round(eval_metrics.get("eval_loss", float("nan")), 4),
        "adapter_dir": str(run_dir),
    }
    save_json(metrics, run_dir / "train_metrics.json")
    print(f"[train] Xong. {json.dumps(metrics, ensure_ascii=False)}")
    finish_tracking(tracking)
    return metrics


def _fmt_time(sec: float) -> str:
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--check", action="store_true", help="chỉ validate config + data (CPU, không nạp model)")
    ap.add_argument("--max-steps", type=int, default=None, help="giới hạn step (dùng cho smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.check:
        check_config(cfg)
        return
    train(cfg, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
