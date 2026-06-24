"""Đọc & hợp nhất config YAML.

Nguyên tắc: `configs/base.yaml` chứa default, mỗi experiment trong `configs/exp/*.yaml`
chỉ override những gì cần. Đổi model/hyperparam KHÔNG cần sửa code Python.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Hợp nhất đệ quy `override` lên `base` (override thắng). Trả về dict mới."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(exp_path: str | Path) -> Dict[str, Any]:
    """Load base.yaml rồi override bằng file experiment.

    `exp_path` có thể là đường dẫn tuyệt đối, hoặc tương đối so với repo root
    (vd "configs/exp/qwen3-4b-lora.yaml" hoặc "exp/qwen3-4b-lora.yaml").
    """
    exp_path = Path(exp_path)
    if not exp_path.is_absolute():
        # cho phép viết gọn "exp/xxx.yaml"
        candidates = [REPO_ROOT / exp_path, REPO_ROOT / "configs" / exp_path]
        exp_path = next((c for c in candidates if c.exists()), REPO_ROOT / exp_path)

    base = _read_yaml(BASE_CONFIG) if BASE_CONFIG.exists() else {}
    override = _read_yaml(exp_path)
    cfg = deep_update(base, override)

    # gắn metadata để truy vết
    cfg.setdefault("_meta", {})
    cfg["_meta"]["exp_config_path"] = str(exp_path)
    cfg["_meta"]["exp_id"] = exp_id_from_config(cfg)
    return cfg


def exp_id_from_config(cfg: Dict[str, Any]) -> str:
    """Sinh exp_id ổn định, gắn YAML ↔ thư mục run ↔ dòng leaderboard ↔ W&B run.

    Ưu tiên `experiment.id` nếu config khai báo; nếu không, tự sinh từ
    model + method + hyperparam chính.
    """
    exp = cfg.get("experiment", {})
    if exp.get("id"):
        return str(exp["id"])

    model_name = cfg.get("model", {}).get("name", "model")
    short = model_name.split("/")[-1].lower()
    method = cfg.get("method", "lora")
    lora = cfg.get("lora", {})
    train = cfg.get("train", {})
    rank = lora.get("r", "NA")
    lr = train.get("learning_rate", "NA")
    epochs = train.get("num_train_epochs", "NA")
    return f"{short}_{method}-r{rank}_lr{lr}_ep{epochs}"
