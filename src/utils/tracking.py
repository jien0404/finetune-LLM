"""Tracking thực nghiệm: ưu tiên W&B nếu kết nối được, fallback TensorBoard.

Quy trình người dùng yêu cầu: KIỂM TRA kết nối W&B trước. `check_wandb_connectivity`
được dùng bởi cả `scripts/check_wandb.py` và `init_tracking` để quyết định backend.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple


def check_wandb_connectivity(timeout: float = 8.0) -> Tuple[bool, str]:
    """Trả về (ok, message). ok=True nghĩa là gọi được W&B API (có thể log online).

    Kiểm tra: (1) package wandb tồn tại, (2) có API key, (3) ping được host.
    Không raise — luôn trả tuple để caller tự quyết fallback.
    """
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False, "wandb chưa được cài (pip install wandb)."

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        # vẫn thử dùng key đã lưu trong ~/.netrc qua wandb.login
        try:
            import wandb

            if not wandb.api.api_key:
                return False, "Không tìm thấy WANDB_API_KEY (env hoặc `wandb login`)."
        except Exception as e:  # noqa: BLE001
            return False, f"Không xác định được API key: {e}"

    # Phép thử thật: gọi API viewer (vừa kiểm tra mạng vừa xác thực key).
    # Không ping HEAD tới host gốc vì URL đó có thể trả 404 dù host vẫn sống.
    host = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")
    try:
        import wandb

        api = wandb.Api(timeout=int(timeout))
        viewer = api.viewer  # gọi để xác thực key + kiểm tra kết nối
        username = getattr(viewer, "username", None) or getattr(viewer, "entity", "?")
    except Exception as e:  # noqa: BLE001
        return False, f"Không kết nối/xác thực được W&B ({host}): {e}"

    return True, f"Kết nối W&B OK (user={username}) — sẽ log online."


def init_tracking(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Khởi tạo backend tracking dựa trên kết nối thực tế.

    Trả về dict: {"backend": "wandb"|"tensorboard"|"none",
                  "report_to": [...], "run": <wandb run hoặc None>,
                  "tb_logdir": <path hoặc None>}.
    `report_to` truyền thẳng vào TrainingArguments của TRL/transformers.
    """
    track = cfg.get("tracking", {})
    prefer = track.get("backend", "auto")  # auto|wandb|tensorboard|none
    exp_id = cfg.get("_meta", {}).get("exp_id", "run")

    if prefer == "none":
        return {"backend": "none", "report_to": [], "run": None, "tb_logdir": None}

    use_wandb = False
    if prefer in ("auto", "wandb"):
        ok, msg = check_wandb_connectivity()
        print(f"[tracking] W&B check: {'PASS' if ok else 'FALLBACK'} — {msg}")
        use_wandb = ok or (prefer == "wandb" and track.get("force", False))

    if use_wandb:
        import wandb

        run = wandb.init(
            project=track.get("project", "finetune-llm-vi"),
            name=exp_id,
            config=cfg,
            tags=track.get("tags", []),
            reinit=True,
        )
        return {"backend": "wandb", "report_to": ["wandb"], "run": run, "tb_logdir": None}

    # fallback TensorBoard
    from .io import ensure_dir

    tb_logdir = track.get("tb_logdir", f"runs/tb/{exp_id}")
    ensure_dir(tb_logdir)
    print(f"[tracking] Dùng TensorBoard tại {tb_logdir}")
    return {
        "backend": "tensorboard",
        "report_to": ["tensorboard"],
        "run": None,
        "tb_logdir": tb_logdir,
    }


def finish_tracking(state: Optional[Dict[str, Any]]) -> None:
    if state and state.get("backend") == "wandb" and state.get("run") is not None:
        state["run"].finish()
