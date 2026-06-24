"""Prompt template chuẩn cho instruction tiếng Việt.

Mỗi template biến (instruction, input, output) -> danh sách `messages`
(role: system/user/assistant). Trainer áp chat template của từng model
(Qwen3/Gemma) lên `messages` này, nên ta KHÔNG hard-code token đặc biệt của model
ở đây — chỉ định hình nội dung user/assistant.
"""
from __future__ import annotations

from typing import Dict, List, Optional

DEFAULT_SYSTEM = "Bạn là trợ lý AI hữu ích, trả lời bằng tiếng Việt rõ ràng và chính xác."


def _compose_user(instruction: str, input_text: Optional[str]) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction


def vi_instruct_v1(
    instruction: str,
    input_text: Optional[str] = None,
    output: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Template instruction tiếng Việt mặc định -> messages."""
    messages = [{"role": "system", "content": system_prompt or DEFAULT_SYSTEM}]
    messages.append({"role": "user", "content": _compose_user(instruction, input_text)})
    if output is not None:
        messages.append({"role": "assistant", "content": output.strip()})
    return messages


# Đăng ký template theo tên (dùng trong configs/data/*.yaml: prompt_template)
TEMPLATES = {
    "vi_instruct_v1": vi_instruct_v1,
}


def build_messages(
    template_name: str,
    instruction: str,
    input_text: Optional[str] = None,
    output: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    if template_name not in TEMPLATES:
        raise KeyError(
            f"Template '{template_name}' không tồn tại. Có: {list(TEMPLATES)}"
        )
    return TEMPLATES[template_name](instruction, input_text, output, system_prompt)
