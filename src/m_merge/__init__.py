# m_merge: LoRA 与基座合并导出模块 (FR-06)
# 将 PEFT adapter 合并为完整 HF safetensors 模型，供 vLLM / xinference 直接加载

from .exporter import merge_and_export, _validate_adapter_is_lora
from .cli import main

__all__ = [
    "merge_and_export",
    "_validate_adapter_is_lora",
    "main",
]
