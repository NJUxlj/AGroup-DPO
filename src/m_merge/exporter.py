"""
LoRA 与基座合并导出 (FR-06)

合并公式（PEFT 库 merge_and_unload 实现）：

    W_merged = W_base + (α/r) · B · A

其中 W_base 为基座模型原权重，A ∈ R^{r×d_in}, B ∈ R^{d_out×r} 为 LoRA 矩阵，
α 为缩放系数（lora_alpha），r 为 rank（lora_rank）。

导出后产物为完整 HF safetensors 模型文件夹，可直接被 vLLM / xinference / HF transformers 加载。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def merge_and_export(
    base_model_path: str,
    adapter_path: str,
    export_dir: str,
    export_size: int = 5,
    export_device: str = "cpu",
    torch_dtype: Optional[torch.dtype] = None,
) -> str:
    """合并 LoRA adapter 到基座模型并导出为 HF safetensors。

    等效于 LLaMA-Factory 的 `llamafactory-cli export`，但作为独立模块可直接嵌入脚本。

    Args:
        base_model_path: 基座模型路径（HF model id 或本地文件夹）
        adapter_path: PEFT adapter 路径（含 adapter_config.json 和 adapter_model.safetensors）
        export_dir: 导出目标目录（若不存在则自动创建）
        export_size: 单个 safetensors 分片最大 GB 数（默认 5GB）
        export_device: 合并时使用的设备（"cpu" 或 "cuda"）
        torch_dtype: 合并后模型的 dtype（默认自动推断）

    Returns:
        导出目录的绝对路径

    Raises:
        FileNotFoundError: 若基座模型或 adapter 路径不存在
        ImportError: 若 peft 未安装

    Usage:
        from m_merge.exporter import merge_and_export

        merge_and_export(
            base_model_path="Qwen/Qwen2.5-1.5B-Instruct",
            adapter_path="saves/.../lora",
            export_dir="merged_models/qwen_insurance_dpo",
        )
    """
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError(
            "peft is not installed. Install with: pip install peft"
        )

    # 验证路径
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"base model not found: {base_model_path}")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"adapter not found: {adapter_path}")

    os.makedirs(export_dir, exist_ok=True)

    logger.info("Loading base model from %s ...", base_model_path)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype or "auto",
        trust_remote_code=True,
    )

    logger.info("Loading adapter from %s ...", adapter_path)
    peft_model = PeftModel.from_pretrained(base, adapter_path)

    logger.info("Merging adapter into base model ...")
    merged = peft_model.merge_and_unload()

    logger.info("Saving merged model to %s (max shard=%dGB, device=%s) ...",
                export_dir, export_size, export_device)
    merged.save_pretrained(
        export_dir,
        max_shard_size=f"{export_size}GB",
        safe_serialization=True,
    )

    logger.info("Saving tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(export_dir)

    logger.info("Merge complete: %s", export_dir)
    return os.path.abspath(export_dir)
