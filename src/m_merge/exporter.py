"""
LoRA 与基座合并导出 (FR-06)

合并公式（PEFT 库 merge_and_unload 实现）：

    W_merged = W_base + (α/r) · B · A

其中 W_base 为基座模型原权重，A ∈ R^{r×d_in}, B ∈ R^{d_out×r} 为 LoRA 矩阵，
α 为缩放系数（lora_alpha），r 为 rank（lora_rank）。

导出后产物为完整 HF safetensors 模型文件夹，可直接被 vLLM / xinference / HF transformers 加载。
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.logger import CustomLogger

log = CustomLogger.get_logger(__name__)


def merge_and_export(
    base_model_path: str,
    adapter_path: str,
    export_dir: str,
    export_size: int = 5,
    export_device: str = "cpu",
    torch_dtype: Optional[torch.dtype] = None,
    offload_folder: Optional[str] = None,
) -> str:
    """合并 LoRA adapter 到基座模型并导出为 HF safetensors。

    等效于 LLaMA-Factory 的 ``llamafactory-cli export``，但作为独立模块可直接嵌入脚本。

    Args:
        base_model_path: 基座模型路径（HF model id 或本地文件夹）
        adapter_path: PEFT adapter 路径（含 adapter_config.json 和 adapter_model.safetensors）
        export_dir: 导出目标目录（若不存在则自动创建）
        export_size: 单个 safetensors 分片最大 GB 数（默认 5GB）
        export_device: 合并时使用的设备（``"cpu"`` 或 ``"cuda"``），默认 ``"cpu"``
        torch_dtype: 合并后模型的 dtype（默认自动推断；cuda 模式下默认 ``torch.bfloat16``）
        offload_folder: 大模型磁盘卸载目录（防止 OOM，仅 ``export_device="cuda"`` 时生效）

    Returns:
        导出目录的绝对路径

    Raises:
        FileNotFoundError: 若基座模型或 adapter 路径不存在
        ValueError: 若 adapter 的 peft_type 不是 ``"LORA"``
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

    # ── 路径校验 ──
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"base model not found: {base_model_path}")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"adapter not found: {adapter_path}")

    # ── Adapter 类型校验 ──
    _validate_adapter_is_lora(adapter_path)

    os.makedirs(export_dir, exist_ok=True)

    # ── 设备策略 ──
    use_cuda = export_device == "cuda" and torch.cuda.is_available()
    if export_device == "cuda" and not torch.cuda.is_available():
        log.warning("export_device='cuda' 但 CUDA 不可用，回退到 CPU")
        use_cuda = False

    # ── 加载基座模型 ──
    _dtype = torch_dtype or (torch.bfloat16 if use_cuda else "auto")
    load_kwargs: dict = {
        "torch_dtype": _dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if use_cuda:
        load_kwargs["device_map"] = "auto"
        if offload_folder:
            load_kwargs["offload_folder"] = offload_folder
            os.makedirs(offload_folder, exist_ok=True)

    log.info("正在加载基座模型 {} (device={}, dtype={}) ...",
             base_model_path, "cuda" if use_cuda else "cpu", _dtype)
    base = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)

    # ── 加载 adapter ──
    log.info("正在加载 adapter {} ...", adapter_path)
    peft_model = PeftModel.from_pretrained(base, adapter_path)

    # ── 合并 ──
    log.info("正在合并 adapter → base ...")
    merged = peft_model.merge_and_unload()

    # ── 释放中间对象 ──
    del peft_model
    del base
    if use_cuda:
        torch.cuda.empty_cache()

    # ── 保存 ──
    log.info("正在保存合并模型到 {} (max_shard={}GB) ...", export_dir, export_size)
    merged.save_pretrained(
        export_dir,
        max_shard_size=f"{export_size}GB",
        safe_serialization=True,
    )

    log.info("正在保存 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(export_dir)

    # ── 清理 ──
    del merged
    if use_cuda:
        torch.cuda.empty_cache()

    log.info("合并完成: {}", export_dir)
    return os.path.abspath(export_dir)


def _validate_adapter_is_lora(adapter_path: str) -> None:
    """校验 adapter 类型是否为 LoRA。

    读取 ``adapter_config.json`` 中的 ``peft_type`` 字段，
    非 ``LORA`` 则抛出 ``ValueError``。
    """
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_path):
        log.warning("adapter_config.json 不存在，跳过类型校验: {}", adapter_path)
        return

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    peft_type = cfg.get("peft_type", "UNKNOWN")
    if peft_type != "LORA":
        raise ValueError(
            f"期望 LORA adapter，但 adapter_config.json 中 peft_type={peft_type!r}。"
            f"当前仅支持 LoRA 合并，不支持 {peft_type}。"
        )
    log.debug("adapter 类型校验通过: peft_type=LORA")
