"""Merge CLI 入口 (FR-06)

提供命令行接口，等价于 llamafactory-cli export 的自研替代。

Usage:
    # 基本用法
    python -m m_merge.cli \\
        --base Qwen/Qwen2.5-1.5B-Instruct \\
        --adapter saves/qwen2_5_1_5b/insurance_dpo_v1.2/lora \\
        --output merged_models/qwen2_5_1_5b_insurance_dpo_v1.2

    # 自定义导出参数（GPU 加速 + bfloat16）
    python -m m_merge.cli \\
        --base /path/to/base_model \\
        --adapter /path/to/lora_adapter \\
        --output /path/to/export \\
        --size 5 \\
        --device cuda \\
        --dtype bfloat16
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

from .exporter import merge_and_export
from utils.logger import CustomLogger

log = CustomLogger.get_logger(__name__)


_DTYPE_MAP = {
    "float16": "torch.float16",
    "bfloat16": "torch.bfloat16",
    "float32": "torch.float32",
    "auto": None,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model and export as HF safetensors",
    )
    parser.add_argument(
        "--base",
        required=True,
        help="Base model path (HF model id or local directory)",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="PEFT adapter path (contains adapter_config.json and adapter_model.safetensors)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Export directory for the merged model",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=5,
        help="Max shard size in GB (default: 5)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device to use during merge (default: cpu)",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
        help="Output model dtype (default: auto; cuda mode defaults to bfloat16)",
    )
    parser.add_argument(
        "--offload-folder",
        default=None,
        help="Disk offload directory for large models (prevents OOM, cuda only)",
    )
    parser.add_argument(
        "--keep-adapter",
        action="store_true",
        help="Copy the original adapter files into the merged model directory",
    )

    args = parser.parse_args()

    # 解析 dtype
    import torch
    _dtype = None
    if args.dtype != "auto":
        _dtype = getattr(torch, args.dtype, None)
        if _dtype is None:
            log.error("不支持的 dtype: {}，可选值: float16, bfloat16, float32, auto", args.dtype)
            sys.exit(1)

    try:
        export_dir = merge_and_export(
            base_model_path=args.base,
            adapter_path=args.adapter,
            export_dir=args.output,
            export_size=args.size,
            export_device=args.device,
            torch_dtype=_dtype,
            offload_folder=args.offload_folder,
        )

        # 保留原始 adapter
        if args.keep_adapter:
            adapter_dest = os.path.join(export_dir, "lora_adapter_backup")
            log.info("正在备份原始 adapter 到 {} ...", adapter_dest)
            shutil.copytree(args.adapter, adapter_dest, dirs_exist_ok=True)

        log.info("合并成功，导出目录: {}", export_dir)
        print(f"\nMerge completed successfully.")
        print(f"Exported model: {export_dir}")
        if args.keep_adapter:
            print(f"Adapter backup: {adapter_dest}")
        sys.exit(0)
    except Exception as e:
        log.exception("合并失败")
        print(f"\nMerge failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
