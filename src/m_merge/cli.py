"""Merge CLI 入口 (FR-06)

提供命令行接口，等价于 llamafactory-cli export 的自研替代。

Usage:
    # 基本用法
    python -m m_merge.cli \
        --base Qwen/Qwen2.5-1.5B-Instruct \
        --adapter saves/qwen2_5_1_5b/insurance_dpo_v1.2/lora \
        --output merged_models/qwen2_5_1_5b_insurance_dpo_v1.2

    # 自定义导出参数
    python -m m_merge.cli \
        --base /path/to/base_model \
        --adapter /path/to/lora_adapter \
        --output /path/to/export \
        --size 5 \
        --device cpu
"""

from __future__ import annotations

import argparse
import logging
import sys

from .exporter import merge_and_export

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s|%(asctime)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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

    args = parser.parse_args()

    try:
        export_dir = merge_and_export(
            base_model_path=args.base,
            adapter_path=args.adapter,
            export_dir=args.output,
            export_size=args.size,
            export_device=args.device,
        )
        print(f"\nMerge completed successfully.")
        print(f"Exported model: {export_dir}")
        sys.exit(0)
    except Exception as e:
        print(f"\nMerge failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
