"""
训练 CLI 入口 (M04)

支持两种训练模式:
  - llamafactory: 委托给 LLaMA-Factory CLI（兼容现有 YAML 配置）
  - 自定义后端: 使用 CustomTrainer + m_trainer 分布式后端

Usage:
    # LLaMA-Factory 模式
    python -m m_trainer.cli --config configs/train_dpo_qwen2_5_1_5b_insurance.yaml --backend llamafactory

    # 自定义后端 (DeepSpeed)
    python -m m_trainer.cli --config configs/my_train.yaml --backend deepspeed

    # 自定义后端 (FSDP)
    python -m m_trainer.cli --config configs/my_train.yaml --backend fsdp
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from utils.logger import CustomLogger

from .custom_trainer import CustomTrainer

log = CustomLogger.get_logger(__name__)

# 可用的自定义后端
CUSTOM_BACKENDS = ["deepspeed", "fsdp", "accelerate", "megatron"]

# 后端选择
ALL_BACKENDS = ["llamafactory"] + CUSTOM_BACKENDS


def _train_llamafactory(config_path: str) -> int:
    """委托给 LLaMA-Factory CLI 执行训练。"""
    if not Path(config_path).exists():
        log.error("Config file not found: %s", config_path)
        return 1

    # 使用当前 Python 解释器对应的 llamafactory-cli，避免 PATH 问题
    import shutil
    py_bin = str(Path(sys.executable).parent)
    lf_cli = shutil.which("llamafactory-cli", path=py_bin)
    if lf_cli is None:
        lf_cli = "llamafactory-cli"  # fallback to PATH

    # 将 Python bin 目录加入 PATH，确保 torchrun 等工具可被找到
    env = os.environ.copy()
    env["PATH"] = py_bin + os.pathsep + env.get("PATH", "")

    log.info("Delegating to %s train %s", lf_cli, config_path)
    cmd = [lf_cli, "train", config_path]
    result = subprocess.run(cmd, env=env)
    return result.returncode


def _train_custom(config_path: str, backend: str) -> int:
    """使用 CustomTrainer + 自定义分布式后端执行训练。"""
    if not Path(config_path).exists():
        log.error("Config file not found: %s", config_path)
        return 1

    log.info("Using custom backend: %s", backend)
    trainer = CustomTrainer.from_yaml(config_path)
    # 覆盖配置中的后端选择（CLI 参数优先）
    trainer.cfg.distributed_backend = backend
    trainer.train()
    return 0


def main(args: list[str] | None = None) -> int:
    """训练 CLI 主入口。

    Args:
        args: 命令行参数列表（用于统一 CLI 透传），None 则从 sys.argv 解析

    Returns:
        0 表示成功，非 0 表示失败
    """
    parser = argparse.ArgumentParser(
        description="M-TRAINER: 训练入口（LLaMA-Factory / 自定义后端）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例:
  # LLaMA-Factory
  python -m m_trainer.cli --config configs/train_dpo_qwen2_5_1_5b_insurance.yaml --backend llamafactory

  # DeepSpeed
  python -m m_trainer.cli --config configs/my_train.yaml --backend deepspeed

  # FSDP
  python -m m_trainer.cli --config configs/my_train.yaml --backend fsdp

可用自定义后端: {', '.join(CUSTOM_BACKENDS)}
        """,
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="训练配置文件路径（YAML 格式，兼容 LLaMA-Factory 格式）",
    )
    parser.add_argument(
        "--backend", "-b",
        choices=ALL_BACKENDS,
        default="llamafactory",
        help="训练后端: llamafactory（默认）| deepspeed | fsdp | accelerate | megatron",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出",
    )

    parsed = parser.parse_args(args)

    CustomLogger.configure(
        level="DEBUG" if parsed.verbose else "INFO",
    )

    log.info("Training config: %s", parsed.config)
    log.info("Backend: %s", parsed.backend)

    if parsed.backend == "llamafactory":
        return _train_llamafactory(parsed.config)
    else:
        return _train_custom(parsed.config, parsed.backend)


if __name__ == "__main__":
    sys.exit(main())
