"""Megatron-LM 后端适配器 (FR-07)

接口预留。当前 1.5B 模型规模暂不启用张量并行（TP）/ 流水线并行（PP）。
当模型规模扩展到 7B+ 时，启用 tensor_model_parallel_size 和 pipeline_model_parallel_size。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import DistributedBackend, TrainerConfig

logger = logging.getLogger(__name__)


class MegatronBackend(DistributedBackend):
    """Megatron-LM 适配器 —— 接口预留。

    当前 1.5B 模型暂不启用 TP/PP，调用 init() 将触发 NotImplementedError。
    预计在模型规模扩展到 7B+ 后实施。

    待实施功能：
      - tensor_model_parallel_size: 张量并行度
      - pipeline_model_parallel_size: 流水线并行度
      - Megatron-LM 模型转换（HF → Megatron checkpoint 格式）
      - 分布式数据加载（bin/index 文件格式）
    """

    def init(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        config: TrainerConfig,
    ) -> tuple[nn.Module, Any]:
        raise NotImplementedError(
            "Megatron backend is reserved for future scaling to 7B+ models. "
            "For 1.5B training, use DeepSpeed (default) or FSDP instead. "
            "Set trainer.distributed_backend to 'deepspeed' or 'fsdp' in your yaml config."
        )

    def wrap_model(self, model: nn.Module) -> nn.Module:
        raise NotImplementedError("Megatron backend not implemented for 1.5B models.")

    def wrap_optimizer(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.Optimizer:
        raise NotImplementedError("Megatron backend not implemented for 1.5B models.")

    def prepare_dataloader(self, dataloader: DataLoader) -> DataLoader:
        raise NotImplementedError("Megatron backend not implemented for 1.5B models.")

    def barrier(self) -> None:
        raise NotImplementedError("Megatron backend not implemented for 1.5B models.")

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError("Megatron backend not implemented for 1.5B models.")

    def load_state_dict(self, state: dict[str, Any]) -> None:
        raise NotImplementedError("Megatron backend not implemented for 1.5B models.")
