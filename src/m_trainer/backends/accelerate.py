"""HuggingFace accelerate 后端适配器 (FR-07)

用于单机多卡调试与 CI 烟雾测试，封装 accelerator.prepare() 统一包装模型/优化器/dataloader。
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.logger import CustomLogger

from .base import DistributedBackend, TrainerConfig

log = CustomLogger.get_logger(__name__)


class AccelerateBackend(DistributedBackend):
    """HuggingFace accelerate 后端。

    适合单卡/小规模调试场景，通过 accelerator.prepare 统一包装。

    Usage:
        backend = AccelerateBackend()
        model, optimizer = backend.init(model, optimizer, config)
        dataloader = backend.prepare_dataloader(dataloader)
        for batch in dataloader:
            loss = model(batch)
            backend.backward(loss)
            backend.step()
    """

    def __init__(self) -> None:
        self._accelerator: Any = None
        self._model: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None

    # ---- DistributedBackend 接口实现 ----

    def init(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        config: TrainerConfig,
    ) -> tuple[nn.Module, Any]:
        """accelerate 初始化。

        optimizer 若为 None，accelerator 不会自动创建；
        调用方需通过 OptimizerFactory 创建后再传入。
        """
        try:
            from accelerate import Accelerator
        except ImportError:
            raise ImportError(
                "accelerate is not installed. Install with: pip install accelerate"
            )

        self._accelerator = Accelerator(
            mixed_precision="bf16" if config.bf16 else "no",
        )

        if optimizer is not None:
            self._model, self._optimizer = self._accelerator.prepare(model, optimizer)
        else:
            self._model = self._accelerator.prepare(model)
            self._optimizer = None

        log.info(
            "accelerate initialized: device=%s, num_processes=%s",
            self._accelerator.device,
            self._accelerator.num_processes,
        )
        return self._model, self._optimizer

    def wrap_model(self, model: nn.Module) -> nn.Module:
        if self._accelerator is not None:
            return self._accelerator.prepare(model)
        return model

    def wrap_optimizer(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.Optimizer:
        if self._accelerator is not None:
            return self._accelerator.prepare(optimizer)
        return optimizer

    def prepare_dataloader(self, dataloader: DataLoader) -> DataLoader:
        if self._accelerator is not None:
            return self._accelerator.prepare(dataloader)
        return dataloader

    def barrier(self) -> None:
        if self._accelerator is not None:
            self._accelerator.wait_for_everyone()

    def state_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self._accelerator is not None:
            result["accelerator"] = self._accelerator.state_dict()
        return result

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._accelerator is not None and "accelerator" in state:
            self._accelerator.load_state_dict(state["accelerator"])

    def backward(self, loss: torch.Tensor) -> None:
        if self._accelerator is not None:
            self._accelerator.backward(loss)
        else:
            loss.backward()

    def step(self) -> None:
        if self._optimizer is not None:
            self._optimizer.step()

    def zero_grad(self) -> None:
        if self._optimizer is not None:
            self._optimizer.zero_grad()
