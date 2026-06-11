"""PyTorch FSDP 后端适配器 (FR-07)

基于 torch.distributed.fsdp.fully_shard (PyTorch 2.4+) 实现，
作为 DeepSpeed 在 PyTorch 原生派系下的等价替代。
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.utils.data import DataLoader, DistributedSampler

from .base import DistributedBackend, TrainerConfig

logger = logging.getLogger(__name__)


def _get_transformer_block_cls(model: nn.Module) -> set[type]:
    """从模型中自动发现 Transformer 层类名，用于 FSDP wrap policy。

    尝试匹配常见的 Transformer block 命名：
    Qwen2DecoderLayer, LlamaDecoderLayer, GPT2Block, TransformerBlock 等。
    """
    candidate: set[type] = set()
    for module in model.modules():
        cls_name = module.__class__.__name__
        if any(
            keyword in cls_name
            for keyword in ("DecoderLayer", "EncoderLayer", "TransformerBlock", "GPT2Block")
        ):
            candidate.add(module.__class__)
    return candidate


class FSDPBackend(DistributedBackend):
    """PyTorch FSDP 后端。

    使用 FSDP (Fully Sharded Data Parallel) 在 2 卡上分片模型参数/梯度/优化器状态。

    Usage:
        backend = FSDPBackend()
        model, optimizer = backend.init(model, optimizer, config)
        for batch in dataloader:
            loss = model(batch)
            backend.backward(loss)
            backend.step()
    """

    def __init__(self) -> None:
        self._model: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None

    # ---- DistributedBackend 接口实现 ----

    def init(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        config: TrainerConfig,
    ) -> tuple[nn.Module, Any]:
        """FSDP 初始化。

        optimizer 若为 None，backend 不会自动创建 —— 调用方需先通过
        OptimizerFactory 创建优化器后再传入。
        """
        wrapped_model = self.wrap_model(model)
        if optimizer is not None:
            wrapped_optimizer = self.wrap_optimizer(wrapped_model, optimizer)
        else:
            wrapped_optimizer = optimizer

        self._model = wrapped_model
        self._optimizer = wrapped_optimizer

        logger.info("FSDP initialized: world_size=%s", config.world_size)
        return wrapped_model, wrapped_optimizer

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """使用 FSDP 包装模型。

        auto_wrap_policy 优先尝试 transformer_auto_wrap_policy，
        若模型中找不到已知的 Transformer block 类，则退化为 size_based_auto_wrap_policy。
        """
        if getattr(self, "_model", None) is not None and self._model is not None:
            return self._model

        transformer_cls = _get_transformer_block_cls(model)

        if transformer_cls:
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=transformer_cls,
            )
            logger.info("FSDP auto_wrap_policy: transformer_auto_wrap (%s)", transformer_cls)
        else:
            auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy, min_num_params=1e6
            )
            logger.info("FSDP auto_wrap_policy: size_based (min_params=1M)")

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        wrapped = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            cpu_offload=CPUOffload(offload_params=False),
            device_id=torch.cuda.current_device(),
        )
        return wrapped

    def wrap_optimizer(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.Optimizer:
        """FSDP 下优化器无需额外包装，直接返回。"""
        return optimizer

    def prepare_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """注入 DistributedSampler（若尚未注入）。"""
        if not isinstance(dataloader.sampler, DistributedSampler):
            if torch.distributed.is_initialized():
                sampler = DistributedSampler(dataloader.dataset)  # type: ignore[arg-type]
                dataloader = DataLoader(
                    dataloader.dataset,
                    batch_size=dataloader.batch_size,
                    sampler=sampler,
                    num_workers=dataloader.num_workers,
                    pin_memory=dataloader.pin_memory,
                    drop_last=dataloader.drop_last,
                )
        return dataloader

    def barrier(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def state_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self._model is not None:
            result["model"] = self._model.state_dict()
        if self._optimizer is not None:
            result["optimizer"] = self._optimizer.state_dict()
        return result

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._model is not None and "model" in state:
            self._model.load_state_dict(state["model"])
        if self._optimizer is not None and "optimizer" in state:
            self._optimizer.load_state_dict(state["optimizer"])

    def step(self) -> None:
        if self._optimizer is not None:
            self._optimizer.step()

    def zero_grad(self) -> None:
        if self._optimizer is not None:
            self._optimizer.zero_grad()
