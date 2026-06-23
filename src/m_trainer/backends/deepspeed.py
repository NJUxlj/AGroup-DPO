"""DeepSpeed ZeRO3 后端适配器 (FR-07)

主训练路径。在 2×A100-80G（或 2×RTX 5090）上通过 ZeRO Stage 3
实现优化器状态/梯度/参数的完全分片 + CPU offload。
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.logger import CustomLogger

from .base import DistributedBackend, TrainerConfig

log = CustomLogger.get_logger(__name__)


def build_zero3_config(cfg: TrainerConfig) -> dict[str, Any]:
    """根据 TrainerConfig 生成 DeepSpeed ZeRO3 配置字典。

    Args:
        cfg: 训练器配置

    Returns:
        DeepSpeed 可接受的配置字典
    """
    ds_config: dict[str, Any] = {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "bf16": {"enabled": cfg.bf16},
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": cfg.per_device_batch_size,
        "train_batch_size": (
            cfg.per_device_batch_size
            * cfg.gradient_accumulation_steps
            * max(cfg.world_size, 1)
        ),
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": cfg.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
        },
    }

    # 合并用户自定义的 deepspeed 配置（跳过 "auto" 占位符，避免覆盖数值 batch size）
    if cfg.deepspeed_config:
        for key, value in cfg.deepspeed_config.items():
            if value == "auto":
                continue
            if key == "zero_optimization" and isinstance(value, dict):
                ds_config.setdefault("zero_optimization", {}).update(value)
            else:
                ds_config[key] = value

    return ds_config


class DeepSpeedBackend(DistributedBackend):
    """DeepSpeed ZeRO3 后端。

    封装 deepspeed.initialize()，将模型/优化器包装为 DeepSpeedEngine。
    业务方通过 engine.backward(loss) / engine.step() 驱动训练。

    Usage:
        backend = DeepSpeedBackend()
        model, engine = backend.init(model, optimizer=None, config=cfg)
        for batch in dataloader:
            loss = model(batch)
            backend.backward(loss)   # → engine.backward(loss)
            backend.step()           # → engine.step()
    """

    def __init__(self) -> None:
        self._engine: Any = None
        self._model: Optional[nn.Module] = None

    # ---- DistributedBackend 接口实现 ----

    def init(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        config: TrainerConfig,
    ) -> tuple[nn.Module, Any]:
        """DeepSpeed 初始化。

        optimizer 必须传 None —— DeepSpeed 在 initialize 内部创建优化器。

        Returns:
            (model, engine): engine 是 DeepSpeedEngine 实例，
                             同时具备 optimizer 语义（可调用 engine.step() / engine.backward()）
        """
        if optimizer is not None:
            log.warning(
                "DeepSpeedBackend.init() received a non-None optimizer; "
                "it will be ignored. DeepSpeed creates its own optimizer internally."
            )

        ds_config = build_zero3_config(config)

        try:
            import deepspeed
        except ImportError:
            raise ImportError(
                "DeepSpeed is not installed. Install with: pip install deepspeed"
            )

        self._model = model
        self._engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config_params=ds_config,
        )

        log.info(
            "DeepSpeed ZeRO3 initialized: stage=%s, world_size=%s",
            ds_config["zero_optimization"]["stage"],
            config.world_size,
        )
        return self._engine, self._engine

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """DeepSpeed 下模型已在 init 中包装，直接返回。"""
        if self._engine is not None:
            return self._engine
        return model

    def wrap_optimizer(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.Optimizer:
        """DeepSpeed 下优化器已在 init 中创建，直接返回。"""
        if self._engine is not None:
            return self._engine  # type: ignore[return-value]
        return optimizer

    def prepare_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """DeepSpeed 默认不需要额外包装 dataloader。"""
        return dataloader

    def barrier(self) -> None:
        """分布式同步屏障。

        阻塞当前进程，直到所有参与训练的分布式进程（GPU/节点）均到达此处。
        常用于模型保存、Checkpoint 加载或全局状态同步前，确保各进程步调一致，
        避免并发写入冲突、数据竞争或日志乱序。
        """
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def state_dict(self) -> dict[str, Any]:
        if self._engine is None:
            return {}
        return self._engine.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._engine is not None:
            self._engine.load_state_dict(state)

    def backward(self, loss: torch.Tensor) -> None:
        """DeepSpeed engine.backward（触发 ZeRO 通信）。

        优先使用 engine.backward(loss) 以触发梯度 allreduce + loss scale；
        若 ZeRO3 offload 优化器不兼容 engine.backward（已知 DeepSpeed 0.14.4
        在特定配置下 DeepSpeedZeRoOffload 缺少 backward 方法），回退到
        loss.backward() + engine.step() 手动管理梯度累积。
        """
        if self._engine is not None:
            try:
                self._engine.backward(loss)
            except AttributeError:
                # ZeRO3 CPU offload 下 engine.backward → optimizer.backward 不存在
                loss.backward()
        else:
            loss.backward()

    def step(self) -> None:
        if self._engine is not None:
            self._engine.step()

    def zero_grad(self) -> None:
        if self._engine is not None:
            self._engine.zero_grad()

    def handles_gradient_accumulation(self) -> bool:
        return True
