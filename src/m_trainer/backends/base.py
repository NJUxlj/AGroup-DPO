"""DistributedBackend 抽象基类 (FR-07)

所有分布式后端的统一接口。业务代码通过此接口与后端交互，
无需关心底层是 DeepSpeed / FSDP / Megatron / accelerate。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainerConfig:
    """训练器配置，与 yaml trainer 段一一对应。

    Attributes:
        distributed_backend: 后端名称 (deepspeed / fsdp / megatron / accelerate)
        output_dir: checkpoint 输出目录
        per_device_batch_size: 每卡 batch size
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 学习率
        num_train_epochs: 训练 epoch 数
        bf16: 是否启用 bf16
        seed: 随机种子
        world_size: 总 GPU 数（自动推断）
        deepspeed_config: DeepSpeed ZeRO 配置字典（仅 deepspeed 后端使用）
        fsdp_config: FSDP 配置字典（仅 fsdp 后端使用）
    """

    distributed_backend: str = "deepspeed"
    output_dir: str = "runs/default"
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    num_train_epochs: float = 3.0
    bf16: bool = True
    seed: int = 42
    world_size: int = 1
    deepspeed_config: dict[str, Any] = field(default_factory=dict)
    fsdp_config: dict[str, Any] = field(default_factory=dict)


class DistributedBackend(ABC):
    """
    所有分布式后端的统一接口。

    每个后端适配器必须实现本接口的所有抽象方法。
    optimizer 的创建策略因后端而异：
      - DeepSpeed ZeRO3：由 deepspeed.initialize 内部创建并分片，
        此时 optimizer 参数传入 None，返回的 wrapped_optimizer 实际为 DeepSpeedEngine。
      - FSDP：由 backend 内部创建并通过 fully_shard 与 model 绑定。
      - accelerate：由 accelerator.prepare 统一包装。
      - Megatron：当前 1.5B 模型触发 NotImplementedError。
    """

    @abstractmethod
    def init(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        config: TrainerConfig,
    ) -> tuple[nn.Module, Any]:
        """
        初始化分布式环境。

        Args:
            model: 原始模型（未包装）
            optimizer: 外部创建的优化器，或 None（由 backend 内部创建）
            config: 训练器配置

        Returns:
            (wrapped_model, wrapped_optimizer):
              - DeepSpeed 下 wrapped_optimizer 为 DeepSpeedEngine
              - 其他后端下为 torch.optim.Optimizer 子类
        """

    @abstractmethod
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """对模型应用分片/并行策略（如 FSDP fully_shard 或 Megatron TP）。"""

    @abstractmethod
    def wrap_optimizer(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.Optimizer:
        """应用 ZeRO / FSDP 优化器包装。"""

    @abstractmethod
    def prepare_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """分发 sampler / 注入 rank-aware collator。"""

    @abstractmethod
    def barrier(self) -> None:
        """跨卡同步（等价于 torch.distributed.barrier）。"""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """统一状态字典，支持断点续训。"""

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """从统一状态字典恢复训练状态。"""

    def backward(self, loss: torch.Tensor) -> None:
        """反向传播。

        默认调用 loss.backward()，各后端可 override（如 DeepSpeed 需
        调用 engine.backward(loss) 以触发 ZeRO 通信）。
        """
        loss.backward()

    def step(self) -> None:
        """执行一步优化器更新。

        默认调用 optimizer.step()，DeepSpeed 需 override 为 engine.step()。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement step()"
        )

    def zero_grad(self) -> None:
        """清零梯度。

        默认调用 optimizer.zero_grad()，DeepSpeed 需 override。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement zero_grad()"
        )
