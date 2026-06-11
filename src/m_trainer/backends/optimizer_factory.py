"""OptimizerFactory 抽象基类 (FR-07)

根据后端选择合适的 optimizer 包装，屏蔽不同后端对优化器创建的差异。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class OptimizerFactory(ABC):
    """优化器工厂抽象基类。

    各后端对优化器的创建位置不同（DeepSpeed 内部创建，FSDP/accelerate 外部创建），
    工厂负责统一这一差异。
    """

    @abstractmethod
    def build(
        self,
        params: Any,
        base_cfg: dict[str, Any],
    ) -> torch.optim.Optimizer:
        """根据后端和配置构建优化器实例。

        Args:
            params: 模型参数迭代器（model.parameters()）
            base_cfg: 包含 lr / betas / weight_decay 等基础优化器超参的字典

        Returns:
            构建好的优化器实例（DeepSpeed 下返回 None，由 engine 内部创建）
        """


class AdamWFactory(OptimizerFactory):
    """标准 AdamW 优化器工厂（FSDP / accelerate 使用）。"""

    def build(
        self,
        params: Any,
        base_cfg: dict[str, Any],
    ) -> torch.optim.Optimizer:
        lr = base_cfg.get("learning_rate", 1e-4)
        betas = base_cfg.get("betas", (0.9, 0.999))
        weight_decay = base_cfg.get("weight_decay", 0.0)
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)


class DeepSpeedOptimizerFactory(OptimizerFactory):
    """DeepSpeed 优化器工厂 — 仅作占位。

    DeepSpeed 在 deepspeed.initialize 内部创建优化器，
    外部不应调用本工厂的 build 方法。
    """

    def build(
        self,
        params: Any,
        base_cfg: dict[str, Any],
    ) -> torch.optim.Optimizer:
        raise RuntimeError(
            "DeepSpeed optimizer is created internally by deepspeed.initialize. "
            "Do not call OptimizerFactory.build() for the DeepSpeed backend."
        )
