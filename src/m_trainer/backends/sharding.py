"""ShardingStrategy 抽象基类 (FR-07)

决定哪些层/参数被分片，不同后端可注册不同策略。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn


class ShardingStrategy(ABC):
    """分片策略抽象基类。

    子类决定模型中的哪些层/参数参与分片（如 ZeRO3 全部分片，
    FSDP 按 transformer block 粒度分片）。
    """

    @abstractmethod
    def apply(self, model: nn.Module) -> nn.Module:
        """对模型应用分片策略，返回分片后的模型。"""


class FullShardingStrategy(ShardingStrategy):
    """全参数分片（DeepSpeed ZeRO3 默认行为）。"""

    def apply(self, model: nn.Module) -> nn.Module:
        # DeepSpeed ZeRO3 在 deepspeed.initialize 中自动处理，
        # 不需要在本层做额外分片。
        return model


class NoShardingStrategy(ShardingStrategy):
    """不分片（accelerate 单机多卡调试场景）。"""

    def apply(self, model: nn.Module) -> nn.Module:
        return model
