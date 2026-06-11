"""测试分片策略。"""

import torch.nn as nn

from m_trainer.backends.sharding import (
    FullShardingStrategy,
    NoShardingStrategy,
)


class TestSharding:
    """分片策略测试。"""

    def test_full_sharding_identity(self):
        """FullSharding 返回原模型（分片由后端内部处理）。"""
        model = nn.Linear(10, 10)
        strategy = FullShardingStrategy()
        result = strategy.apply(model)
        assert result is model

    def test_no_sharding_identity(self):
        """NoSharding 返回原模型。"""
        model = nn.Linear(10, 10)
        strategy = NoShardingStrategy()
        result = strategy.apply(model)
        assert result is model
