"""测试优化器工厂。"""

import torch
import torch.nn as nn
import pytest

from m_trainer.backends.optimizer_factory import (
    AdamWFactory,
    DeepSpeedOptimizerFactory,
)


class TestOptimizerFactory:
    """优化器工厂测试。"""

    def test_adamw_factory(self):
        """AdamW 工厂正常构建优化器。"""
        model = nn.Linear(10, 10)
        factory = AdamWFactory()
        opt = factory.build(model.parameters(), {"learning_rate": 1e-4})
        assert isinstance(opt, torch.optim.AdamW)
        assert opt.param_groups[0]["lr"] == 1e-4

    def test_adamw_factory_default_lr(self):
        """AdamW 工厂默认学习率。"""
        model = nn.Linear(10, 10)
        factory = AdamWFactory()
        opt = factory.build(model.parameters(), {})
        assert opt.param_groups[0]["lr"] == 1e-4

    def test_deepspeed_factory_raises(self):
        """DeepSpeed 优化器工厂调用 build 抛出 RuntimeError。"""
        model = nn.Linear(10, 10)
        factory = DeepSpeedOptimizerFactory()
        with pytest.raises(RuntimeError, match="internally"):
            factory.build(model.parameters(), {})
