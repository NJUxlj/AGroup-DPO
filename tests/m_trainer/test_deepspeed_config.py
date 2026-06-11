"""测试 DeepSpeed ZeRO3 配置生成器。"""

from m_trainer.backends.base import TrainerConfig
from m_trainer.backends.deepspeed import build_zero3_config


class TestDeepSpeedConfig:
    """DeepSpeed 配置生成测试。"""

    def test_build_minimal_config(self):
        """最小配置生成正确。"""
        cfg = TrainerConfig(
            per_device_batch_size=2,
            gradient_accumulation_steps=8,
            world_size=1,
        )
        ds = build_zero3_config(cfg)
        assert ds["zero_optimization"]["stage"] == 3
        assert ds["bf16"]["enabled"] is True
        assert ds["train_micro_batch_size_per_gpu"] == 2
        assert ds["gradient_accumulation_steps"] == 8
        assert ds["train_batch_size"] == 2 * 8 * 1

    def test_build_config_with_custom(self):
        """自定义 deepspeed_config 被合并。"""
        cfg = TrainerConfig(
            deepspeed_config={"zero_optimization": {"stage": 2}},
        )
        ds = build_zero3_config(cfg)
        # 用户覆盖了 stage
        assert ds["zero_optimization"]["stage"] == 2

    def test_train_batch_size_calculation(self):
        """train_batch_size 计算正确：bs × ga × world_size。"""
        cfg = TrainerConfig(
            per_device_batch_size=4,
            gradient_accumulation_steps=16,
            world_size=2,
        )
        ds = build_zero3_config(cfg)
        assert ds["train_batch_size"] == 4 * 16 * 2
