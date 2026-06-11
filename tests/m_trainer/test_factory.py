"""测试后端工厂 build_backend。"""

import pytest

from m_trainer.backends.base import TrainerConfig
from m_trainer.factory import build_backend
from m_trainer.backends.deepspeed import DeepSpeedBackend
from m_trainer.backends.accelerate import AccelerateBackend
from m_trainer.backends.megatron import MegatronBackend


class TestFactory:
    """后端工厂测试。"""

    def test_build_deepspeed(self):
        """构建 DeepSpeed 后端。"""
        cfg = TrainerConfig(distributed_backend="deepspeed")
        backend = build_backend(cfg)
        assert isinstance(backend, DeepSpeedBackend)

    def test_build_accelerate(self):
        """构建 accelerate 后端。"""
        cfg = TrainerConfig(distributed_backend="accelerate")
        backend = build_backend(cfg)
        assert isinstance(backend, AccelerateBackend)

    def test_build_megatron(self):
        """构建 Megatron 后端（预期触发 NotImplementedError 但实例化成功）。"""
        cfg = TrainerConfig(distributed_backend="megatron")
        backend = build_backend(cfg)
        assert isinstance(backend, MegatronBackend)
        # 但调用 init() 会抛 NotImplementedError
        with pytest.raises(NotImplementedError, match="7B\\+"):
            backend.init(None, None, cfg)  # type: ignore[arg-type]

    def test_build_unknown_backend_raises(self):
        """未知后端抛出 ValueError。"""
        cfg = TrainerConfig(distributed_backend="nonexistent")
        with pytest.raises(ValueError, match="unsupported backend"):
            build_backend(cfg)

    def test_trainer_config_defaults(self):
        """TrainerConfig 默认值正确。"""
        cfg = TrainerConfig()
        assert cfg.distributed_backend == "deepspeed"
        assert cfg.per_device_batch_size == 2
        assert cfg.gradient_accumulation_steps == 8
        assert cfg.bf16 is True
        assert cfg.seed == 42
