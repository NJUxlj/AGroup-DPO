"""测试后端工厂 build_backend。"""

import pytest
import torch.nn as nn

from m_trainer.backends.base import TrainerConfig
from m_trainer.factory import build_backend
from m_trainer.backends.deepspeed import DeepSpeedBackend
from m_trainer.backends.accelerate import AccelerateBackend
from m_trainer.backends.megatron import MegatronBackend, detect_model_type


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
        """构建 Megatron 后端（实例化成功，init 在无 megatron-core 时 graceful）。"""
        cfg = TrainerConfig(distributed_backend="megatron")
        backend = build_backend(cfg)
        assert isinstance(backend, MegatronBackend)

    def test_megatron_init_with_unsupported_model(self):
        """Megatron init 对未知模型抛出 ValueError。"""
        cfg = TrainerConfig(distributed_backend="megatron")
        backend = build_backend(cfg)
        dummy_model = nn.Linear(10, 10)  # 不是 GPT2/Qwen2
        with pytest.raises(ValueError, match="Unsupported model type"):
            backend.init(dummy_model, None, cfg)

    def test_megatron_init_tp1_no_megatron_core(self):
        """TP=1 时无需 megatron-core 即可初始化。"""
        cfg = TrainerConfig(
            distributed_backend="megatron",
            megatron_config={"tensor_model_parallel_size": 1},
        )
        backend = build_backend(cfg)
        dummy_model = DummyGPT2Model()
        model, opt = backend.init(dummy_model, None, cfg)
        assert model is not None
        assert opt is not None
        assert isinstance(opt, nn.Module) or hasattr(opt, 'step')

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

    def test_trainer_config_megatron_field(self):
        """TrainerConfig 包含 megatron_config 字段。"""
        cfg = TrainerConfig(
            megatron_config={"tensor_model_parallel_size": 2}
        )
        assert cfg.megatron_config["tensor_model_parallel_size"] == 2


class TestDetectModelType:
    """模型类型检测测试。"""

    def test_detect_gpt2(self):
        """检测 GPT2 模型。"""
        model = DummyGPT2Model()
        assert detect_model_type(model) == "gpt2"

    def test_detect_qwen2(self):
        """检测 Qwen2 模型。"""
        model = DummyQwen2Model()
        assert detect_model_type(model) == "qwen2"

    def test_detect_unknown(self):
        """未知模型返回 'unknown'。"""
        model = nn.Linear(10, 10)
        assert detect_model_type(model) == "unknown"


# ---- Dummy 模型用于测试 ----

class DummyGPT2Config:
    model_type = "gpt2"
    architectures = ["GPT2LMHeadModel"]


class DummyGPT2Model(nn.Module):
    """最小 GPT2 模型桩，用于测试模型检测。"""
    def __init__(self):
        super().__init__()
        self.config = DummyGPT2Config()
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(1024, 64)
        self.transformer.h = nn.ModuleList([
            self._make_block() for _ in range(2)
        ])
        self.lm_head = nn.Linear(64, 1024)

    def _make_block(self):
        block = nn.Module()
        block.attn = nn.Module()
        block.attn.c_attn = nn.Linear(64, 192)
        block.attn.c_proj = nn.Linear(64, 64)
        block.mlp = nn.Module()
        block.mlp.c_fc = nn.Linear(64, 256)
        block.mlp.c_proj = nn.Linear(256, 64)
        return block


class DummyQwen2Config:
    model_type = "qwen2"
    architectures = ["Qwen2ForCausalLM"]


class DummyQwen2Model(nn.Module):
    """最小 Qwen2 模型桩，用于测试模型检测。"""
    def __init__(self):
        super().__init__()
        self.config = DummyQwen2Config()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(1024, 64)
        self.model.layers = nn.ModuleList([
            self._make_layer() for _ in range(2)
        ])
        self.lm_head = nn.Linear(64, 1024, bias=False)

    def _make_layer(self):
        layer = nn.Module()
        # self_attn
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(64, 64, bias=False)
        layer.self_attn.k_proj = nn.Linear(64, 16, bias=False)
        layer.self_attn.v_proj = nn.Linear(64, 16, bias=False)
        layer.self_attn.o_proj = nn.Linear(64, 64, bias=False)
        # mlp
        layer.mlp = nn.Module()
        layer.mlp.gate_proj = nn.Linear(64, 128, bias=False)
        layer.mlp.up_proj = nn.Linear(64, 128, bias=False)
        layer.mlp.down_proj = nn.Linear(128, 64, bias=False)
        return layer
