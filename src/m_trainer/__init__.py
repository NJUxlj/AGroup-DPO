# m_trainer: 自研分布式 Trainer 模块 (FR-07)
# 提供 4 种分布式后端（DeepSpeed / FSDP / Megatron / accelerate）的统一抽象与可插拔切换

from .callbacks import MetricsLogger, StepMetrics, build_metrics_callbacks
from .factory import build_backend, setup_distributed

__all__ = [
    "MetricsLogger",
    "StepMetrics",
    "build_backend",
    "build_metrics_callbacks",
    "setup_distributed",
]
