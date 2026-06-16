"""
后端工厂 (FR-07)

通过配置文件中的字符串名称自动实例化对应的 DistributedBackend 实现。
支持动态导入，业务方无需硬编码后端类。
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

from .registry import BACKEND_REGISTRY

if TYPE_CHECKING:
    from .backends.base import DistributedBackend, TrainerConfig

logger = logging.getLogger(__name__)


def build_backend(config: "TrainerConfig") -> "DistributedBackend":
    """根据 TrainerConfig 的 distributed_backend 字段构建后端实例。

    Args:
        config: 训练器配置，其中 distributed_backend 字段指定后端名称

    Returns:
        实现了 DistributedBackend ABC 的后端实例

    Raises:
        ValueError: 若指定的后端名称不在注册表中
        ImportError: 若后端模块无法导入

    Usage:
        from m_trainer.factory import build_backend
        from m_trainer.backends.base import TrainerConfig

        cfg = TrainerConfig(distributed_backend="deepspeed")
        backend = build_backend(cfg)
        model, engine = backend.init(model, optimizer=None, config=cfg)
    """
    name = config.distributed_backend

    if name not in BACKEND_REGISTRY:
        raise ValueError(
            f"unsupported backend: '{name}'. "
            f"Available backends: {list(BACKEND_REGISTRY.keys())}"
        )

    module_path, cls_name = BACKEND_REGISTRY[name].split(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Failed to import backend module '{module_path}' for backend '{name}'. "
            f"Is the required package installed? Original error: {e}"
        ) from e

    if not hasattr(module, cls_name):
        raise AttributeError(
            f"Module '{module_path}' does not export class '{cls_name}'. "
            f"Check the BACKEND_REGISTRY entry for '{name}'."
        )

    backend_cls = getattr(module, cls_name)
    logger.info("Built backend: %s → %s.%s", name, module_path, cls_name)
    return backend_cls()
