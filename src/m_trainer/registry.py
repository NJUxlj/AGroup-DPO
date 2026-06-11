"""后端注册表 (FR-07)

维护后端名称 → 适配器类的映射。业务方可通过注册表
查询可用后端，也可注册自定义后端（高级用法）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backends.base import DistributedBackend

BACKEND_REGISTRY: dict[str, str] = {
    "deepspeed": "m_trainer.backends.deepspeed:DeepSpeedBackend",
    "fsdp": "m_trainer.backends.fsdp:FSDPBackend",
    "megatron": "m_trainer.backends.megatron:MegatronBackend",
    "accelerate": "m_trainer.backends.accelerate:AccelerateBackend",
}


def list_backends() -> list[str]:
    """列出所有已注册的后端名称。"""
    return list(BACKEND_REGISTRY.keys())


def register_backend(name: str, full_qualname: str) -> None:
    """注册自定义后端（高级用法）。

    Args:
        name: 后端简称（如 "my_custom_backend"）
        full_qualname: 完全限定名（如 "my_package.backends.custom:CustomBackend"）

    Raises:
        ValueError: 若后端名已被注册
    """
    if name in BACKEND_REGISTRY:
        raise ValueError(
            f"backend '{name}' is already registered. "
            f"Use a different name or unregister it first."
        )
    BACKEND_REGISTRY[name] = full_qualname


def unregister_backend(name: str) -> None:
    """取消注册后端。

    Args:
        name: 后端简称

    Raises:
        KeyError: 若后端名不存在
    """
    if name not in BACKEND_REGISTRY:
        raise KeyError(f"backend '{name}' not found in registry.")
    del BACKEND_REGISTRY[name]
