"""推理后端工厂 (FR-08)

通过配置文件中的字符串名称自动实例化对应的 InferBackend 实现。
与 m_trainer/factory.py 保持一致的工厂模式。
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

from .registry import INFER_REGISTRY

if TYPE_CHECKING:
    from .base import InferBackend

logger = logging.getLogger(__name__)


def build_infer_backend(name: str, model_path: str, **kwargs) -> "InferBackend":
    """根据后端名称构建推理后端实例并加载模型。

    Args:
        name: 后端名称 (vllm / xinference)
        model_path: 模型路径（HF 格式目录）
        **kwargs: 后端特定参数

    Returns:
        实现了 InferBackend ABC 的推理后端实例（已调用 load）

    Raises:
        ValueError: 若指定的后端名称不在注册表中
        ImportError: 若后端模块无法导入

    Usage:
        from m_infer.factory import build_infer_backend

        backend = build_infer_backend("vllm", "merged_models/...", tensor_parallel_size=1)
        resp = backend.infer(InferRequest(prompt="你好"))
    """
    if name not in INFER_REGISTRY:
        raise ValueError(
            f"unsupported infer backend: '{name}'. "
            f"Available backends: {list(INFER_REGISTRY.keys())}"
        )

    module_path, cls_name = INFER_REGISTRY[name].split(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Failed to import infer backend module '{module_path}' for '{name}'. "
            f"Is the required package installed? Original error: {e}"
        ) from e

    if not hasattr(module, cls_name):
        raise AttributeError(
            f"Module '{module_path}' does not export class '{cls_name}'. "
            f"Check the INFER_REGISTRY entry for '{name}'."
        )

    backend_cls = getattr(module, cls_name)
    backend = backend_cls()
    backend.load(model_path, **kwargs)
    logger.info("Built infer backend: %s → %s.%s (model=%s)", name, module_path, cls_name, model_path)
    return backend
