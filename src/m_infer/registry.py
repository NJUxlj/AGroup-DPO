"""推理后端注册表 (FR-08)

维护所有支持的推理后端名称 → 模块路径的映射。
与 m_trainer/registry.py 保持一致的注册模式。
"""

from __future__ import annotations

INFER_REGISTRY: dict[str, str] = {
    "vllm": "m_infer.vllm_backend:VLLMBackend",
    "xinference": "m_infer.xinference_backend:XinferenceBackend",
}


def list_backends() -> list[str]:
    """列出所有已注册的推理后端名称。"""
    return list(INFER_REGISTRY.keys())


def register_backend(name: str, module_class: str) -> None:
    """
    注册自定义推理后端。

    Args:
        name: 后端名称
        module_class: "module.path:ClassName" 格式

    Raises:
        ValueError: 若名称已存在
    """
    if name in INFER_REGISTRY:
        raise ValueError(
            f"infer backend '{name}' is already registered: {INFER_REGISTRY[name]}"
        )
    INFER_REGISTRY[name] = module_class


def unregister_backend(name: str) -> None:
    """
    注销推理后端。

    Args:
        name: 后端名称

    Raises:
        KeyError: 若名称不存在
    """
    if name not in INFER_REGISTRY:
        raise KeyError(f"infer backend '{name}' not found in registry")
    del INFER_REGISTRY[name]
