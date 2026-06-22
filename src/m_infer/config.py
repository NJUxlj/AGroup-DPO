"""M-INFER 配置加载 (M05 § 3.4)

从 configs/infer.yaml 解析 backend / model_path 及后端特定参数。
CLI 显式参数优先于配置文件。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_infer_config(path: str) -> dict[str, Any]:
    """加载 infer.yaml 配置文件。"""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"infer config not found: {path}")
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "infer" not in data:
        raise ValueError(f"infer config missing top-level 'infer' key: {path}")
    return data


def resolve_infer_settings(
    cfg: dict[str, Any],
    *,
    backend: str | None = None,
    model_path: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """解析推理配置，返回 (backend_name, model_path, backend_kwargs)。

    Args:
        cfg: load_infer_config 返回的完整 YAML dict
        backend: CLI 覆盖的后端名称
        model_path: CLI 覆盖的模型路径
    """
    infer = cfg["infer"]
    resolved_backend = backend or infer.get("backend", "vllm")
    resolved_model = model_path or infer.get("model_path")
    if not resolved_model:
        raise ValueError("model_path is required (via config or --model)")

    backend_cfg = dict(infer.get(resolved_backend, {}) or {})
    # model_uid: null in yaml → treat as absent
    if backend_cfg.get("model_uid") is None and "model_uid" in backend_cfg:
        backend_cfg.pop("model_uid")

    return resolved_backend, resolved_model, backend_cfg
