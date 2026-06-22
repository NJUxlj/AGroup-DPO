"""M-EVAL 配置加载 (M05 D-M05-12)

从 configs/eval.yaml 解析评测数据集、采样参数与验收阈值。
CLI 显式参数优先于配置文件。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_eval_config(path: str) -> dict[str, Any]:
    """加载 eval.yaml 配置文件。"""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"eval config not found: {path}")
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "eval" not in data:
        raise ValueError(f"eval config missing top-level 'eval' key: {path}")
    return data


def resolve_eval_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """解析评测配置为扁平 dict，供 CLI / run_eval 使用。"""
    ev = cfg["eval"]
    return {
        "output_dir": ev.get("output_dir", "reports/"),
        "max_new_tokens": ev.get("max_new_tokens", 256),
        "temperature": ev.get("temperature", 0.3),
        "datasets": ev.get("datasets", []),
        "thresholds": ev.get("thresholds", {}),
    }
