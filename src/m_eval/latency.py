"""推理延迟统计 (FR-08)

计算 p50/p95/p99 首 token 时延、全句时延，以及总吞吐量。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 延迟导入避免在纯 CPU 环境 import 报错
# InferResponse 在 m_infer.base 中定义


@dataclass
class LatencyStat:
    """推理延迟统计汇总。

    Attributes:
        p50_first_token_ms: 首 token 时延 p50（ms）
        p95_first_token_ms: 首 token 时延 p95（ms）
        p99_first_token_ms: 首 token 时延 p99（ms）
        p50_total_ms: 全句时延 p50（ms）
        p95_total_ms: 全句时延 p95（ms）
        p99_total_ms: 全句时延 p99（ms）
        throughput_samples_per_s: 吞吐量（samples/sec）
    """

    p50_first_token_ms: float = 0.0
    p95_first_token_ms: float = 0.0
    p99_first_token_ms: float = 0.0
    p50_total_ms: float = 0.0
    p95_total_ms: float = 0.0
    p99_total_ms: float = 0.0
    throughput_samples_per_s: float = 0.0


def aggregate_latency(records: list) -> LatencyStat:
    """从 InferResponse 列表中计算延迟统计。

    Args:
        records: InferResponse 列表（包含 latency_ms / total_latency_ms）

    Returns:
        LatencyStat 汇总
    """
    if not records:
        return LatencyStat()

    first = np.array([r.latency_ms for r in records if r.latency_ms > 0])
    total = np.array([r.total_latency_ms for r in records if r.total_latency_ms > 0])

    elapsed = sum(r.total_latency_ms for r in records) / 1000.0

    def _p(arr, q):
        return float(np.percentile(arr, q)) if len(arr) > 0 else 0.0

    return LatencyStat(
        p50_first_token_ms=_p(first, 50),
        p95_first_token_ms=_p(first, 95),
        p99_first_token_ms=_p(first, 99),
        p50_total_ms=_p(total, 50),
        p95_total_ms=_p(total, 95),
        p99_total_ms=_p(total, 99),
        throughput_samples_per_s=len(records) / elapsed if elapsed > 0 else 0.0,
    )
