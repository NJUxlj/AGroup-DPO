"""M-EVAL: 评测指标与报告生成 (FR-08 下半)

提供 Accuracy / BLEU-4 / ROUGE-L 指标计算与 JSON + Markdown 报告产出。
"""

from .metrics import accuracy_score, bleu_4_score, rouge_l_score
from .latency import LatencyStat, aggregate_latency
from .reporter import EvalReporter

__all__ = [
    "accuracy_score",
    "bleu_4_score",
    "rouge_l_score",
    "LatencyStat",
    "aggregate_latency",
    "EvalReporter",
]
