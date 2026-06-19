"""评测报告生成器 (FR-08)

产出 JSON + Markdown 双格式评测报告。
"""

from __future__ import annotations

import json
from utils.logger import CustomLogger
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .latency import LatencyStat

log = CustomLogger.get_logger(__name__)


class EvalReporter:
    """评测报告生成器。

    收集各评测集的指标 + 延迟统计，输出 JSON 和 Markdown。

    Usage:
        reporter = EvalReporter(
            model_version="qwen2_5_1_5b_insurance_dpo_v1.2",
            infer_backend="vllm",
        )
        reporter.add_dataset("medical_qa_1000", accuracy=0.682, bleu_4=0.341, rouge_l=0.482, n_samples=1000)
        reporter.set_latency(latency)
        reporter.write("reports/eval_report_dpo_v1.2")
    """

    def __init__(
        self,
        model_version: str = "unknown",
        infer_backend: str = "vllm",
        baseline_model: Optional[str] = None,
    ) -> None:
        self.model_version = model_version
        self.infer_backend = infer_backend
        self.baseline_model = baseline_model
        self._datasets: dict[str, dict[str, Any]] = {}
        self._latency: Optional[LatencyStat] = None
        self._baseline_metrics: dict[str, Any] = {}

    def add_dataset(
        self,
        name: str,
        accuracy: float = 0.0,
        bleu_4: float = 0.0,
        rouge_l: float = 0.0,
        n_samples: int = 0,
    ) -> None:
        """添加一个评测集的结果。"""
        self._datasets[name] = {
            "accuracy": round(accuracy, 4),
            "bleu_4": round(bleu_4, 4),
            "rouge_l": round(rouge_l, 4),
            "n_samples": n_samples,
        }

    def set_latency(self, latency: LatencyStat) -> None:
        """设置延迟统计。"""
        self._latency = latency

    def set_baseline(self, **metrics: float) -> None:
        """设置 baseline 对比指标。"""
        self._baseline_metrics = metrics

    def to_dict(self) -> dict[str, Any]:
        """转为字典（用于 JSON 序列化）。"""
        result: dict[str, Any] = {
            "model_version": self.model_version,
            "infer_backend": self.infer_backend,
            "datasets": self._datasets,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        if self._latency is not None:
            result["latency"] = {
                "p50_first_token_ms": round(self._latency.p50_first_token_ms, 1),
                "p95_first_token_ms": round(self._latency.p95_first_token_ms, 1),
                "p99_first_token_ms": round(self._latency.p99_first_token_ms, 1),
                "p50_total_ms": round(self._latency.p50_total_ms, 1),
                "p95_total_ms": round(self._latency.p95_total_ms, 1),
                "p99_total_ms": round(self._latency.p99_total_ms, 1),
                "throughput_samples_per_s": round(self._latency.throughput_samples_per_s, 1),
            }

        if self._baseline_metrics:
            comparison = {"baseline_model": self.baseline_model or "baseline"}
            for key, baseline_val in self._baseline_metrics.items():
                # 对比第一个（主）评测集
                if self._datasets:
                    main = next(iter(self._datasets.values()))
                    current_val = main.get(key, 0.0)
                    if baseline_val > 0:
                        gain = (current_val - baseline_val) / baseline_val * 100
                        comparison[f"{key}_relative_gain"] = f"{gain:+.1f}%"
                    comparison[f"baseline_{key}"] = baseline_val
            result["baseline_comparison"] = comparison

        return result

    def write(self, output_base: str) -> tuple[Path, Path]:
        """写入 JSON 和 Markdown 双格式报告。

        Args:
            output_base: 输出文件基名（不含扩展名）

        Returns:
            (json_path, md_path)
        """
        data = self.to_dict()
        base = Path(output_base)

        # JSON
        json_path = base.with_suffix(".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log.info("JSON report saved to %s", json_path)

        # Markdown
        md_path = base.with_suffix(".md")
        self._write_markdown(md_path, data)
        log.info("Markdown report saved to %s", md_path)

        return json_path, md_path

    def _write_markdown(self, path: Path, data: dict[str, Any]) -> None:
        """渲染 Markdown 报告。"""
        lines = [
            f"# 评测报告: {data['model_version']}",
            "",
            f"- **推理后端**: {data['infer_backend']}",
            f"- **生成时间**: {data['generated_at']}",
            "",
            "## 1. 评测指标",
            "",
            "| 数据集 | Accuracy | BLEU-4 | ROUGE-L | 样本数 |",
            "|--------|----------|--------|---------|--------|",
        ]

        for name, metrics in data.get("datasets", {}).items():
            lines.append(
                f"| {name} | {metrics['accuracy']:.4f} | {metrics['bleu_4']:.4f} | "
                f"{metrics['rouge_l']:.4f} | {metrics['n_samples']} |"
            )

        if "latency" in data:
            lat = data["latency"]
            lines.extend([
                "",
                "## 2. 推理延迟",
                "",
                "| 指标 | 值 |",
                "|------|-----|",
                f"| p50 首 token 时延 | {lat['p50_first_token_ms']:.1f} ms |",
                f"| p95 首 token 时延 | {lat['p95_first_token_ms']:.1f} ms |",
                f"| p99 首 token 时延 | {lat['p99_first_token_ms']:.1f} ms |",
                f"| p50 全句时延 | {lat['p50_total_ms']:.1f} ms |",
                f"| p95 全句时延 | {lat['p95_total_ms']:.1f} ms |",
                f"| p99 全句时延 | {lat['p99_total_ms']:.1f} ms |",
                f"| 吞吐量 | {lat['throughput_samples_per_s']:.1f} samples/s |",
            ])

        if "baseline_comparison" in data:
            bc = data["baseline_comparison"]
            lines.extend([
                "",
                "## 3. Baseline 对比",
                "",
                f"- **Baseline 模型**: {bc.get('baseline_model', 'N/A')}",
            ])
            for key, val in bc.items():
                if key != "baseline_model":
                    lines.append(f"- **{key}**: {val}")

        lines.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
