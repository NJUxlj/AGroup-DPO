"""评测 CLI 入口 (FR-08)

运行完整的评测流水线：加载数据 → 推理 → 计算指标 → 产出报告。
支持 configs/eval.yaml 配置驱动与 baseline 报告对比（M05 § 7.5）。
"""

from __future__ import annotations

import argparse
import json
from utils.logger import CustomLogger
import sys
import time
from pathlib import Path
from typing import Any, Optional

from m_infer.base import InferRequest
from m_infer.factory import build_infer_backend
from m_eval.config import load_eval_config, resolve_eval_settings
from m_eval.metrics import accuracy_score, bleu_4_score, rouge_l_score
from m_eval.latency import aggregate_latency
from m_eval.reporter import EvalReporter

log = CustomLogger.get_logger(__name__)


def load_eval_data(path: str) -> list[dict]:
    """加载评测数据集（JSON Lines 格式）。

    每行: {"question": "...", "reference_answer": "...", "answer_type": "open"|"choice", ...}
    """
    data_path = Path(path)
    samples = []
    if data_path.is_file():
        for line in open(data_path, encoding="utf-8"):
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    elif data_path.is_dir():
        for f in sorted(data_path.glob("*.jsonl")):
            dataset_name = f.stem
            dataset_samples = []
            for line in open(f, encoding="utf-8"):
                line = line.strip()
                if line:
                    dataset_samples.append(json.loads(line))
            if dataset_samples:
                samples.append({"__dataset_name__": dataset_name, "__samples__": dataset_samples})
    else:
        raise FileNotFoundError(f"eval data path not found: {path}")

    return samples


def load_baseline_metrics(report_path: str) -> tuple[str, dict[str, float]]:
    """从 baseline 评测报告 JSON 加载对比指标。"""
    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)

    baseline_model = data.get("model_version", "baseline")
    metrics: dict[str, float] = {}

    datasets = data.get("datasets", {})
    if datasets:
        main = next(iter(datasets.values()))
        for key in ("accuracy", "bleu_4", "rouge_l"):
            if key in main:
                metrics[key] = main[key]

    return baseline_model, metrics


def run_eval(
    model_path: str,
    eval_data: str,
    output_base: str,
    backend_name: str = "vllm",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    baseline_report: Optional[str] = None,
    thresholds: Optional[dict[str, float]] = None,
    **backend_kwargs,
) -> int:
    """执行完整评测流水线。

    Returns:
        0 表示成功，1 表示失败
    """
    t_start = time.perf_counter()

    log.info("Loading eval data from %s ...", eval_data)
    raw = load_eval_data(eval_data)
    log.info("Loaded %d entries", len(raw))

    log.info("Building infer backend: %s, model=%s", backend_name, model_path)
    backend = build_infer_backend(backend_name, model_path, **backend_kwargs)

    model_version = Path(model_path).name if Path(model_path).exists() else model_path

    baseline_model = None
    baseline_metrics: dict[str, float] = {}
    if baseline_report:
        baseline_model, baseline_metrics = load_baseline_metrics(baseline_report)
        log.info("Loaded baseline report: %s from %s", baseline_model, baseline_report)

    reporter = EvalReporter(
        model_version=model_version,
        infer_backend=backend_name,
        baseline_model=baseline_model,
    )
    if baseline_metrics:
        reporter.set_baseline(**baseline_metrics)

    all_responses = []
    datasets_mode = any("__dataset_name__" in item for item in raw)

    if datasets_mode:
        for item in raw:
            dataset_name = item["__dataset_name__"]
            samples = item["__samples__"]
            log.info("Evaluating dataset: %s (%d samples)", dataset_name, len(samples))
            _eval_dataset(
                backend, samples, dataset_name, reporter,
                max_new_tokens, temperature, all_responses,
            )
    else:
        _eval_dataset(
            backend, raw, "eval", reporter,
            max_new_tokens, temperature, all_responses,
        )

    if all_responses:
        latency = aggregate_latency(all_responses)
        reporter.set_latency(latency)

    backend.shutdown()

    reporter.write(output_base)

    elapsed = time.perf_counter() - t_start
    log.info("Evaluation complete in %.1fs (%.1f min)", elapsed, elapsed / 60)

    thresholds = thresholds or {}
    bleu_threshold = thresholds.get("bleu_4", 0.30)
    rouge_threshold = thresholds.get("rouge_l", 0.45)
    for name, metrics in reporter._datasets.items():
        if metrics["bleu_4"] < bleu_threshold:
            log.warning(
                "BLEU-4 for '%s' (%.4f) below threshold %.2f",
                name, metrics["bleu_4"], bleu_threshold,
            )
        if metrics["rouge_l"] < rouge_threshold:
            log.warning(
                "ROUGE-L for '%s' (%.4f) below threshold %.2f",
                name, metrics["rouge_l"], rouge_threshold,
            )

    return 0


def _eval_dataset(
    backend, samples, dataset_name, reporter,
    max_new_tokens, temperature, all_responses,
):
    requests = [
        InferRequest(
            prompt=s.get("question", s.get("prompt", "")),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        for s in samples
    ]

    responses = backend.batch_infer(requests)
    all_responses.extend(responses)

    preds = [r.text for r in responses]
    refs = [s.get("reference_answer", s.get("ref", "")) for s in samples]

    needs_judge = any(s.get("judge_required", False) for s in samples)
    acc = accuracy_score(
        preds, refs,
        samples=samples,
        judge_backend=backend if needs_judge else None,
    )
    bleu = bleu_4_score(preds, refs)
    rouge = rouge_l_score(preds, refs)

    log.info(
        "  %s: accuracy=%.4f, bleu_4=%.4f, rouge_l=%.4f",
        dataset_name, acc, bleu, rouge,
    )
    reporter.add_dataset(
        dataset_name, accuracy=acc, bleu_4=bleu, rouge_l=rouge, n_samples=len(samples),
    )


def main():
    parser = argparse.ArgumentParser(
        description="M-EVAL: 保险问答评测流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 配置驱动多评测集（M05 § 7.5）
  copaw-dpo evaluate --config configs/eval.yaml \\
      --model merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \\
      --output reports/eval_report_dpo_v1.2

  # 单评测集
  copaw-dpo evaluate --model merged_models/... \\
      --eval-data data/eval/medical_qa_1000.jsonl \\
      --output reports/eval_report_dpo_v1.2

  # baseline 对比
  copaw-dpo evaluate --model merged_models/... \\
      --eval-data data/eval/ \\
      --output reports/eval_report_dpo_v1.2 \\
      --baseline-report reports/eval_report_baseline_v0.json
        """,
    )
    parser.add_argument("--config", "-c", default=None,
                        help="评测配置文件（configs/eval.yaml）")
    parser.add_argument("--model", default=None, help="模型路径（HF 格式目录）")
    parser.add_argument("--eval-data", default=None,
                        help="评测数据集路径（.jsonl 文件或目录；config 模式下可省略）")
    parser.add_argument("--output", default=None,
                        help="报告输出基名（不含扩展名）")
    parser.add_argument("--backend", choices=["vllm", "xinference"], default="vllm")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--baseline-report", default=None,
                        help="baseline 评测报告 JSON，用于产出 baseline_comparison")

    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--xinference-endpoint", default="http://127.0.0.1:9997")

    args = parser.parse_args()

    CustomLogger.configure(
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    thresholds: dict[str, float] = {}
    try:
        if args.config:
            cfg = load_eval_config(args.config)
            eval_settings = resolve_eval_settings(cfg)
            thresholds = eval_settings.get("thresholds", {})

            model_path = args.model
            if not model_path:
                raise ValueError("--model is required (even with --config)")

            if args.eval_data:
                eval_data = args.eval_data
            elif eval_settings["datasets"]:
                # config 模式：临时目录不可行，直接用 datasets 路径
                # 若只有一个 dataset 用其 path，否则用父目录
                ds_list = eval_settings["datasets"]
                if len(ds_list) == 1:
                    eval_data = ds_list[0]["path"]
                else:
                    # 多数据集：构造与 load_eval_data 目录模式兼容的路径
                    parent = Path(ds_list[0]["path"]).parent
                    eval_data = str(parent)
            else:
                raise ValueError("--eval-data is required when config has no datasets")

            output_base = args.output or str(
                Path(eval_settings["output_dir"]) / "eval_report"
            )
            max_new_tokens = (
                args.max_new_tokens
                if args.max_new_tokens is not None
                else eval_settings["max_new_tokens"]
            )
            temperature = (
                args.temperature
                if args.temperature is not None
                else eval_settings["temperature"]
            )
        else:
            if not args.model or not args.eval_data:
                parser.error("--model and --eval-data are required when --config is not provided")
            model_path = args.model
            eval_data = args.eval_data
            output_base = args.output or "reports/eval_report"
            max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else 256
            temperature = args.temperature if args.temperature is not None else 0.7

        backend_kwargs: dict[str, Any] = {}
        if args.backend == "vllm":
            backend_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
            backend_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
        elif args.backend == "xinference":
            backend_kwargs["server_endpoint"] = args.xinference_endpoint

        ret = run_eval(
            model_path=model_path,
            eval_data=eval_data,
            output_base=output_base,
            backend_name=args.backend,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            baseline_report=args.baseline_report,
            thresholds=thresholds,
            **backend_kwargs,
        )
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)

    sys.exit(ret)


if __name__ == "__main__":
    main()
