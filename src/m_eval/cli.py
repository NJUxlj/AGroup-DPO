"""评测 CLI 入口 (FR-08)

运行完整的评测流水线：加载数据 → 推理 → 计算指标 → 产出报告。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from m_infer.base import InferRequest
from m_infer.factory import build_infer_backend
from m_eval.metrics import accuracy_score, bleu_4_score, rouge_l_score
from m_eval.latency import aggregate_latency
from m_eval.reporter import EvalReporter

logger = logging.getLogger(__name__)


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
        # 多数据集模式：加载目录下所有 .jsonl 文件
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


def run_eval(
    model_path: str,
    eval_data: str,
    output_base: str,
    backend_name: str = "vllm",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    **backend_kwargs,
) -> int:
    """执行完整评测流水线。

    Returns:
        0 表示成功，1 表示失败
    """
    t_start = time.perf_counter()

    # 1. 加载数据
    logger.info("Loading eval data from %s ...", eval_data)
    raw = load_eval_data(eval_data)
    logger.info("Loaded %d entries", len(raw))

    # 2. 构建推理后端
    logger.info("Building infer backend: %s, model=%s", backend_name, model_path)
    backend = build_infer_backend(backend_name, model_path, **backend_kwargs)

    # 确定模型版本
    model_version = Path(model_path).name if Path(model_path).exists() else model_path

    reporter = EvalReporter(
        model_version=model_version,
        infer_backend=backend_name,
    )

    all_responses = []

    # 3. 逐数据集/逐样本推理
    datasets_mode = any("__dataset_name__" in item for item in raw)

    if datasets_mode:
        for item in raw:
            dataset_name = item["__dataset_name__"]
            samples = item["__samples__"]
            logger.info("Evaluating dataset: %s (%d samples)", dataset_name, len(samples))
            _eval_dataset(
                backend, samples, dataset_name, reporter,
                max_new_tokens, temperature, all_responses,
            )
    else:
        _eval_dataset(
            backend, raw, "eval", reporter,
            max_new_tokens, temperature, all_responses,
        )

    # 4. 延迟统计
    if all_responses:
        latency = aggregate_latency(all_responses)
        reporter.set_latency(latency)

    backend.shutdown()

    # 5. 产出报告
    json_path, md_path = reporter.write(output_base)

    elapsed = time.perf_counter() - t_start
    logger.info("Evaluation complete in %.1fs (%.1f min)", elapsed, elapsed / 60)

    # 6. 检查验收标准
    for name, metrics in reporter._datasets.items():
        if metrics["bleu_4"] < 0.30:
            logger.warning("BLEU-4 for '%s' (%.4f) below threshold 0.30", name, metrics["bleu_4"])
        if metrics["rouge_l"] < 0.45:
            logger.warning("ROUGE-L for '%s' (%.4f) below threshold 0.45", name, metrics["rouge_l"])

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

    # 批量推理
    responses = backend.batch_infer(requests)
    all_responses.extend(responses)

    # 提取预测和参考
    preds = [r.text for r in responses]
    refs = [s.get("reference_answer", s.get("ref", "")) for s in samples]

    # 计算指标
    acc = accuracy_score(preds, refs)
    bleu = bleu_4_score(preds, refs)
    rouge = rouge_l_score(preds, refs)

    logger.info(
        "  %s: accuracy=%.4f, bleu_4=%.4f, rouge_l=%.4f",
        dataset_name, acc, bleu, rouge,
    )
    reporter.add_dataset(dataset_name, accuracy=acc, bleu_4=bleu, rouge_l=rouge, n_samples=len(samples))


def main():
    parser = argparse.ArgumentParser(
        description="M-EVAL: 保险问答评测流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单评测集
  python -m m_eval.cli --model merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \\
      --eval-data data/eval/medical_qa_1000.jsonl \\
      --output reports/eval_report_dpo_v1.2

  # 多评测集目录
  python -m m_eval.cli --model merged_models/... \\
      --eval-data data/eval/ \\
      --output reports/eval_report_dpo_v1.2
        """,
    )
    parser.add_argument("--model", required=True, help="模型路径（HF 格式目录）")
    parser.add_argument("--eval-data", required=True, help="评测数据集路径（.jsonl 文件或目录）")
    parser.add_argument("--output", default="reports/eval_report", help="报告输出基名（不含扩展名）")
    parser.add_argument("--backend", choices=["vllm", "xinference"], default="vllm")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)

    # vLLM 参数
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)

    # xinference 参数
    parser.add_argument("--xinference-endpoint", default="http://127.0.0.1:9997")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    backend_kwargs = {}
    if args.backend == "vllm":
        backend_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
        backend_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    elif args.backend == "xinference":
        backend_kwargs["server_endpoint"] = args.xinference_endpoint

    ret = run_eval(
        model_path=args.model,
        eval_data=args.eval_data,
        output_base=args.output,
        backend_name=args.backend,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        **backend_kwargs,
    )

    sys.exit(ret)


if __name__ == "__main__":
    main()
