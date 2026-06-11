"""M05 Baseline 评测 + 对比报告生成 (D-M05-16)

步骤：
  1. 下载 Qwen2.5-1.5B-Instruct（baseline，未做 DPO）
  2. 在同一评测集上运行 vLLM 推理
  3. 与 DPO 模型 smoke_m05.json 对比
  4. 生成 reports/baseline_comparison.{json,md}
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from m_infer.base import InferRequest
from m_infer.factory import build_infer_backend
from m_eval.metrics import accuracy_score, bleu_4_score, rouge_l_score
from m_eval.latency import aggregate_latency
from m_eval.reporter import EvalReporter

# ---- 评测样本（与 smoke_m05.py 一致） ----
EVAL_SAMPLES = [
    {"question": "保险等待期内确诊疾病是否赔付？", "reference_answer": "等待期内确诊一般不予赔付，合同另有约定的除外。", "answer_type": "open"},
    {"question": "百万医疗险的免赔额是怎么计算的？", "reference_answer": "百万医疗险通常设1万元年度免赔额，社保报销部分可抵扣免赔额。", "answer_type": "open"},
    {"question": "投保前未告知高血压，理赔会被拒吗？", "reference_answer": "投保前未如实告知高血压，保险公司在两年内可拒赔。", "answer_type": "open"},
    {"question": "什么是重大疾病保险？", "reference_answer": "重大疾病保险是确诊合同约定重疾时一次性给付保额的保险。", "answer_type": "open"},
    {"question": "保单现金价值是什么？", "reference_answer": "保单现金价值是投保人退保时可领取的金额。", "answer_type": "open"},
    {"question": "健康保险到期后能否自动续保？", "reference_answer": "自动续保取决于合同约定，部分产品支持保证续保。", "answer_type": "open"},
    {"question": "住院医疗险的理赔流程是什么？", "reference_answer": "理赔流程包括报案、提交病历发票、保险公司审核、赔付。", "answer_type": "open"},
    {"question": "既往症在健康保险中如何处理？", "reference_answer": "既往症通常属于责任免除范围，保险公司不予赔付。", "answer_type": "open"},
    {"question": "保险宽限期是多长时间？", "reference_answer": "保险宽限期通常为30天或60天。", "answer_type": "open"},
    {"question": "意外伤害保险的保障范围包括哪些？", "reference_answer": "意外伤害保险保障外来的、突发的、非本意的意外事故导致的身故或伤残。", "answer_type": "open"},
]

BASELINE_MODEL = os.environ.get(
    "BASELINE_MODEL",
    "Qwen/Qwen2.5-1.5B-Instruct",
)
DPO_MODEL_PATH = os.environ.get(
    "DPO_MODEL_PATH",
    "/root/autodl-tmp/ant-group-dpo/merged_models/qwen2_5_1_5b_insurance_dpo_v1.2",
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


def evaluate_model(model_path: str, model_label: str):
    """加载模型并评测，返回 (accuracy, bleu, rouge, latency, reporter)。"""
    logger.info("=" * 60)
    logger.info("Evaluating: %s (%s)", model_label, model_path)
    logger.info("=" * 60)

    backend = build_infer_backend("vllm", model_path, tensor_parallel_size=1)

    batch_reqs = [
        InferRequest(prompt=s["question"], max_new_tokens=128, temperature=0.7)
        for s in EVAL_SAMPLES
    ]
    t0 = time.perf_counter()
    batch_responses = backend.batch_infer(batch_reqs)
    batch_ms = (time.perf_counter() - t0) * 1000
    logger.info("  Batch: %d samples, %.0fms (%.1f ms/sample)",
                len(batch_responses), batch_ms, batch_ms / len(batch_responses))

    preds = [r.text for r in batch_responses]
    refs = [s["reference_answer"] for s in EVAL_SAMPLES]

    acc = accuracy_score(preds, refs)
    bleu = bleu_4_score(preds, refs)
    rouge = rouge_l_score(preds, refs)
    latency = aggregate_latency(batch_responses)

    logger.info("  Accuracy: %.4f", acc)
    logger.info("  BLEU-4:   %.4f", bleu)
    logger.info("  ROUGE-L:  %.4f", rouge)
    logger.info("  p50 total: %.0fms", latency.p50_total_ms)

    reporter = EvalReporter(model_version=model_label, infer_backend="vllm")
    reporter.add_dataset("insurance_qa_smoke", accuracy=acc, bleu_4=bleu, rouge_l=rouge, n_samples=10)
    reporter.set_latency(latency)

    backend.shutdown()
    return acc, bleu, rouge, latency


def compute_relative_change(dpo_val: float, base_val: float) -> str:
    """计算相对变化百分比字符串。"""
    if base_val == 0:
        return "N/A (baseline=0)"
    delta = (dpo_val - base_val) / abs(base_val) * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- 1. 加载已有的 DPO 结果 ----
    dpo_json_path = os.path.join(OUTPUT_DIR, "smoke_m05.json")
    if not os.path.exists(dpo_json_path):
        logger.error("DPO result not found: %s. Run smoke_m05.py first.", dpo_json_path)
        return 1

    with open(dpo_json_path) as f:
        dpo_data = json.load(f)

    dpo_metrics = dpo_data["datasets"]["insurance_qa_smoke"]
    dpo_latency = dpo_data["latency"]

    # ---- 2. 评测 baseline 模型 ----
    baseline_acc, baseline_bleu, baseline_rouge, baseline_latency = evaluate_model(
        BASELINE_MODEL, model_label="Qwen2.5-1.5B-Instruct (no DPO)"
    )

    # ---- 3. 计算相对变化 ----
    acc_change = compute_relative_change(dpo_metrics["accuracy"], baseline_acc)
    bleu_change = compute_relative_change(dpo_metrics["bleu_4"], baseline_bleu)
    rouge_change = compute_relative_change(dpo_metrics["rouge_l"], baseline_rouge)
    latency_change = compute_relative_change(dpo_latency["p50_total_ms"], baseline_latency.p50_total_ms)

    # ---- 4. 生成对比报告 ----
    comparison = {
        "models": {
            "dpo": {
                "name": "qwen2_5_1_5b_insurance_dpo_v1.2",
                "accuracy": dpo_metrics["accuracy"],
                "bleu_4": dpo_metrics["bleu_4"],
                "rouge_l": dpo_metrics["rouge_l"],
                "p50_total_ms": dpo_latency["p50_total_ms"],
                "throughput": dpo_latency["throughput_samples_per_s"],
            },
            "baseline": {
                "name": "Qwen2.5-1.5B-Instruct (no DPO)",
                "accuracy": baseline_acc,
                "bleu_4": baseline_bleu,
                "rouge_l": baseline_rouge,
                "p50_total_ms": baseline_latency.p50_total_ms,
                "throughput": baseline_latency.throughput_samples_per_s,
            },
        },
        "relative_changes": {
            "accuracy": acc_change,
            "bleu_4": bleu_change,
            "rouge_l": rouge_change,
            "p50_total_ms": latency_change,
        },
        "n_samples": 10,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    json_path = os.path.join(OUTPUT_DIR, "baseline_comparison.json")
    with open(json_path, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    logger.info("JSON report: %s", json_path)

    # ---- Markdown 报告 ----
    d = comparison
    md_content = f"""# DPO vs Baseline 对比报告

> 生成时间：{d['generated_at']}
> 评测样本：10 条保险问答

## 模型

| 模型 | 说明 |
|------|------|
| **DPO** | `{d['models']['dpo']['name']}`（经过保险领域 DPO 对齐） |
| **Baseline** | `{d['models']['baseline']['name']}`（原始 Instruct，未做 DPO） |

## 指标对比

| 指标 | Baseline | DPO | 相对变化 |
|------|----------|-----|----------|
| Accuracy | {d['models']['baseline']['accuracy']:.4f} | {d['models']['dpo']['accuracy']:.4f} | {d['relative_changes']['accuracy']} |
| BLEU-4 | {d['models']['baseline']['bleu_4']:.4f} | {d['models']['dpo']['bleu_4']:.4f} | {d['relative_changes']['bleu_4']} |
| ROUGE-L | {d['models']['baseline']['rouge_l']:.4f} | {d['models']['dpo']['rouge_l']:.4f} | {d['relative_changes']['rouge_l']} |
| p50 延迟 | {d['models']['baseline']['p50_total_ms']:.0f}ms | {d['models']['dpo']['p50_total_ms']:.0f}ms | {d['relative_changes']['p50_total_ms']} |
| 吞吐 | {d['models']['baseline']['throughput']:.1f} samples/s | {d['models']['dpo']['throughput']:.1f} samples/s | — |

## 结论

本次对比基于 10 条保险问答样本，在相同硬件（RTX 5090 + vLLM 0.22.1）上评测。
DPO 对齐后的模型在保险领域问答的指标变化如上表所示。

**注意**：10 条样本的评测结果仅作趋势参考，全量对比需使用完整的 1000+ 条评测集。
"""

    md_path = os.path.join(OUTPUT_DIR, "baseline_comparison.md")
    with open(md_path, "w") as f:
        f.write(md_content)
    logger.info("Markdown report: %s", md_path)

    # ---- 汇总 ----
    logger.info("\n" + "=" * 60)
    logger.info("Baseline Comparison Summary")
    logger.info("=" * 60)
    logger.info("  Accuracy:  baseline=%.4f  DPO=%.4f  Δ=%s",
                baseline_acc, dpo_metrics["accuracy"], acc_change)
    logger.info("  BLEU-4:    baseline=%.4f  DPO=%.4f  Δ=%s",
                baseline_bleu, dpo_metrics["bleu_4"], bleu_change)
    logger.info("  ROUGE-L:   baseline=%.4f  DPO=%.4f  Δ=%s",
                baseline_rouge, dpo_metrics["rouge_l"], rouge_change)
    logger.info("  p50 total: baseline=%.0fms  DPO=%.0fms  Δ=%s",
                baseline_latency.p50_total_ms, dpo_latency["p50_total_ms"], latency_change)
    logger.info("\n  Reports: %s, %s", json_path, md_path)
    logger.info("  ✅ Baseline comparison complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
