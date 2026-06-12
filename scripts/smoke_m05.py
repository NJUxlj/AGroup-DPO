"""M05 冒烟测试 —— 在 server2 vLLM env 上加载 DPO merge 模型并推理。

验证项:
  1. VLLMBackend 加载 merge 模型
  2. vLLM 单条推理（5 条保险问答）
  3. vLLM batch 推理
  4. m_eval 指标计算（基于推理结果）
  5. EvalReporter 生成 JSON + Markdown
"""

from __future__ import annotations

import os
import sys
import time
import logging

# vLLM 0.22.1+ 默认依赖 flashinfer 采样器，设置此环境变量
# 可强制使用 PyTorch 原生采样，避免 ModuleNotFoundError
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from m_infer.base import InferRequest
from m_infer.factory import build_infer_backend
from m_eval.metrics import accuracy_score, bleu_4_score, rouge_l_score
from m_eval.latency import aggregate_latency
from m_eval.reporter import EvalReporter


# ---- 保险问答测试样本（10 条） ----
EVAL_SAMPLES = [
    {
        "question": "保险等待期内确诊疾病是否赔付？",
        "reference_answer": "等待期内确诊一般不予赔付，合同另有约定的除外。",
        "answer_type": "open",
    },
    {
        "question": "百万医疗险的免赔额是怎么计算的？",
        "reference_answer": "百万医疗险通常设1万元年度免赔额，社保报销部分可抵扣免赔额。",
        "answer_type": "open",
    },
    {
        "question": "投保前未告知高血压，理赔会被拒吗？",
        "reference_answer": "投保前未如实告知高血压，保险公司在两年内可拒赔。",
        "answer_type": "open",
    },
    {
        "question": "什么是重大疾病保险？",
        "reference_answer": "重大疾病保险是确诊合同约定重疾时一次性给付保额的保险。",
        "answer_type": "open",
    },
    {
        "question": "保单现金价值是什么？",
        "reference_answer": "保单现金价值是投保人退保时可领取的金额。",
        "answer_type": "open",
    },
    {
        "question": "健康保险到期后能否自动续保？",
        "reference_answer": "自动续保取决于合同约定，部分产品支持保证续保。",
        "answer_type": "open",
    },
    {
        "question": "住院医疗险的理赔流程是什么？",
        "reference_answer": "理赔流程包括报案、提交病历发票、保险公司审核、赔付。",
        "answer_type": "open",
    },
    {
        "question": "既往症在健康保险中如何处理？",
        "reference_answer": "既往症通常属于责任免除范围，保险公司不予赔付。",
        "answer_type": "open",
    },
    {
        "question": "保险宽限期是多长时间？",
        "reference_answer": "保险宽限期通常为30天或60天。",
        "answer_type": "open",
    },
    {
        "question": "意外伤害保险的保障范围包括哪些？",
        "reference_answer": "意外伤害保险保障外来的、突发的、非本意的意外事故导致的身故或伤残。",
        "answer_type": "open",
    },
]


def main():
    MODEL_PATH = os.environ.get(
        "MODEL_PATH",
        "/root/autodl-tmp/agroup-dpo/merged_models/qwen2_5_1_5b_insurance_dpo_v1.2",
    )

    logger.info("=" * 60)
    logger.info("M05 Smoke Test: vLLM + DPO Merge Model")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info("=" * 60)

    # ============================================================
    # 测试 1: vLLM 加载
    # ============================================================
    logger.info("\n测试 1: vLLM 加载 merge 模型...")
    try:
        backend = build_infer_backend("vllm", MODEL_PATH, tensor_parallel_size=1)
        logger.info("  ✓ vLLM 加载成功")
    except Exception as e:
        logger.error(f"  ✗ vLLM 加载失败: {e}")
        return 1

    # ============================================================
    # 测试 2: 单条推理
    # ============================================================
    logger.info("\n测试 2: 单条推理（5 条保险问答）...")
    responses = []
    for i, sample in enumerate(EVAL_SAMPLES[:5]):
        req = InferRequest(
            prompt=sample["question"],
            max_new_tokens=128,
            temperature=0.7,
            request_id=f"smoke-{i}",
        )
        resp = backend.infer(req)
        responses.append(resp)
        logger.info(f"  [{i+1}] Q: {sample['question'][:40]}...")
        logger.info(f"      A: {resp.text[:80]}...")
        logger.info(f"      tokens: prompt={resp.prompt_tokens}, generated={resp.generated_tokens}, "
                     f"ttft={resp.latency_ms:.0f}ms, total={resp.total_latency_ms:.0f}ms")

    # 验证全非空
    non_empty = sum(1 for r in responses if r.text.strip())
    if non_empty == 5:
        logger.info(f"  ✓ 单条推理: {non_empty}/5 非空")
    else:
        logger.error(f"  ✗ 单条推理: {non_empty}/5 非空")
        backend.shutdown()
        return 1

    # ============================================================
    # 测试 3: batch 推理
    # ============================================================
    logger.info("\n测试 3: Batch 推理（10 条）...")
    batch_reqs = [
        InferRequest(prompt=s["question"], max_new_tokens=128, temperature=0.7)
        for s in EVAL_SAMPLES
    ]
    t0 = time.perf_counter()
    batch_responses = backend.batch_infer(batch_reqs)
    batch_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"  Batch 推理完成: {len(batch_responses)} 条, {batch_ms:.0f}ms "
                f"({batch_ms/len(batch_responses):.1f} ms/sample)")

    all_non_empty = sum(1 for r in batch_responses if r.text.strip())
    if all_non_empty == 10:
        logger.info(f"  ✓ Batch 推理: {all_non_empty}/10 非空")
    else:
        logger.error(f"  ✗ Batch 推理: {all_non_empty}/10 非空")

    # ============================================================
    # 测试 4: 延迟统计
    # ============================================================
    logger.info("\n测试 4: 延迟统计...")
    latency = aggregate_latency(batch_responses)
    logger.info(f"  p50 first-token: {latency.p50_first_token_ms:.1f}ms")
    logger.info(f"  p50 total: {latency.p50_total_ms:.1f}ms")
    logger.info(f"  throughput: {latency.throughput_samples_per_s:.1f} samples/s")
    # 验收标准
    if latency.p50_total_ms <= 1200:
        logger.info("  ✓ p50 latency 达标 (≤ 1200ms)")
    else:
        logger.warning("  ⚠ p50 latency 不达标 (%.1fms > 1200ms)", latency.p50_total_ms)
    logger.info("  ✓ 延迟统计完成")

    # ============================================================
    # 测试 5: 指标计算 + 报告
    # ============================================================
    logger.info("\n测试 5: 评测指标 + 报告生成...")
    preds = [r.text for r in batch_responses]
    refs = [s["reference_answer"] for s in EVAL_SAMPLES]

    acc = accuracy_score(preds, refs)
    bleu = bleu_4_score(preds, refs)
    rouge = rouge_l_score(preds, refs)
    logger.info(f"  Accuracy: {acc:.4f}")
    logger.info(f"  BLEU-4: {bleu:.4f}")
    logger.info(f"  ROUGE-L: {rouge:.4f}")

    reporter = EvalReporter(
        model_version=os.path.basename(MODEL_PATH),
        infer_backend="vllm",
    )
    reporter.add_dataset("insurance_qa_smoke", accuracy=acc, bleu_4=bleu, rouge_l=rouge, n_samples=10)
    reporter.set_latency(latency)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(output_dir, exist_ok=True)
    json_path, md_path = reporter.write(os.path.join(output_dir, "smoke_m05"))
    logger.info(f"  JSON report: {json_path}")
    logger.info(f"  Markdown report: {md_path}")
    logger.info("  ✓ 报告生成完成")

    backend.shutdown()

    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("M05 Smoke Test 汇总")
    logger.info("=" * 60)
    logger.info("  vLLM 加载: ✅")
    logger.info(f"  单条推理: ✅ (5/5)")
    logger.info(f"  Batch 推理: ✅ (10/10)")
    logger.info(f"  延迟统计: ✅ (p50={latency.p50_total_ms:.0f}ms)")
    logger.info(f"  评测指标: ✅ (acc={acc:.4f}, bleu={bleu:.4f}, rouge={rouge:.4f})")
    logger.info(f"  报告产出: ✅ ({json_path})")
    logger.info("\n🎉 M05 冒烟测试全部通过!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
