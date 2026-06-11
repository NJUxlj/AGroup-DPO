"""
评测依赖单元测试 - M01 阶段交付物 D-M01-07
scripts/smoke_eval.py
验证 sacrebleu / rouge-score / nltk 在 5 条样本上的最小指标计算
"""
import sys
import json


def main():
    print("[smoke-eval] starting ...")

    # 1. 检查依赖
    print("[smoke-eval] [step1] checking deps ...")
    import sacrebleu
    from rouge_score import rouge_scorer
    print(f"[smoke-eval] sacrebleu={sacrebleu.__version__}, rouge_scorer OK")

    # 2. 5 条样本
    samples = [
        {"pred": "等待期内确诊一般不予赔付, 合同另有约定的除外。", "ref": "等待期内确诊通常不予赔付, 合同另有约定除外。"},
        {"pred": "百万医疗险通常设 1 万元年度免赔额。", "ref": "百万医疗险一般 1 万元年度免赔额。"},
        {"pred": "两年不可抗辩期内, 故意不如实告知可能被拒赔。", "ref": "未如实告知, 两年内保险公司可拒赔。"},
        {"pred": "重大疾病保险是确诊即给付保额。", "ref": "重疾险确诊合同约定重疾即给付保额。"},
        {"pred": "现金价值是退保时可领的金额。", "ref": "保单现金价值是退保时领取的金额。"},
    ]

    # 3. 计算 BLEU-4
    print("[smoke-eval] [step2] computing BLEU-4 ...")
    preds = [s["pred"] for s in samples]
    refs = [s["ref"] for s in samples]
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    print(f"[smoke-eval] BLEU-4: {bleu:.2f}")

    # 4. 计算 ROUGE-L
    print("[smoke-eval] [step3] computing ROUGE-L ...")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_scores = []
    for s in samples:
        rs = scorer.score(s["ref"], s["pred"])["rougeL"].fmeasure
        rouge_scores.append(rs)
    rouge_l = sum(rouge_scores) / len(rouge_scores)
    print(f"[smoke-eval] ROUGE-L: {rouge_l:.4f}")

    # 5. 计算 Accuracy (字符级严格匹配)
    print("[smoke-eval] [step4] computing accuracy ...")
    correct = sum(1 for s in samples if s["pred"] == s["ref"])
    accuracy = correct / len(samples)
    print(f"[smoke-eval] accuracy: {accuracy:.2f}")

    # 6. 报告
    report = {
        "n_samples": len(samples),
        "metrics": {
            "bleu_4": bleu / 100.0,
            "rouge_l": rouge_l,
            "accuracy": accuracy,
        },
    }
    print(f"\n[smoke-eval] report: {json.dumps(report, ensure_ascii=False, indent=2)}")

    # 7. 校验
    assert bleu >= 0, f"BLEU-4 invalid: {bleu}"
    assert 0 <= rouge_l <= 1, f"ROUGE-L invalid: {rouge_l}"
    assert 0 <= accuracy <= 1, f"accuracy invalid: {accuracy}"

    print("[smoke-eval] status: PASS (3/3 metrics produced)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[smoke-eval] status: FAIL - {e}")
        sys.exit(1)
