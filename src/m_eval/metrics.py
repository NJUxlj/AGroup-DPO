"""评测指标计算 (FR-08)

accuracy: 分类式问答用正则抽取 + 严格匹配，开放式用 LLM-as-Judge
bleu_4: sacrebleu corpus_bleu（tokenized 13a + brevity penalty）
rouge_l: rouge-score ROUGE-L（基于 LCS）
"""

from __future__ import annotations

import re


def normalize_answer(text: str) -> str:
    """标准化答案文本：去空格、转小写、去标点。"""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def accuracy_score(preds: list[str], refs: list[str]) -> float:
    """计算 Accuracy。

    分类式问答（如 ABCD 选项）：正则抽取第一个大写字母与 reference 严格匹配。
    开放式问答：标准化后严格匹配。

    Args:
        preds: 模型预测答案列表
        refs: 参考答案列表

    Returns:
        accuracy ∈ [0, 1]
    """
    if not preds:
        return 0.0

    correct = 0
    for pred, ref in zip(preds, refs):
        # 尝试抽取选择题答案（如 "A" 或 "A."）
        choice_match = re.search(r"\b([A-D])\b", pred.strip().upper())
        if choice_match:
            pred_choice = choice_match.group(1)
            ref_choice = ref.strip().upper().rstrip(".")
            if pred_choice == ref_choice:
                correct += 1
                continue

        # 开放式：标准化后严格匹配
        if normalize_answer(pred) == normalize_answer(ref):
            correct += 1

    return correct / len(preds)


def bleu_4_score(preds: list[str], refs: list[str]) -> float:
    """计算 corpus-level BLEU-4。

    使用 sacrebleu 保证跨环境可复现（tokenized 13a + brevity penalty）。
    返回值已归一化到 [0, 1]。

    Args:
        preds: 模型预测列表（每个元素是一条完整回答）
        refs: 参考答案列表

    Returns:
        BLEU-4 ∈ [0, 1]
    """
    import sacrebleu

    if not preds:
        return 0.0

    # sacrebleu 期望 refs 是 list[list[str]]
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return bleu.score / 100.0


def rouge_l_score(preds: list[str], refs: list[str]) -> float:
    """计算 ROUGE-L（所有样本均值）。

    ROUGE-L 基于最长公共子序列（LCS），对中文分词/字符级均可。

    Args:
        preds: 模型预测列表
        refs: 参考答案列表

    Returns:
        ROUGE-L ∈ [0, 1]
    """
    from rouge_score import rouge_scorer

    if not preds:
        return 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = []
    for pred, ref in zip(preds, refs):
        rs = scorer.score(ref, pred)["rougeL"].fmeasure
        scores.append(rs)

    return sum(scores) / len(scores)
