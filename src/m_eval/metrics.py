"""评测指标计算 (FR-08)

accuracy: 分类式问答用正则抽取 + 严格匹配，开放式 judge_required 时用 LLM-as-Judge
bleu_4: sacrebleu corpus_bleu（tokenized 13a + brevity penalty）
rouge_l: rouge-score ROUGE-L（基于 LCS）
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from m_infer.base import InferBackend


def normalize_answer(text: str) -> str:
    """标准化答案文本：去空格、转小写、去标点。"""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _is_choice_ref(ref: str) -> bool:
    """判断参考答案是否为选择题选项（A-D）。"""
    return ref.strip().upper().rstrip(".") in {"A", "B", "C", "D"}


def _match_choice(pred: str, ref: str) -> bool:
    """正则抽取第一个大写字母选项并与 reference 严格匹配。"""
    choice_match = re.search(r"\b([A-D])\b", pred.strip().upper())
    if not choice_match:
        return False
    return choice_match.group(1) == ref.strip().upper().rstrip(".")


def llm_judge_correct(
    question: str,
    pred: str,
    ref: str,
    backend: "InferBackend",
) -> bool:
    """LLM-as-Judge：判定开放式答案是否与参考答案语义一致（M05 § 3.6）。

    固定 temperature=0.0，输出 1（正确）或 0（错误）。
    """
    from m_infer.base import InferRequest

    prompt = (
        "你是医疗/保险问答评测员。给定问题、参考答案和模型答案，"
        "判断模型答案是否在语义上与参考答案一致。\n"
        "仅回复数字 1（正确）或 0（错误），不要输出其他内容。\n\n"
        f"问题：{question}\n"
        f"参考答案：{ref}\n"
        f"模型答案：{pred}\n\n"
        "正确(1)还是错误(0)："
    )
    resp = backend.infer(
        InferRequest(prompt=prompt, temperature=0.0, max_new_tokens=8)
    )
    text = resp.text.strip()
    return text.startswith("1") or text == "1"


def accuracy_score(
    preds: list[str],
    refs: list[str],
    samples: Optional[list[dict[str, Any]]] = None,
    judge_backend: Optional["InferBackend"] = None,
    judge_fn: Optional[Callable[[str, str, str], bool]] = None,
) -> float:
    """计算 Accuracy（M05 § 3.6 / § 3.9）。

    - answer_type=choice 或 ref 为 A-D：正则抽取 + 严格匹配
    - judge_required=true：LLM-as-Judge（需 judge_backend 或 judge_fn）
    - 其他开放式：标准化后严格匹配

    Args:
        preds: 模型预测答案列表
        refs: 参考答案列表
        samples: 可选评测样本元数据（含 question / answer_type / judge_required）
        judge_backend: 用于 LLM-as-Judge 的推理后端
        judge_fn: 自定义 judge 函数 (question, pred, ref) -> bool

    Returns:
        accuracy ∈ [0, 1]
    """
    if not preds:
        return 0.0

    correct = 0
    for i, (pred, ref) in enumerate(zip(preds, refs)):
        sample = samples[i] if samples and i < len(samples) else {}
        answer_type = sample.get("answer_type", "open")
        judge_required = sample.get("judge_required", False)
        question = sample.get("question", sample.get("prompt", ""))

        if answer_type == "choice" or _is_choice_ref(ref):
            if _match_choice(pred, ref):
                correct += 1
            continue

        if judge_required:
            if judge_fn is not None:
                if judge_fn(question, pred, ref):
                    correct += 1
            elif judge_backend is not None:
                if llm_judge_correct(question, pred, ref, judge_backend):
                    correct += 1
            elif normalize_answer(pred) == normalize_answer(ref):
                correct += 1
            continue

        if normalize_answer(pred) == normalize_answer(ref):
            correct += 1

    return correct / len(preds)


def bleu_4_score(preds: list[str], refs: list[str]) -> float:
    """计算 corpus-level BLEU-4。

    使用 sacrebleu 保证跨环境可复现（tokenized 13a + brevity penalty）。
    返回值已归一化到 [0, 1]。
    """
    import sacrebleu

    if not preds:
        return 0.0

    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return bleu.score / 100.0


def rouge_l_score(preds: list[str], refs: list[str]) -> float:
    """计算 ROUGE-L（所有样本均值）。

    ROUGE-L 基于最长公共子序列（LCS）。
    """
    from rouge_score import rouge_scorer

    if not preds:
        return 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(preds, refs):
        rs = scorer.score(ref, pred)["rougeL"].fmeasure
        scores.append(rs)

    return sum(scores) / len(scores)
