"""SFT 数据集构造器。

M02 § 3.7: 从规范化后的记录构造 instruction-input-output 格式的 SFT 样本。
与 LLaMA-Factory alpaca 模板兼容。
"""

import hashlib
import logging
import random
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# 系统提示词（固定）
_SYSTEM_PROMPT = "你是蚂蚁保险的智能客服，需严格依据条款作答。"

# 指令前缀
_INSTRUCTION_PREFIX = "请回答用户的保险问题。"


class SFTBuilder:
    """从采集记录构造 SFT 格式样本。

    构造策略（M02 § 3.7）：
    - 策略 S-A：条款 + FAQ → instruction/input/output（占比 ≥ 50%）
    - 策略 S-B：历史工单 → 改写为 instruction 格式（占比 ≥ 30%）
    - 策略 S-C：专家标注补全（占比 ≥ 10%，长尾问题，通过外部注入）

    实际策略分配通过外部调度控制，本类提供统一构造方法。
    """

    def __init__(self, include_system: bool = True):
        """
        Args:
            include_system: 是否在样本中包含 system 字段。
        """
        self._include_system = include_system
        self._seen_ids: set[str] = set()

    def build_from_records(
        self, records: list[dict[str, Any]], strategy_label: str = "S-A"
    ) -> Iterator[dict[str, Any]]:
        """从记录列表构造 SFT 样本。

        Args:
            records: 规范化后的记录列表。
            strategy_label: 策略标签（S-A / S-B / S-C），用于溯源。

        Yields:
            SFT 样本字典，含 instruction/input/output/system/source/version。
        """
        for rec in records:
            sample = self._build_one(rec, strategy_label)
            if sample:
                yield sample

    def build_from_qa_pairs(
        self, qa_pairs: list[dict[str, str]], strategy_label: str = "S-C"
    ) -> Iterator[dict[str, Any]]:
        """从简单问答对列表构造 SFT 样本（用于策略 S-C 专家标注）。

        Args:
            qa_pairs: [{"question": "...", "answer": "..."}, ...]
            strategy_label: 策略标签。

        Yields:
            SFT 样本字典。
        """
        for pair in qa_pairs:
            question = pair.get("question", "")
            answer = pair.get("answer", "")
            if not question or not answer:
                continue
            yield self._build_one(
                {"question": question, "answer": answer}, strategy_label
            )

    def _build_one(
        self, rec: dict[str, Any], strategy_label: str
    ) -> dict[str, Any] | None:
        """从单条记录构造一个 SFT 样本。"""
        question = rec.get("question") or rec.get("user_question", "")
        answer = rec.get("answer") or rec.get("agent_answer", "")
        article_content = rec.get("article_content", "")

        if not question:
            return None

        # 若有条款内容，拼接到 answer 前面
        if article_content:
            answer = f"根据条款：{article_content}\n{answer}"

        # 去重
        dedup_key = hashlib.md5(f"{question}|{answer}".encode()).hexdigest()
        if dedup_key in self._seen_ids:
            return None
        self._seen_ids.add(dedup_key)

        sample: dict[str, Any] = {
            "instruction": _INSTRUCTION_PREFIX,
            "input": question,
            "output": answer,
            "source": f"sft_{strategy_label.lower()}",
            "version": "sft_v1",
        }

        if self._include_system:
            sample["system"] = _SYSTEM_PROMPT

        return sample

    def split_train_eval(
        self,
        samples: list[dict[str, Any]],
        eval_ratio: float = 0.1,
        seed: int = 42,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """将 SFT 样本拆分为训练集与评测留出集。

        Args:
            samples: 全量样本列表。
            eval_ratio: 留出比例，默认 10%。
            seed: 随机种子。

        Returns:
            (训练集, 评测留出集)。
        """
        rng = random.Random(seed)
        indices = list(range(len(samples)))
        rng.shuffle(indices)
        split = int(len(samples) * eval_ratio)
        eval_indices = set(indices[:split])
        train = [s for i, s in enumerate(samples) if i not in eval_indices]
        eval_set = [s for i, s in enumerate(samples) if i in eval_indices]
        return train, eval_set
