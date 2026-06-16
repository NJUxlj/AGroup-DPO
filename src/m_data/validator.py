"""规则校验器。

M02 § 3.4: 6 类质量校验规则，确保产出数据集质量。
"""

import re
from typing import Any

from m_data.pii_patterns import PII_PATTERNS
REQUIRED_TERMS_FOR_CLAIMS = ["条款", "保单", "合同", "法", "约定"]

# 触发条款引用的业务关键词
# 依据 GB/T 36687-2018《保险术语》国家标准，覆盖承保、理赔、权益、费用、责任五大类
CLAIM_KEYWORDS = [
    # ── 承保类 ──
    "投保", "承保", "退保", "续保",
    # ── 理赔类 ──
    "理赔", "赔付", "拒赔", "给付", "赔偿",
    # ── 保单权益类 ──
    "等待期", "犹豫期", "宽限期", "现金价值", "受益人",
    # ── 费用与金额类 ──
    "保费", "保额", "保险金", "免赔",
    # ── 责任与告知类 ──
    "告知", "除外", "免责",
]

# PII 扫描模式（仅扫描，脱敏在 PIIScrubber 中完成）
# 从共享模块 PII_PATTERNS 中提取纯正则模式
_PII_SCAN_PATTERNS: list[re.Pattern] = [pat for _, pat, _ in PII_PATTERNS]


class Validator:
    """DPO 数据集规则校验器。

    对每条样本执行 6 类校验，返回 (passed, reason)。
    同时提供批量校验与统计功能。
    """

    def __init__(
        self,
        min_prompt_len: int = 5,
        max_prompt_len: int = 1024,
        min_response_len: int = 10,
        max_response_len: int = 2048,
        max_chosen_rejected_similarity: float = 0.95,
        required_terms: list[str] | None = None,
        claim_keywords: list[str] | None = None,
    ):
        self.min_prompt_len = min_prompt_len
        self.max_prompt_len = max_prompt_len
        self.min_response_len = min_response_len
        self.max_response_len = max_response_len
        self.max_similarity = max_chosen_rejected_similarity
        self.required_terms = required_terms or REQUIRED_TERMS_FOR_CLAIMS
        self.claim_keywords = claim_keywords or CLAIM_KEYWORDS

    def validate(self, sample: dict[str, Any]) -> tuple[bool, str]:
        """校验单条 DPO 样本。

        Args:
            sample: 含 prompt / chosen / rejected 字段的字典。

        Returns:
            (是否通过, 原因描述)。通过时原因为 "ok"。
        """
        # 1. 长度检查
        prompt = sample.get("prompt", "")
        if not (self.min_prompt_len <= len(prompt) <= self.max_prompt_len):
            return False, f"prompt length {len(prompt)} out of range"

        for field, label in [("chosen", "chosen"), ("rejected", "rejected")]:
            text = sample.get(field, "")
            if not (self.min_response_len <= len(text) <= self.max_response_len):
                return False, f"{label} length {len(text)} out of range"

        # 2. PII 检查
        for field in ("prompt", "chosen", "rejected"):
            text = sample.get(field, "")
            for pat in _PII_SCAN_PATTERNS:
                if pat.search(text):
                    return False, f"PII detected in {field}"

        # 3. 必引条款检查
        chosen = sample.get("chosen", "")
        if any(kw in chosen for kw in self.claim_keywords):
            if not any(t in chosen for t in self.required_terms):
                return False, "missing required policy reference"

        # 4. chosen ≠ rejected 相似度检查
        rejected = sample.get("rejected", "")
        sim = self._char_similarity(chosen, rejected)
        if sim > self.max_similarity:
            return False, f"chosen-rejected similarity {sim:.3f} > {self.max_similarity}"

        # 5. pii_scrubbed 标记检查
        if not sample.get("pii_scrubbed", False):
            return False, "pii_scrubbed flag not set"

        return True, "ok"

    def validate_batch(
        self, samples: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], bool, str]]:
        """批量校验，返回每条的 (sample, passed, reason)。"""
        return [(s, *self.validate(s)) for s in samples]

    def stats(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """统计校验结果。

        Returns:
            含 total / passed / failed / pass_rate / fail_reasons 的字典。
        """
        results = self.validate_batch(samples)
        total = len(results)
        passed = sum(1 for _, ok, _ in results if ok)
        failed = total - passed
        reasons: dict[str, int] = {}
        for _, ok, reason in results:
            if not ok:
                reasons[reason] = reasons.get(reason, 0) + 1

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "fail_reasons": reasons,
        }

    @staticmethod
    def _char_similarity(a: str, b: str) -> float:
        """字符级相似度（Jaccard 基于 3-gram）。"""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0

        def ngrams(s: str, n: int = 3) -> set[str]:
            return {s[i : i + n] for i in range(len(s) - n + 1)}

        set_a = ngrams(a)
        set_b = ngrams(b)
        if not set_a and not set_b:
            return 1.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union) if union else 0.0
