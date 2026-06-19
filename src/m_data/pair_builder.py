"""Chosen/Rejected 配对构造器。

M02 § 3.2: 针对保险业务场景，采用三种互补策略：
- 策略 A（rule_based）：基于业务规则的硬负例
- 策略 B（llm_judge）：基于 LLM-as-Judge 的软负例
- 策略 C（retrieval_diff）：基于检索召回差异

三种策略产生的配对最终合并去重。
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterator, Optional, TYPE_CHECKING

from utils.logger import CustomLogger

from m_data.templates import HARD_NEGATIVE_TEMPLATES

if TYPE_CHECKING:
    from llm.llm_provider import LLMProvider

log = CustomLogger.get_logger(__name__)

# 加载 LLM-as-Judge 提示词模板
_JUDGE_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "judge_pairwise.txt"
_JUDGE_TEMPLATE = _JUDGE_TEMPLATE_PATH.read_text(encoding="utf-8").strip()


class PairBuilder:
    """Chosen/Rejected 配对构造器。

    支持三种策略，通过 enabled_strategies 控制启用哪些。
    每种策略的产物通过 _dedup_key 去重。
    """

    def __init__(
        self,
        enabled_strategies: Optional[list[str]] = None,
        judge_model: str = "qwen2.5-7b-instruct",
        judge_endpoint: Optional[str] = None,
        judge_provider: Optional["LLMProvider"] = None,
        rag_endpoint: Optional[str] = None,
    ):
        """
        Args:
            enabled_strategies: 启用的策略列表，默认全部。
                ["rule_based", "llm_judge", "retrieval_diff"]
            judge_model: LLM-as-Judge 模型名。
            judge_endpoint: Judge 模型的 OpenAI 兼容 endpoint（旧方式，向后兼容）。
            judge_provider: LLMProvider 实例（支持第三方 API key）。
                若提供则优先使用，否则回退到 judge_endpoint 原始 HTTP 调用。
            rag_endpoint: 司内 RAG 端 endpoint。
        """
        self._enabled = enabled_strategies or ["rule_based", "llm_judge", "retrieval_diff"]
        self._judge_model = judge_model
        self._judge_endpoint = judge_endpoint
        self._judge_provider = judge_provider
        self._rag_endpoint = rag_endpoint
        self._seen_keys: set[str] = set()

    def build_from_records(
        self, records: list[dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        """从规范化后的记录列表构造 DPO 配对。

        Args:
            records: 规范化后的记录列表，每条记录含 question/answer 等字段。

        Yields:
            DPO 样本字典，含 prompt/chosen/rejected/source 等字段。
        """
        for strategy in self._enabled:
            if strategy == "rule_based":
                yield from self._build_rule_based(records)
            elif strategy == "llm_judge":
                yield from self._build_llm_judge(records)
            elif strategy == "retrieval_diff":
                yield from self._build_retrieval_diff(records)
            else:
                log.warning("Unknown strategy: %s, skipping", strategy)

    # ------------------------------------------------------------------
    # 策略 A: 基于业务规则的硬负例
    # ------------------------------------------------------------------

    def _build_rule_based(
        self, records: list[dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        """基于硬负例模板构造配对。

        从内置模板 + records 中匹配构造。
        """
        # 1. 先产出内置模板
        for tmpl in HARD_NEGATIVE_TEMPLATES:
            sample = self._make_dpo_sample(
                prompt=tmpl["prompt"],
                chosen=tmpl["chosen"],
                rejected=tmpl["rejected"],
                source="rule_based",
                policy_id=tmpl.get("policy_id"),
            )
            if sample and self._dedup(sample):
                yield sample

        # 2. 从 FAQ/工单 records 中抽取问答对，构造正例
        #    负例为简单否定/模糊回答
        for rec in records:
            question = rec.get("question") or rec.get("user_question", "")
            answer = rec.get("answer") or rec.get("agent_answer", "")
            if not question or not answer:
                continue

            rejected_variants = [
                "这个问题我不太清楚，建议您咨询客服。",
                "所有情况都会理赔，请放心。",
                "这个没有具体规定，视情况而定。",
                "公司会妥善处理的，不用担心。",
            ]
            rejected = random.choice(rejected_variants)

            sample = self._make_dpo_sample(
                prompt=question,
                chosen=answer,
                rejected=rejected,
                source="rule_based",
                policy_id=rec.get("policy_id"),
            )
            if sample and self._dedup(sample):
                yield sample

    # ------------------------------------------------------------------
    # 策略 B: 基于 LLM-as-Judge 的软负例
    # ------------------------------------------------------------------

    def _build_llm_judge(
        self, records: list[dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        """基于 LLM-as-Judge 构造配对。

        用 Qwen2.5-7B-Instruct 对比"RAG 端答案"与"专家答案"，
        输出 pairwise preference。

        优先使用 judge_provider（LLMProvider），回退到 judge_endpoint 原始调用。
        两者均不可用时跳过。
        """
        if not self._judge_provider and not self._judge_endpoint:
            log.info("LLM-as-Judge not configured (no provider or endpoint), skipping strategy B")
            return

        for rec in records:
            question = rec.get("question") or rec.get("user_question", "")
            expert_answer = rec.get("answer") or rec.get("agent_answer", "")
            rag_answer = rec.get("rag_baseline_answer", "")

            if not question or not expert_answer or not rag_answer:
                continue

            winner, score_chosen, score_rejected = self._call_judge(
                prompt=question,
                candidate_a=expert_answer,
                candidate_b=rag_answer,
            )

            if winner == "A":
                chosen, rejected = expert_answer, rag_answer
                chosen_score, rejected_score = score_chosen, score_rejected
            elif winner == "B":
                chosen, rejected = rag_answer, expert_answer
                chosen_score, rejected_score = score_rejected, score_chosen
            else:
                continue  # TIE, skip

            sample = self._make_dpo_sample(
                prompt=question,
                chosen=chosen,
                rejected=rejected,
                source="llm_judge",
                policy_id=rec.get("policy_id"),
                judge_model=self._judge_provider.model if self._judge_provider else self._judge_model,
                judge_score_chosen=chosen_score,
                judge_score_rejected=rejected_score,
            )
            if sample and self._dedup(sample):
                yield sample

    def _call_judge(
        self, prompt: str, candidate_a: str, candidate_b: str
    ) -> tuple[str, float, float]:
        """调用 Judge 模型获取 pairwise preference。

        优先使用 LLMProvider（支持第三方 API + 本地模型），
        回退到原始 HTTP 调用（仅本地无认证模型，向后兼容）。

        Returns:
            (winner, score_a, score_b) 其中 winner ∈ {"A", "B", "TIE"}。
        """
        judge_prompt = _JUDGE_TEMPLATE.format(
            prompt=prompt, candidate_a=candidate_a, candidate_b=candidate_b
        )

        # 通过 LLMProvider 调用（支持 API key + JSON 提取）
        if self._judge_provider:
            try:
                content = self._judge_provider.chat(
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=0.0,
                    max_tokens=512,
                    response_format={"type": "json_object"},
                )
                # 清理 thinking tags / code fences 等杂质
                from llm.llm_provider import LLMProvider

                content = LLMProvider.extract_json(content)
                result = json.loads(content)
                return (
                    result.get("winner", "TIE"),
                    float(result.get("score_a", 0.5)),
                    float(result.get("score_b", 0.5)),
                )
            except Exception as e:
                log.warning("Judge (LLMProvider) call failed: %s", e)
                return "TIE", 0.5, 0.5

        # 旧路径：原始 HTTP 调用（向后兼容，无 API key）
        try:
            import requests

            resp = requests.post(
                self._judge_endpoint,
                json={
                    "model": self._judge_model,
                    "messages": [{"role": "user", "content": judge_prompt}],
                    "temperature": 0.0,
                    "max_tokens": 256,
                },
                timeout=30,
            )
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            result = json.loads(content)
            return (
                result.get("winner", "TIE"),
                float(result.get("score_a", 0.5)),
                float(result.get("score_b", 0.5)),
            )
        except Exception as e:
            log.warning("Judge call failed: %s", e)
            return "TIE", 0.5, 0.5

    # ------------------------------------------------------------------
    # 策略 C: 基于检索召回差异
    # ------------------------------------------------------------------

    def _build_retrieval_diff(
        self, records: list[dict[str, Any]]
    ) -> Iterator[dict[str, Any]]:
        """基于检索召回差异构造配对。

        chosen = 完整索引 RAG 答案（经规则校验通过）
        rejected = 截断索引 RAG 答案

        需要 rag_endpoint 可用；若不可用则跳过。
        """
        if not self._rag_endpoint:
            log.info("RAG endpoint not configured, skipping strategy C")
            return

        for rec in records:
            question = rec.get("question") or rec.get("user_question", "")
            if not question:
                continue

            try:
                full_answer = self._call_rag(question, index_type="full")
                trunc_answer = self._call_rag(question, index_type="trunc")
            except Exception as e:
                log.warning("RAG call failed for '%s': %s", question[:30], e)
                continue

            if not full_answer or not trunc_answer:
                continue
            if full_answer == trunc_answer:
                continue

            sample = self._make_dpo_sample(
                prompt=question,
                chosen=full_answer,
                rejected=trunc_answer,
                source="retrieval_diff",
                policy_id=rec.get("policy_id"),
            )
            if sample and self._dedup(sample):
                yield sample

    def _call_rag(self, query: str, index_type: str = "full") -> str:
        """调用司内 RAG 端获取答案。

        Args:
            query: 用户问题。
            index_type: 索引类型，full 或 trunc。

        Returns:
            RAG 端返回的答案文本。
        """
        import requests

        resp = requests.post(
            self._rag_endpoint,
            json={
                "user_query": query,
                "context_docs": [],
                "index_type": index_type,
            },
            timeout=30,
        )
        data = resp.json()
        return data.get("answer", "")

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _make_dpo_sample(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        source: str,
        policy_id: Optional[str] = None,
        judge_model: Optional[str] = None,
        judge_score_chosen: Optional[float] = None,
        judge_score_rejected: Optional[float] = None,
    ) -> dict[str, Any]:
        """构造标准 DPO 样本字典（M02 § 7.1 Schema）。"""
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "source": source,
            "policy_id": policy_id,
            "judge_model": judge_model,
            "judge_score_chosen": judge_score_chosen,
            "judge_score_rejected": judge_score_rejected,
            "pii_scrubbed": True,
            "version": "dpo_v1.2",
        }

    def _dedup(self, sample: dict[str, Any]) -> bool:
        """去重检查，返回 False 表示重复已跳过。"""
        key = self._dedup_key(sample)
        if key in self._seen_keys:
            return False
        self._seen_keys.add(key)
        return True

    @staticmethod
    def _dedup_key(sample: dict[str, Any]) -> str:
        raw = f"{sample['prompt']}|{sample['chosen']}|{sample['rejected']}"
        return hashlib.md5(raw.encode()).hexdigest()
