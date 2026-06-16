"""测试 Chosen/Rejected 配对构造器 (m_data/pair_builder.py)"""

from unittest.mock import MagicMock, patch

import pytest

from m_data.pair_builder import PairBuilder
from llm.llm_provider import LLMProvider


class TestPairBuilder:
    def test_build_rule_based_yields_samples(self):
        builder = PairBuilder(enabled_strategies=["rule_based"])
        samples = list(builder.build_from_records([]))
        # 内置模板至少产出 5 条
        assert len(samples) >= 5
        # 检查 schema
        for s in samples:
            assert "prompt" in s
            assert "chosen" in s
            assert "rejected" in s
            assert "source" in s
            assert s["source"] == "rule_based"
            assert s["version"] == "dpo_v1.2"
            assert s["pii_scrubbed"] is True

    def test_build_rule_based_from_records(self):
        records = [
            {
                "question": "什么是重大疾病保险？",
                "answer": "重大疾病保险是指以特定重大疾病为给付条件的保险。",
                "source": "faq_v2",
            },
            {
                "question": "退保需要什么手续？",
                "answer": "退保需要携带身份证和保单原件到柜台办理。",
                "source": "faq_v2",
            },
        ]
        builder = PairBuilder(enabled_strategies=["rule_based"])
        samples = list(builder.build_from_records(records))
        # 内置 5 条 + records 2 条 = 7 条
        assert len(samples) >= 7

    def test_dedup_removes_duplicates(self):
        # 去重基于 (prompt, chosen, rejected) 三元组的 hash。
        # 验证：两次 build_from_records 之间不会产生重复（同一个 builder 实例）。
        builder = PairBuilder(enabled_strategies=["rule_based"])
        # 第一次 build 产生所有样本
        samples1 = list(builder.build_from_records([]))
        assert len(samples1) >= 5
        # 第二次 build（同样的空 records）时，模板样本全部被去重
        samples2 = list(builder.build_from_records([]))
        assert len(samples2) == 0

    def test_llm_judge_no_endpoint_skips(self):
        builder = PairBuilder(
            enabled_strategies=["llm_judge"],
            judge_endpoint=None,
        )
        records = [
            {
                "question": "测试问题",
                "answer": "专家答案",
                "rag_baseline_answer": "RAG答案",
            },
        ]
        samples = list(builder.build_from_records(records))
        # 无 endpoint 时跳过策略 B
        assert len(samples) == 0

    def test_retrieval_diff_no_endpoint_skips(self):
        builder = PairBuilder(
            enabled_strategies=["retrieval_diff"],
            rag_endpoint=None,
        )
        records = [{"question": "测试问题"}]
        samples = list(builder.build_from_records(records))
        assert len(samples) == 0

    def test_all_strategies_default(self):
        builder = PairBuilder()
        assert "rule_based" in builder._enabled
        assert "llm_judge" in builder._enabled
        assert "retrieval_diff" in builder._enabled

    def test_sample_schema_complete(self):
        builder = PairBuilder(enabled_strategies=["rule_based"])
        samples = list(builder.build_from_records([]))
        first = samples[0]
        assert isinstance(first["prompt"], str)
        assert isinstance(first["chosen"], str)
        assert isinstance(first["rejected"], str)
        assert first["chosen"] != first["rejected"]
        assert first["source"] == "rule_based"
        assert first["version"] == "dpo_v1.2"
        assert first["pii_scrubbed"] is True

    def test_dedup_key_stable(self):
        from m_data.pair_builder import PairBuilder as PB

        s1 = {"prompt": "a", "chosen": "b", "rejected": "c"}
        s2 = {"prompt": "a", "chosen": "b", "rejected": "c"}
        assert PB._dedup_key(s1) == PB._dedup_key(s2)

        s3 = {"prompt": "a", "chosen": "b", "rejected": "d"}
        assert PB._dedup_key(s1) != PB._dedup_key(s3)


# ------------------------------------------------------------------
# judge_provider (LLMProvider) 集成测试
# ------------------------------------------------------------------


class TestPairBuilderWithJudgeProvider:
    """测试 PairBuilder 通过 LLMProvider 使用第三方 API 的路径。"""

    @staticmethod
    def _make_mock_provider(winner_response: str = '{"winner": "A", "score_a": 0.9, "score_b": 0.3}'):
        """创建一个 mock LLMProvider，chat() 返回预设的 JSON。"""
        provider = MagicMock(spec=LLMProvider)
        provider.chat.return_value = winner_response
        provider.model = "test-judge"
        return provider

    def test_llm_judge_with_provider_yields_samples(self):
        """验证通过 LLMProvider 的 judge 路径生成样本。"""
        mock_provider = self._make_mock_provider()
        builder = PairBuilder(
            enabled_strategies=["llm_judge"],
            judge_provider=mock_provider,
        )
        records = [
            {
                "question": "测试问题",
                "answer": "专家答案",
                "rag_baseline_answer": "RAG答案",
            },
        ]
        samples = list(builder.build_from_records(records))
        # 应该产出 1 条样本（winner=A → expert 是 chosen）
        assert len(samples) == 1
        assert samples[0]["source"] == "llm_judge"
        assert samples[0]["judge_model"] == "test-judge"

    def test_llm_judge_with_provider_tie_skips(self):
        """验证 judge 判定 TIE 时跳过。"""
        mock_provider = self._make_mock_provider(
            '{"winner": "TIE", "score_a": 0.5, "score_b": 0.5}'
        )
        builder = PairBuilder(
            enabled_strategies=["llm_judge"],
            judge_provider=mock_provider,
        )
        records = [
            {
                "question": "测试问题",
                "answer": "专家答案",
                "rag_baseline_answer": "RAG答案",
            },
        ]
        samples = list(builder.build_from_records(records))
        assert len(samples) == 0

    def test_llm_judge_with_provider_winner_b(self):
        """验证 judge 判定 B 为 winner 时交换 chosen/rejected。"""
        mock_provider = self._make_mock_provider(
            '{"winner": "B", "score_a": 0.2, "score_b": 0.8}'
        )
        builder = PairBuilder(
            enabled_strategies=["llm_judge"],
            judge_provider=mock_provider,
        )
        records = [
            {
                "question": "测试问题",
                "answer": "专家答案",
                "rag_baseline_answer": "RAG答案很优秀",
            },
        ]
        samples = list(builder.build_from_records(records))
        assert len(samples) == 1
        # winner=B → RAG 答案（candidate_b）是 chosen
        assert samples[0]["chosen"] == "RAG答案很优秀"
        assert samples[0]["rejected"] == "专家答案"

    def test_llm_judge_with_provider_no_rag_answer_skips(self):
        """验证缺少 rag_baseline_answer 时跳过。"""
        mock_provider = self._make_mock_provider()
        builder = PairBuilder(
            enabled_strategies=["llm_judge"],
            judge_provider=mock_provider,
        )
        records = [
            {
                "question": "测试问题",
                "answer": "专家答案",
                # 缺少 rag_baseline_answer
            },
        ]
        samples = list(builder.build_from_records(records))
        assert len(samples) == 0

    def test_llm_judge_provider_error_returns_tie(self):
        """验证 LLMProvider 异常时优雅降级（不崩溃，返回 TIE 跳过）。"""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.chat.side_effect = Exception("API down")
        mock_provider.model = "test-judge"

        builder = PairBuilder(
            enabled_strategies=["llm_judge"],
            judge_provider=mock_provider,
        )
        records = [
            {
                "question": "测试问题",
                "answer": "专家答案",
                "rag_baseline_answer": "RAG答案",
            },
        ]
        samples = list(builder.build_from_records(records))
        # 异常时返回 TIE，跳过
        assert len(samples) == 0

    def test_llm_judge_provider_with_thinking_tags(self):
        """验证 LLMProvider 响应包含 thinking tags 时正确提取 JSON。"""
        mock_provider = MagicMock(spec=LLMProvider)
        # 模拟带 thinking tags 的响应
        mock_provider.chat.return_value = (
            '<thinking>需要分析两个答案的合规性</thinking>\n'
            '{"winner": "A", "score_a": 0.92, "score_b": 0.31}'
        )
        mock_provider.model = "test-judge"

        builder = PairBuilder(
            enabled_strategies=["llm_judge"],
            judge_provider=mock_provider,
        )
        records = [
            {
                "question": "测试问题",
                "answer": "专家答案非常合规",
                "rag_baseline_answer": "RAG答案不够好",
            },
        ]
        samples = list(builder.build_from_records(records))
        assert len(samples) == 1
        assert samples[0]["judge_score_chosen"] == 0.92

    def test_provider_and_endpoint_both_set_uses_provider(self):
        """验证同时设置 provider 和 endpoint 时优先使用 provider。"""
        mock_provider = self._make_mock_provider()
        builder = PairBuilder(
            enabled_strategies=["llm_judge"],
            judge_provider=mock_provider,
            judge_endpoint="http://old-endpoint:8000/v1",
        )
        records = [
            {
                "question": "测试",
                "answer": "答案A",
                "rag_baseline_answer": "答案B",
            },
        ]
        samples = list(builder.build_from_records(records))
        # provider 被调用（产出样本），endpoint 被忽略
        assert len(samples) == 1
        mock_provider.chat.assert_called()

    def test_no_provider_no_endpoint_skips(self):
        """验证既无 provider 也无 endpoint 时跳过策略 B。"""
        builder = PairBuilder(
            enabled_strategies=["llm_judge"],
            judge_provider=None,
            judge_endpoint=None,
        )
        records = [
            {
                "question": "测试",
                "answer": "答案A",
                "rag_baseline_answer": "答案B",
            },
        ]
        samples = list(builder.build_from_records(records))
        assert len(samples) == 0
