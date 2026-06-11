"""测试 Chosen/Rejected 配对构造器 (m_data/pair_builder.py)"""

import pytest

from m_data.pair_builder import PairBuilder


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
