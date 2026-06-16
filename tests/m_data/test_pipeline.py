"""测试 DPO 数据流水线编排 (m_data/pipeline.py)

覆盖 Fix #1, #3, #4, #5, #6, #7 的所有新增功能。
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest

from m_data.pipeline import Pipeline
from m_data.sources.base import DataSource, RawRecord


# ------------------------------------------------------------------
# Mock 数据源
# ------------------------------------------------------------------


class MockDataSource(DataSource):
    """可注入测试数据的 Mock 数据源。"""

    def __init__(self, name: str, records_data: list[dict[str, Any]]):
        self._name = name
        self._records = records_data

    def fetch(
        self, since: datetime | None = None, limit: int = 0
    ) -> Iterator[RawRecord]:
        data = self._records
        if limit > 0:
            data = data[:limit]
        for i, d in enumerate(data):
            yield RawRecord(
                source=self._name,
                content=d,
                raw_meta={},
                record_id=f"{self._name}_{i}",
            )

    @property
    def source_name(self) -> str:
        return self._name


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------


def make_minimal_config(
    sources: list[dict] | None = None,
    quality: dict | None = None,
    output: dict | None = None,
    **kwargs,
) -> dict:
    """构造最小化流水线配置。"""
    cfg = {
        "seed": 42,
        "version": "test_v1.0",
        "sources": sources or {},
        "strategies": {
            "rule_based": {"enabled": True},
            "llm_judge": {"enabled": False},
            "retrieval_diff": {"enabled": False},
        },
        "quality": quality or {
            "min_prompt_len": 5,
            "max_prompt_len": 1024,
            "min_response_len": 10,
            "max_response_len": 2048,
            "max_chosen_rejected_similarity": 0.95,
        },
        **kwargs,
    }
    # 默认 output 路径放到临时目录
    if output is None and "output" not in kwargs:
        cfg["output"] = {}
    elif output is not None:
        cfg["output"] = output
    return cfg


def make_record(
    question: str = "测试问题",
    answer: str = "测试答案",
    source: str = "faq_v2",
    policy_id: str | None = None,
) -> dict:
    """构造一条模拟采集记录。"""
    rec = {
        "question": question,
        "answer": answer,
        "source": source,
    }
    if policy_id:
        rec["policy_id"] = policy_id
    return rec


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    """创建临时目录并在测试后清理。"""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def mock_source():
    """创建一个标准 Mock 数据源。"""
    records = [
        make_record(
            question="重疾险等待期内确诊是否赔付？",
            answer="等待期内确诊一般不予赔付，具体参见条款第5.2条。",
            source="faq_v2",
            policy_id="POL-CRIT-001",
        ),
        make_record(
            question="百万医疗险的免赔额计算方式？",
            answer="百万医疗险通常设置1万元年度免赔额，通过社保报销部分可抵扣。具体以保单条款为准。",
            source="faq_v2",
        ),
        make_record(
            question="投保前未告知高血压理赔会被拒吗？",
            answer="若未如实告知，保险公司有权据《保险法》第十六条解除合同或拒赔。但超过两年不可抗辩期且非故意则受保护。",
            source="ticket_v3",
        ),
    ]
    return MockDataSource("faq_v2", records)


# ------------------------------------------------------------------
# Fix #7: per-source limit
# ------------------------------------------------------------------


class TestPerSourceLimit:
    def test_collect_uses_per_source_limit(self, tmp_dir):
        """验证 collect 使用每个 source 自己的 limit。"""
        records = [make_record(question=f"问题{i}", answer=f"答案{i}") for i in range(10)]
        cfg = make_minimal_config(
            sources={
                "faq": {
                    "enabled": True,
                    "type": "faq",
                    "path": "dummy",
                    "limit": 3,  # 只取 3 条
                },
            },
        )
        pipeline = Pipeline(cfg)
        pipeline._sources = [MockDataSource("faq_v2", records)]
        pipeline._source_limits = [3]

        result = pipeline.collect()
        assert len(result) == 3

    def test_collect_limit_zero_means_all(self, tmp_dir):
        """验证 limit=0 表示不限制。"""
        records = [make_record(question=f"问题{i}", answer=f"答案{i}") for i in range(5)]
        cfg = make_minimal_config(
            sources={
                "faq": {
                    "enabled": True,
                    "type": "faq",
                    "path": "dummy",
                    "limit": 0,
                },
            },
        )
        pipeline = Pipeline(cfg)
        pipeline._sources = [MockDataSource("faq_v2", records)]
        pipeline._source_limits = [0]

        result = pipeline.collect()
        assert len(result) == 5


# ------------------------------------------------------------------
# Fix #5: Filter 敏感词过滤
# ------------------------------------------------------------------


class TestFilterSensitiveWords:
    def test_filter_removes_sensitive_word(self, tmp_dir):
        """验证敏感词过滤生效。"""
        cfg = make_minimal_config(
            quality={
                "min_prompt_len": 2,
                "min_response_len": 2,
                "sensitive_words": ["内部机密", "绝密"],
            },
        )
        pipeline = Pipeline(cfg)
        records = [
            make_record(question="正常问题", answer="正常答案"),
            make_record(question="正常问题2", answer="这是内部机密信息"),
            make_record(question="绝密资料", answer="正常答案"),
        ]
        result = pipeline.filter_records(records)
        # 只有第一条不含敏感词
        assert len(result) == 1
        assert result[0]["question"] == "正常问题"

    def test_filter_no_sensitive_words_configured(self, tmp_dir):
        """验证未配置敏感词时不过滤。"""
        cfg = make_minimal_config(
            quality={
                "min_prompt_len": 2,
                "min_response_len": 2,
            },
        )
        pipeline = Pipeline(cfg)
        records = [
            make_record(question="正常问题", answer="正常答案"),
            make_record(question="问题2", answer="答案2"),
        ]
        result = pipeline.filter_records(records)
        assert len(result) == 2


# ------------------------------------------------------------------
# Fix #6: prompt 全局去重
# ------------------------------------------------------------------


class TestDedupByPrompt:
    def test_dedup_removes_duplicate_prompts(self):
        """验证按 prompt 去重。"""
        samples = [
            {"prompt": "问题A", "chosen": "答案A1", "rejected": "坏答案A1"},
            {"prompt": "问题B", "chosen": "答案B1", "rejected": "坏答案B1"},
            {"prompt": "问题A", "chosen": "答案A2", "rejected": "坏答案A2"},  # 重复 prompt
        ]
        result = Pipeline._dedup_by_prompt(samples)
        assert len(result) == 2
        # 保留首次出现的
        prompts = [s["prompt"] for s in result]
        assert prompts == ["问题A", "问题B"]

    def test_dedup_empty_list(self):
        """空列表去重返回空列表。"""
        assert Pipeline._dedup_by_prompt([]) == []

    def test_dedup_single_item(self):
        """单条样本去重不变。"""
        samples = [{"prompt": "唯一问题", "chosen": "答案", "rejected": "坏答案"}]
        assert Pipeline._dedup_by_prompt(samples) == samples


# ------------------------------------------------------------------
# Fix #4: 必引条款回流修复
# ------------------------------------------------------------------


class TestRepairMissingReference:
    def test_repair_with_policy_id(self):
        """验证有 policy_id 时追加具体引用。"""
        sample = {
            "prompt": "赔付流程？",
            "chosen": "赔付流程很简单，提交材料就行。",
            "rejected": "不知道。",
            "policy_id": "POL-CRIT-001",
            "pii_scrubbed": True,
            "version": "dpo_v1.2",
        }
        fixed = Pipeline._repair_missing_reference(sample)
        assert fixed is not None
        assert "POL-CRIT-001" in fixed["chosen"]
        assert "相关条款及保单约定" in fixed["chosen"]

    def test_repair_without_policy_id(self):
        """验证无 policy_id 时追加通用引用。"""
        sample = {
            "prompt": "赔付流程？",
            "chosen": "赔付流程很简单。",
            "rejected": "不知道。",
            "policy_id": None,
            "pii_scrubbed": True,
            "version": "dpo_v1.2",
        }
        fixed = Pipeline._repair_missing_reference(sample)
        assert fixed is not None
        assert "相关保险条款及保单约定" in fixed["chosen"]

    def test_repair_no_duplicate_append(self):
        """验证不会重复追加引用语。"""
        sample = {
            "prompt": "赔付流程？",
            "chosen": "赔付流程很简单。 具体参见相关保险条款及保单约定。",
            "rejected": "不知道。",
            "policy_id": None,
            "pii_scrubbed": True,
            "version": "dpo_v1.2",
        }
        fixed = Pipeline._repair_missing_reference(sample)
        # 不应重复追加
        assert fixed["chosen"].count("相关保险条款及保单约定") == 1

    def test_repair_empty_chosen_returns_none(self):
        """空 chosen 返回 None。"""
        sample = {
            "prompt": "测试",
            "chosen": "",
            "rejected": "不知道。",
            "policy_id": "POL-001",
            "pii_scrubbed": True,
            "version": "dpo_v1.2",
        }
        assert Pipeline._repair_missing_reference(sample) is None


class TestValidateAndRepair:
    def test_validate_and_repair_fixes_missing_reference(self, tmp_dir):
        """端到端验证回流修复流程。"""
        cfg = make_minimal_config(
            quality={
                "min_prompt_len": 2,
                "max_prompt_len": 500,
                "min_response_len": 2,
                "max_response_len": 500,
                "max_chosen_rejected_similarity": 0.95,
            },
        )
        pipeline = Pipeline(cfg)
        pipeline.init_from_config()

        # 构造一个缺少条款引用但含"赔付"关键词的样本 + 一个正常样本
        samples = [
            {
                "prompt": "重疾险赔付流程是什么？",
                "chosen": "赔付流程很简单，提交材料就行。",  # 缺条款引用
                "rejected": "不知道赔付流程。",
                "source": "test",
                "policy_id": "POL-001",
                "pii_scrubbed": True,
                "version": "dpo_v1.2",
            },
            {
                "prompt": "如何查询保单？",
                "chosen": "您可以登录APP查询保单信息。",
                "rejected": "无法查询。",
                "source": "test",
                "pii_scrubbed": True,
                "version": "dpo_v1.2",
            },
        ]

        valid = pipeline._validate_and_repair(samples)
        # 两个都应该通过（第一个被修复）
        assert len(valid) == 2

    def test_validate_and_repair_all_valid(self, tmp_dir):
        """全部有效的样本直接通过。"""
        cfg = make_minimal_config(
            quality={
                "min_prompt_len": 2,
                "max_prompt_len": 500,
                "min_response_len": 2,
                "max_response_len": 500,
                "max_chosen_rejected_similarity": 0.95,
            },
        )
        pipeline = Pipeline(cfg)
        pipeline.init_from_config()

        samples = [
            {
                "prompt": "如何查询保单？",
                "chosen": "您可以登录APP查询保单信息。",
                "rejected": "无法查询。",
                "source": "test",
                "pii_scrubbed": True,
                "version": "dpo_v1.2",
            },
        ]
        valid = pipeline._validate_and_repair(samples)
        assert len(valid) == 1


# ------------------------------------------------------------------
# Fix #1: 评测留出集
# ------------------------------------------------------------------


class TestGenerateHoldoutSet:
    def test_generates_holdout_file(self, tmp_dir):
        """验证自动生成评测留出集。"""
        holdout_path = os.path.join(tmp_dir, "insurance_qa_500.jsonl")
        cfg = make_minimal_config(
            output={"eval_holdout_path": holdout_path},
            seed=42,
        )
        pipeline = Pipeline(cfg)

        # 构造 20 条 SFT 样本
        sft_samples = []
        for i in range(20):
            sft_samples.append({
                "instruction": "请回答用户的保险问题。",
                "input": f"问题{i}",
                "output": f"答案{i}",
                "source": "sft_s-a",
                "version": "sft_v1",
                "system": "你是AI财保助理，需严格依据条款作答。",
            })

        pipeline._generate_holdout_set(sft_samples, holdout_path)

        # 验证文件存在且有内容
        assert os.path.exists(holdout_path)
        with open(holdout_path, encoding="utf-8") as f:
            lines = f.readlines()
        # 10% 留出 = 约 2 条
        assert len(lines) >= 1
        for line in lines:
            sample = json.loads(line)
            assert "instruction" in sample
            assert "input" in sample
            assert "output" in sample

        # 验证 stats 被更新
        assert "holdout" in pipeline._stats
        assert pipeline._stats["holdout"]["count"] >= 1

    def test_holdout_does_not_overlap_with_none(self, tmp_dir):
        """验证留出集路径不存在时自动创建目录。"""
        holdout_path = os.path.join(tmp_dir, "subdir", "insurance_qa_500.jsonl")
        cfg = make_minimal_config(
            output={"eval_holdout_path": holdout_path},
            seed=42,
        )
        pipeline = Pipeline(cfg)
        sft_samples = [
            {
                "instruction": "请回答用户的保险问题。",
                "input": "测试问题",
                "output": "测试答案",
                "source": "sft_s-a",
                "version": "sft_v1",
                "system": "你是AI财保助理。",
            },
        ]
        pipeline._generate_holdout_set(sft_samples, holdout_path)
        assert os.path.exists(holdout_path)


# ------------------------------------------------------------------
# Fix #3: 数据质量报告
# ------------------------------------------------------------------


class TestGenerateQualityReport:
    def test_generates_report_file(self, tmp_dir):
        """验证数据质量报告自动生成。"""
        report_path = os.path.join(tmp_dir, "dpo_data_quality_v1.2.md")
        cfg = make_minimal_config(
            output={"report_path": report_path},
            version="dpo_v1.2",
        )
        pipeline = Pipeline(cfg)
        # 手动填充 stats
        pipeline._stats["collector"] = {
            "total": 100,
            "by_source": {"faq_v2": 60, "ticket_v3": 40},
        }
        pipeline._stats["scrubber"] = {"total": 100, "pii_hits": 5}
        pipeline._stats["validator"] = {
            "total": 80,
            "passed": 75,
            "failed": 5,
            "pass_rate": 0.9375,
            "repaired": 3,
        }
        pipeline._stats["holdout"] = {"path": "data/eval/insurance_qa_500.jsonl", "count": 50}

        dpo_raw = [{"prompt": "q", "chosen": "c", "rejected": "r"}] * 80
        dpo_valid = dpo_raw[:75]
        sft_raw = [{"instruction": "i", "input": "q", "output": "a"}] * 70
        sft_valid = sft_raw[:65]

        pipeline._generate_quality_report(
            report_path, dpo_raw, dpo_valid, sft_raw, sft_valid
        )

        assert os.path.exists(report_path)
        content = Path(report_path).read_text(encoding="utf-8")
        assert "# DPO 数据质量报告" in content
        assert "dpo_v1.2" in content
        assert "faq_v2" in content
        assert "PII 脱敏" in content
        assert "回流修复" in content
        assert "评测留出集" in content

    def test_report_creates_subdirectory(self, tmp_dir):
        """验证自动创建报告子目录。"""
        report_path = os.path.join(tmp_dir, "sub", "dir", "report.md")
        cfg = make_minimal_config(output={"report_path": report_path})
        pipeline = Pipeline(cfg)
        pipeline._generate_quality_report(report_path, [], [], [], [])
        assert os.path.exists(report_path)


# ------------------------------------------------------------------
# 端到端测试
# ------------------------------------------------------------------


class TestPipelineEndToEnd:
    def test_run_dry_mode_no_write(self, tmp_dir):
        """干跑模式不写文件。"""
        dpo_path = os.path.join(tmp_dir, "dpo.jsonl")
        sft_path = os.path.join(tmp_dir, "sft.jsonl")
        holdout_path = os.path.join(tmp_dir, "holdout.jsonl")
        report_path = os.path.join(tmp_dir, "report.md")

        cfg = make_minimal_config(
            sources={
                "faq": {
                    "enabled": True,
                    "type": "faq",
                    "path": "dummy",
                    "limit": 0,
                },
            },
            quality={
                "min_prompt_len": 2,
                "max_prompt_len": 500,
                "min_response_len": 2,
                "max_response_len": 500,
                "max_chosen_rejected_similarity": 0.95,
            },
            output={
                "path": dpo_path,
                "shared_path": "",
                "sft_path": sft_path,
                "shared_sft_path": "",
                "eval_holdout_path": holdout_path,
                "report_path": report_path,
            },
        )

        pipeline = Pipeline(cfg)
        pipeline.init_from_config()
        # 注入 mock source（覆盖 init_from_config 构建的 sources）
        records = [
            make_record(
                question="如何查询保单？",
                answer="您可以登录APP查询保单信息。",
                source="faq_v2",
            ),
        ]
        pipeline._sources = [MockDataSource("faq_v2", records)]
        pipeline._source_limits = [0]

        stats = pipeline.run(dry_run=True)

        # 验证统计
        assert stats["dpo_total"] >= 0
        assert stats["sft_total"] >= 0
        assert "elapsed_seconds" in stats

        # 验证未写文件
        assert not os.path.exists(dpo_path)
        assert not os.path.exists(sft_path)
        assert not os.path.exists(holdout_path)
        assert not os.path.exists(report_path)

    def test_run_full_pipeline_writes_output(self, tmp_dir):
        """全量运行写文件。"""
        dpo_path = os.path.join(tmp_dir, "dpo.jsonl")
        sft_path = os.path.join(tmp_dir, "sft.jsonl")
        holdout_path = os.path.join(tmp_dir, "holdout.jsonl")
        report_path = os.path.join(tmp_dir, "report.md")

        cfg = make_minimal_config(
            sources={
                "faq": {
                    "enabled": True,
                    "type": "faq",
                    "path": "dummy",
                    "limit": 0,
                },
            },
            quality={
                "min_prompt_len": 2,
                "max_prompt_len": 500,
                "min_response_len": 2,
                "max_response_len": 500,
                "max_chosen_rejected_similarity": 0.95,
            },
            output={
                "path": dpo_path,
                "shared_path": "",
                "sft_path": sft_path,
                "shared_sft_path": "",
                "eval_holdout_path": holdout_path,
                "report_path": report_path,
            },
        )

        pipeline = Pipeline(cfg)
        pipeline.init_from_config()
        records = [
            make_record(
                question="如何查询保单？",
                answer="您可以登录APP查询保单信息。具体参见条款第3条。",
                source="faq_v2",
            ),
            make_record(
                question="退保需要什么手续？",
                answer="退保需要携带身份证和保单原件到柜台办理。依据合同第7条规定。",
                source="faq_v2",
            ),
            make_record(
                question="重疾险等待期多久？",
                answer="等待期一般为90天至180天不等，以合同约定为准。详见条款。",
                source="faq_v2",
            ),
            make_record(
                question="如何修改受益人？",
                answer="您可以登录APP或致电客服修改受益人信息。",
                source="faq_v2",
            ),
        ]
        pipeline._sources = [MockDataSource("faq_v2", records)]
        pipeline._source_limits = [0]

        stats = pipeline.run(dry_run=False)

        # DPO 文件存在
        assert os.path.exists(dpo_path)
        dpo_content = Path(dpo_path).read_text(encoding="utf-8")
        dpo_lines = [l for l in dpo_content.split("\n") if l.strip()]
        assert len(dpo_lines) >= 1
        for line in dpo_lines:
            sample = json.loads(line)
            assert "prompt" in sample
            assert "chosen" in sample
            assert "rejected" in sample

        # SFT 文件存在
        assert os.path.exists(sft_path)
        sft_lines = Path(sft_path).read_text(encoding="utf-8").strip().split("\n")
        assert len(sft_lines) >= 1

        # 评测留出集存在
        assert os.path.exists(holdout_path)

        # 质量报告存在
        assert os.path.exists(report_path)
        report_content = Path(report_path).read_text(encoding="utf-8")
        assert "# DPO 数据质量报告" in report_content

        # 验证统计
        assert stats["dpo_total"] >= 1
        assert stats["sft_total"] >= 1
        assert "holdout" in stats

    def test_run_with_sensitive_words_filter(self, tmp_dir):
        """验证敏感词过滤在流水线中生效。"""
        dpo_path = os.path.join(tmp_dir, "dpo_sensitive.jsonl")
        sft_path = os.path.join(tmp_dir, "sft_sensitive.jsonl")
        holdout_path = os.path.join(tmp_dir, "holdout_sensitive.jsonl")
        report_path = os.path.join(tmp_dir, "report_sensitive.md")

        cfg = make_minimal_config(
            sources={
                "faq": {
                    "enabled": True,
                    "type": "faq",
                    "path": "dummy",
                    "limit": 0,
                },
            },
            quality={
                "min_prompt_len": 2,
                "max_prompt_len": 500,
                "min_response_len": 2,
                "max_response_len": 500,
                "max_chosen_rejected_similarity": 0.95,
                "sensitive_words": ["内部机密"],
            },
            output={
                "path": dpo_path,
                "shared_path": "",
                "sft_path": sft_path,
                "shared_sft_path": "",
                "eval_holdout_path": holdout_path,
                "report_path": report_path,
            },
        )

        pipeline = Pipeline(cfg)
        pipeline.init_from_config()
        records = [
            make_record(
                question="正常问题",
                answer="正常答案依据条款。",
                source="faq_v2",
            ),
            make_record(
                question="内部机密信息",
                answer="这是内部机密不应该出现在训练数据中。",
                source="faq_v2",
            ),
        ]
        pipeline._sources = [MockDataSource("faq_v2", records)]
        pipeline._source_limits = [0]

        stats = pipeline.run(dry_run=False)

        # 敏感词那条应该被过滤掉，只有正常的那条
        if os.path.exists(dpo_path):
            dpo_content = Path(dpo_path).read_text(encoding="utf-8")
            dpo_lines = [l for l in dpo_content.split("\n") if l.strip()]
            # 只有正常问题的样本
            for line in dpo_lines:
                sample = json.loads(line)
                assert "内部机密" not in sample["prompt"]

    def test_run_with_per_source_limit(self, tmp_dir):
        """验证 per-source limit 在流水线中生效。"""
        dpo_path = os.path.join(tmp_dir, "dpo_limit.jsonl")
        sft_path = os.path.join(tmp_dir, "sft_limit.jsonl")

        cfg = make_minimal_config(
            sources={
                "faq": {
                    "enabled": True,
                    "type": "faq",
                    "path": "dummy",
                    "limit": 2,  # 只取 2 条
                },
            },
            quality={
                "min_prompt_len": 2,
                "max_prompt_len": 500,
                "min_response_len": 2,
                "max_response_len": 500,
                "max_chosen_rejected_similarity": 0.95,
            },
            output={
                "path": dpo_path,
                "shared_path": "",
                "sft_path": sft_path,
                "shared_sft_path": "",
            },
        )

        # 准备 5 条记录但 limit=2
        records = [
            make_record(question=f"问题{i}", answer=f"答案{i}依据条款。", source="faq_v2")
            for i in range(5)
        ]
        mock_source = MockDataSource("faq_v2", records)

        # 使用 patch 让 _build_source 返回 mock，确保 limit 从 config 读取
        with patch.object(Pipeline, "_build_source", return_value=mock_source):
            pipeline = Pipeline(cfg)
            stats = pipeline.run(dry_run=True)

        # 只取了 2 条记录（limit=2）
        assert stats["collector"]["total"] == 2
        assert stats["collector"]["by_source"]["faq_v2"] == 2


# ------------------------------------------------------------------
# 回归测试：确保原有逻辑不受影响
# ------------------------------------------------------------------


class TestRegression:
    def test_pipeline_attributes_initialized(self):
        """验证 Pipeline 属性初始化正确。"""
        cfg = make_minimal_config()
        pipeline = Pipeline(cfg)
        assert pipeline._sources == []
        assert pipeline._source_limits == []
        assert pipeline._normalizer is None
        assert pipeline._scrubber is None
        assert pipeline._pair_builder is None
        assert pipeline._sft_builder is None
        assert pipeline._validator is None
        assert pipeline._dpo_exporter is None
        assert pipeline._sft_exporter is None

    def test_init_from_config_builds_components(self):
        """验证 init_from_config 正确构建组件。"""
        cfg = make_minimal_config(
            sources={
                "faq": {
                    "enabled": True,
                    "type": "faq",
                    "path": "dummy",
                    "limit": 100,
                },
            },
        )
        pipeline = Pipeline(cfg)
        # 注入 mock
        pipeline._sources = [MockDataSource("faq_v2", [make_record()])]
        pipeline._source_limits = [100]
        pipeline.init_from_config()

        assert pipeline._normalizer is not None
        assert pipeline._scrubber is not None
        assert pipeline._pair_builder is not None
        assert pipeline._sft_builder is not None
        assert pipeline._validator is not None
        assert pipeline._dpo_exporter is not None
        assert pipeline._sft_exporter is not None

    def test_validate_original_method_still_works(self):
        """验证原有的 validate 方法仍可用。"""
        cfg = make_minimal_config(
            quality={
                "min_prompt_len": 2,
                "max_prompt_len": 500,
                "min_response_len": 2,
                "max_response_len": 500,
                "max_chosen_rejected_similarity": 0.95,
            },
        )
        pipeline = Pipeline(cfg)
        pipeline.init_from_config()

        samples = [
            {
                "prompt": "如何查询保单？",
                "chosen": "您可以登录APP查询。",
                "rejected": "不知道。",
                "source": "test",
                "pii_scrubbed": True,
                "version": "dpo_v1.2",
            },
        ]
        valid = pipeline.validate(samples)
        assert len(valid) == 1
