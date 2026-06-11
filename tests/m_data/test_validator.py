"""测试规则校验器 (m_data/validator.py)"""

import pytest

from m_data.validator import Validator


def make_sample(
    prompt: str = "重疾险等待期内确诊是否赔付？",
    chosen: str = "等待期内确诊一般不予赔付，具体参见条款第5.2条。",
    rejected: str = "等待期内确诊也会赔付。",
    pii_scrubbed: bool = True,
    **kwargs,
) -> dict:
    sample = {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "pii_scrubbed": pii_scrubbed,
        "source": "test",
        "version": "dpo_v1.2",
    }
    sample.update(kwargs)
    return sample


class TestValidator:
    def test_valid_sample_passes(self):
        v = Validator()
        ok, reason = v.validate(make_sample())
        assert ok is True
        assert reason == "ok"

    def test_prompt_too_short(self):
        v = Validator(min_prompt_len=5)
        ok, reason = v.validate(make_sample(prompt="Hi"))
        assert ok is False
        assert "prompt length" in reason

    def test_prompt_too_long(self):
        v = Validator(max_prompt_len=10)
        ok, reason = v.validate(make_sample(prompt="A" * 20))
        assert ok is False
        assert "prompt length" in reason

    def test_chosen_too_short(self):
        v = Validator(min_response_len=10)
        ok, reason = v.validate(make_sample(chosen="短"))
        assert ok is False
        assert "chosen length" in reason

    def test_rejected_too_long(self):
        v = Validator(max_response_len=25)
        # chosen 必须在范围内（≥10 且 ≤25），rejected 超限
        ok, reason = v.validate(make_sample(chosen="合规答案依据条款第X条", rejected="B" * 30))
        assert ok is False
        assert "rejected length" in reason

    def test_pii_detected_in_prompt(self):
        v = Validator()
        ok, reason = v.validate(make_sample(prompt="13812345678的保单"))
        assert ok is False
        assert "PII detected" in reason

    def test_pii_detected_in_chosen(self):
        v = Validator()
        ok, reason = v.validate(make_sample(chosen="联系zhangsan@test.com"))
        assert ok is False
        assert "PII detected" in reason

    def test_missing_policy_reference(self):
        v = Validator()
        # chosen 含"赔付"但无条款引用词
        ok, reason = v.validate(
            make_sample(
                prompt="重疾险赔付流程？",
                chosen="赔付流程很简单，提交材料就行。",
            )
        )
        assert ok is False
        assert "missing required policy reference" in reason

    def test_policy_reference_present(self):
        v = Validator()
        # chosen 含"赔付"且有"条款"
        ok, reason = v.validate(
            make_sample(
                prompt="重疾险赔付流程？",
                chosen="根据条款第3条，赔付流程如下...",
            )
        )
        assert ok is True

    def test_chosen_rejected_too_similar(self):
        v = Validator(max_chosen_rejected_similarity=0.5)
        ok, reason = v.validate(
            make_sample(
                chosen="等待期内确诊一般不予赔付，具体参见条款。",
                rejected="等待期内确诊一般不予赔付，具体参见条款规定。",
            )
        )
        assert ok is False
        assert "similarity" in reason

    def test_chosen_rejected_different(self):
        v = Validator()
        ok, reason = v.validate(
            make_sample(
                chosen="等待期内确诊不予赔付，参见条款第5.2条。",
                rejected="等待期内确诊也会赔的，不用担心。",
            )
        )
        assert ok is True

    def test_pii_scrubbed_flag_not_set(self):
        v = Validator()
        ok, reason = v.validate(make_sample(pii_scrubbed=False))
        assert ok is False
        assert "pii_scrubbed" in reason

    def test_validate_batch(self):
        v = Validator()
        samples = [
            make_sample(),
            make_sample(prompt="短"),  # invalid
            make_sample(),
        ]
        results = v.validate_batch(samples)
        assert len(results) == 3
        assert results[0][1] is True
        assert results[1][1] is False
        assert results[2][1] is True

    def test_stats(self):
        v = Validator()
        samples = [
            make_sample(),
            make_sample(prompt="太短"),  # fail
            make_sample(),
            make_sample(chosen="短"),  # fail
        ]
        stats = v.stats(samples)
        assert stats["total"] == 4
        assert stats["passed"] == 2
        assert stats["failed"] == 2
        assert stats["pass_rate"] == 0.5
        assert len(stats["fail_reasons"]) == 2

    def test_no_claim_keywords_skips_reference_check(self):
        v = Validator()
        # 不含赔付/等待期/免赔/告知等关键词，不检查条款引用
        ok, reason = v.validate(
            make_sample(
                prompt="如何查询保单？",
                chosen="您可以登录APP查询保单信息。",
            )
        )
        assert ok is True

    def test_char_similarity_identical(self):
        sim = Validator._char_similarity("abc", "abc")
        assert sim == 1.0

    def test_char_similarity_different(self):
        sim = Validator._char_similarity("abc", "xyz")
        assert sim < 0.1

    def test_char_similarity_empty(self):
        sim = Validator._char_similarity("", "")
        assert sim == 1.0
        sim2 = Validator._char_similarity("abc", "")
        assert sim2 == 0.0
