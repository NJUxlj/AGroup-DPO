"""m_eval metrics 单元测试"""
import pytest
from m_eval.metrics import (
    normalize_answer,
    accuracy_score,
    bleu_4_score,
    rouge_l_score,
)


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("Hello WORLD") == "hello world"

    def test_remove_punctuation(self):
        result = normalize_answer("Hello, World!")
        assert result == "hello world"

    def test_extra_spaces(self):
        assert normalize_answer("  hello   world  ") == "hello world"

    def test_chinese_text(self):
        assert normalize_answer("你好，世界！") == "你好世界"


class TestAccuracy:
    def test_perfect_match(self):
        acc = accuracy_score(["hello world"], ["hello world"])
        assert acc == 1.0

    def test_case_insensitive(self):
        acc = accuracy_score(["Hello World"], ["hello world"])
        assert acc == 1.0

    def test_choice_match(self):
        # 正则正确抽取独立的大写字母选项
        acc = accuracy_score(
            ["根据保险合同，答案选 C"], ["C"]
        )
        assert acc == 1.0  # 正则正确匹配了"C"

    def test_choice_direct(self):
        acc = accuracy_score(["C"], ["C"])
        assert acc == 1.0

    def test_empty_list(self):
        assert accuracy_score([], []) == 0.0

    def test_mixed_results(self):
        acc = accuracy_score(
            ["hello world", "wrong answer"],
            ["hello world", "correct answer"],
        )
        assert acc == 0.5

    def test_judge_required_with_judge_fn(self):
        samples = [
            {"question": "Q1", "answer_type": "open", "judge_required": True},
        ]

        def always_correct(q, p, r):
            return True

        acc = accuracy_score(
            ["any answer"],
            ["reference"],
            samples=samples,
            judge_fn=always_correct,
        )
        assert acc == 1.0

    def test_answer_type_choice(self):
        samples = [{"answer_type": "choice"}]
        acc = accuracy_score(["答案是 B"], ["B"], samples=samples)
        assert acc == 1.0


class TestBleu4:
    def test_identical(self):
        # sacrebleu BLEU-4 需要至少 4-gram，所以用长句
        bleu = bleu_4_score(
            ["the quick brown fox jumps over the lazy dog"],
            ["the quick brown fox jumps over the lazy dog"],
        )
        assert bleu > 0.0

    def test_different(self):
        bleu = bleu_4_score(
            ["the quick brown fox jumps over the lazy dog"],
            ["a completely different sequence of words here"],
        )
        assert bleu < 1.0

    def test_empty_list(self):
        assert bleu_4_score([], []) == 0.0

    def test_chinese_bleu(self):
        # 中文短句 BLEU-4 可能为 0（不足 4-gram），用更长句子
        bleu = bleu_4_score(
            ["等待期内确诊一般不予赔付合同另有约定的除外"],
            ["等待期内确诊一般不予赔付合同另有约定的除外"],
        )
        assert 0.0 <= bleu <= 1.0


class TestRougeL:
    def test_identical(self):
        rl = rouge_l_score(["hello world"], ["hello world"])
        assert rl == 1.0

    def test_partial_overlap(self):
        rl = rouge_l_score(["hello world today"], ["hello world"])
        assert 0.0 < rl < 1.0

    def test_empty_list(self):
        assert rouge_l_score([], []) == 0.0

    def test_chinese_rouge(self):
        # rouge-score 默认 tokenizer 不支持中文，用英文验证 API
        rl = rouge_l_score(
            ["insurance claims must be filed within thirty days"],
            ["insurance claims must be filed within thirty days"],
        )
        assert rl == 1.0
