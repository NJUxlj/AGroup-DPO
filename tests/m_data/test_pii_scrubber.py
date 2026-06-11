"""测试 PII 脱敏器 (m_data/pii_scrubber.py)"""

import pytest

from m_data.pii_scrubber import PII_PATTERNS, PIIScrubber


class TestPIIScrubber:
    def test_scrub_id_card(self):
        scrubber = PIIScrubber()
        text = "张三的身份证号是110101199001011234。"
        result, hit = scrubber.scrub(text)
        assert hit is True
        assert "110101199001011234" not in result
        assert "[身份证号]" in result

    def test_scrub_phone(self):
        scrubber = PIIScrubber()
        text = "请联系13812345678获取详情。"
        result, hit = scrubber.scrub(text)
        assert hit is True
        assert "13812345678" not in result
        assert "[手机号]" in result

    def test_scrub_bank_card(self):
        scrubber = PIIScrubber()
        text = "卡号6222021234567890123已挂失。"
        result, hit = scrubber.scrub(text)
        assert hit is True
        assert "[银行卡号]" in result

    def test_scrub_email(self):
        scrubber = PIIScrubber()
        text = "请发邮件到zhangsan@insurance.com。"
        result, hit = scrubber.scrub(text)
        assert hit is True
        assert "zhangsan@insurance.com" not in result
        assert "[邮箱]" in result

    def test_no_pii(self):
        scrubber = PIIScrubber()
        text = "等待期内确诊一般不予赔付。"
        result, hit = scrubber.scrub(text)
        assert hit is False
        assert result == text

    def test_empty_text(self):
        scrubber = PIIScrubber()
        result, hit = scrubber.scrub("")
        assert hit is False
        assert result == ""

    def test_scrub_record(self):
        scrubber = PIIScrubber()
        record = {
            "question": "13812345678的保单怎么样？",
            "answer": "您的保单正常，请联系zhangsan@test.com。",
            "meta": "not a text field",
        }
        record, hit = scrubber.scrub_record(record, ["question", "answer"])
        assert hit is True
        assert "[手机号]" in record["question"]
        assert "[邮箱]" in record["answer"]

    def test_name_dict_scrub(self):
        scrubber = PIIScrubber(name_dict={"张三", "李四"})
        text = "张三的保单和李四的理赔。"
        result, hit = scrubber.scrub(text)
        assert hit is True
        assert "张三" not in result
        assert "李四" not in result
        assert "[姓名]" in result

    def test_scan(self):
        scrubber = PIIScrubber()
        text = "联系13812345678或zhangsan@test.com，身份证110101199001011234。"
        hits = scrubber.scan(text)
        assert "phone" in hits
        assert "email" in hits
        assert "id_card" in hits

    def test_custom_patterns(self):
        custom = [
            ("test", __import__("re").compile(r"TEST\d{3}"), "[TEST]"),
        ]
        scrubber = PIIScrubber(patterns=custom)
        result, hit = scrubber.scrub("码是TEST123请记下。")
        assert hit is True
        assert "[TEST]" in result
