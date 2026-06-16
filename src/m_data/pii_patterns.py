"""PII 正则模式定义（pii_scrubber 与 validator 共享）。

注意：不使用 \\b 词边界，因为中文文本中不会有 ASCII 词边界。
顺序很重要：bank_card 必须在 id_card 之前（19 位银行卡会被 18 位身份证模式误匹配）。
"""

import re

PII_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # 银行卡号：匹配 16-17 位或 19 位数字，前后使用负向环视确保不匹配更长数字串的一部分
    ("bank_card", re.compile(r"(?<!\d)\d{16,17}(?!\d)|(?<!\d)\d{19}(?!\d)"), "[银行卡号]"),
    # 身份证号：匹配 18 位身份证号（前 17 位为数字，第 18 位为数字或 X/x 校验码）
    ("id_card", re.compile(r"\d{17}[\dXx]"), "[身份证号]"),
    ("phone", re.compile(r"1[3-9]\d{9}"), "[手机号]"),
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[邮箱]"),
]
