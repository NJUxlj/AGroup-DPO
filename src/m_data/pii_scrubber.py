"""PII 脱敏器。

M02 § 3.3: 正则 + 词典双重脱敏。
覆盖：身份证、手机号、银行卡号、邮箱、姓名（词典）。
"""

import re
from typing import Optional

from m_data.pii_patterns import PII_PATTERNS


class PIIScrubber:
    """PII（个人身份信息）脱敏器。

    支持两种脱敏策略：
    1. 正则模式匹配（身份证/手机号/银行卡/邮箱）
    2. 词典精确匹配（姓名等）
    """

    def __init__(
        self,
        patterns: Optional[list[tuple[str, re.Pattern, str]]] = None,
        name_dict: Optional[set[str]] = None,
    ):
        """
        Args:
            patterns: 自定义 PII 正则模式列表，格式 [(name, pattern, replacement), ...]。
            name_dict: 姓名词典（set），用于精确匹配替换。
        """
        self._patterns = patterns or PII_PATTERNS
        self._name_dict = name_dict or set()

    def scrub(self, text: str) -> tuple[str, bool]:
        """对文本执行 PII 脱敏。

        Args:
            text: 原始文本。

        Returns:
            (脱敏后文本, 是否命中任何 PII)。
        """
        if not text:
            return text, False

        result = text
        hit = False

        for _name, pattern, replacement in self._patterns:
            if pattern.search(result):
                result = pattern.sub(replacement, result)
                hit = True

        # 词典匹配（姓名）
        for name in self._name_dict:
            if name in result:
                result = result.replace(name, "[姓名]")
                hit = True

        return result, hit

    def scrub_record(self, record: dict, fields: list[str]) -> tuple[dict, bool]:
        """对记录中指定字段执行 PII 脱敏。

        Args:
            record: 字段字典。
            fields: 需要脱敏的字段名列表。

        Returns:
            (脱敏后记录, 是否命中任何 PII)。
        注意：此方法会原地修改 record。
        """
        any_hit = False
        for field in fields:
            if field in record and isinstance(record[field], str):
                scrubbed, hit = self.scrub(record[field])
                record[field] = scrubbed
                any_hit = any_hit or hit
        return record, any_hit

    def scan(self, text: str) -> list[str]:
        """扫描文本中的 PII，返回命中的模式名称列表。

        Args:
            text: 待扫描文本。

        Returns:
            命中的 PII 类型名称列表。
        """
        hits = []
        for name, pattern, _replacement in self._patterns:
            if pattern.search(text):
                hits.append(name)
        for name in self._name_dict:
            if name in text:
                hits.append("name")
        return hits
