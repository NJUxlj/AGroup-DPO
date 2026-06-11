"""文本规范化器。

M02 § 3.3: 去除多余空白、统一标点（半角/全角）、HTML tag 清除、长度截断。
"""

import re
from typing import Optional


class Normalizer:
    """文本规范化器，对采集到的原始文本做清洗。

    清洗步骤：
    1. HTML 标签移除
    2. 全角标点转半角
    3. 多余空白合并
    4. 首尾空白去除
    5. 长度截断（可选）
    """

    # 全角 → 半角映射
    _FULLWIDTH_MAP: dict[int, int] = {
        0xFF01: 0x0021,  # ！
        0xFF08: 0x0028,  # （
        0xFF09: 0x0029,  # ）
        0xFF0C: 0x002C,  # ，
        0xFF0E: 0x002E,  # ．
        0xFF1A: 0x003A,  # ：
        0xFF1B: 0x003B,  # ；
        0xFF1F: 0x003F,  # ？
    }

    _HTML_TAG_RE = re.compile(r"<[^>]+>")
    _MULTI_SPACE_RE = re.compile(r"\s+")
    _ELLIPSIS_RE = re.compile(r"\.{3,}")

    def __init__(self, max_length: int = 0):
        """
        Args:
            max_length: 文本最大长度（字符数），0 表示不截断。
        """
        self._max_length = max_length

    def normalize(self, text: str, max_length: Optional[int] = None) -> str:
        """对文本执行全部规范化步骤。

        Args:
            text: 原始文本。
            max_length: 覆盖实例级别的最大长度，None 表示使用实例默认值。

        Returns:
            规范化后的文本。
        """
        if not text:
            return ""

        # 1. 移除 HTML 标签
        text = self._HTML_TAG_RE.sub("", text)

        # 2. 全角标点转半角
        text = text.translate(self._FULLWIDTH_MAP)

        # 3. 连续空白合并为单个空格
        text = self._MULTI_SPACE_RE.sub(" ", text)

        # 4. 首尾空白去除
        text = text.strip()

        # 5. 连续句号合并
        text = self._ELLIPSIS_RE.sub("...", text)

        # 6. 长度截断
        limit = max_length if max_length is not None else self._max_length
        if limit > 0 and len(text) > limit:
            text = text[:limit]

        return text

    def normalize_record(self, record: dict, fields: list[str]) -> dict:
        """对记录中指定字段执行规范化。

        Args:
            record: 字段字典。
            fields: 需要规范化的字段名列表。

        Returns:
            规范化后的记录（原地修改）。
        """
        for field in fields:
            if field in record and isinstance(record[field], str):
                record[field] = self.normalize(record[field])
        return record
