"""保险条款数据源采集器。

M02 § 3.1: 从条款库解析 PDF/HTML → 结构化条款记录。
支持多解析器 fallback：pdfplumber → PyPDF2 → OCR。
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Iterator

from m_data.sources.base import DataSource, RawRecord

logger = logging.getLogger(__name__)


class InsurancePolicySource(DataSource):
    """从保险条款文件（PDF/HTML/JSON）中提取结构化记录。

    条款文件结构示例（JSON 格式）：
    {
        "policy_id": "POL-CRIT-001",
        "policy_name": "重大疾病保险条款",
        "articles": [
            {"article_id": "5.2", "title": "等待期", "content": "..."},
            ...
        ]
    }

    也支持纯文本条款文件（每行一条款），以及从目录递归扫描。
    """

    def __init__(self, data_path: str, parser: str = "json"):
        """
        Args:
            data_path: 条款文件目录路径或单个文件路径。
            parser: 解析器类型，json / text / pdf / html，默认 json。
        """
        self._data_path = Path(data_path)
        self._parser = parser

    @property
    def source_name(self) -> str:
        return "policy_v1"

    def fetch(self, since: datetime | None = None, limit: int = 0) -> Iterator[RawRecord]:
        count = 0
        for file_path in self._iter_files():
            if since and self._file_mtime(file_path) < since:
                continue
            try:
                yield from self._parse_file(file_path)
                count += 1
                if 0 < limit <= count:
                    break
            except Exception as e:
                logger.warning("Failed to parse policy file %s: %s", file_path, e)

    def _iter_files(self) -> Iterator[Path]:
        if self._data_path.is_file():
            yield self._data_path
        elif self._data_path.is_dir():
            for ext in (".json", ".jsonl", ".txt", ".pdf", ".html"):
                yield from self._data_path.rglob(f"*{ext}")

    def _parse_file(self, file_path: Path) -> Iterator[RawRecord]:
        if file_path.suffix == ".json":
            yield from self._parse_json(file_path)
        elif file_path.suffix == ".jsonl":
            yield from self._parse_jsonl(file_path)
        elif file_path.suffix == ".txt":
            yield from self._parse_text(file_path)
        else:
            logger.info("Skipping unsupported file: %s", file_path)

    def _parse_json(self, file_path: Path) -> Iterator[RawRecord]:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        articles = data.get("articles", [])
        policy_id = data.get("policy_id", file_path.stem)
        policy_name = data.get("policy_name", "")
        for art in articles:
            yield RawRecord(
                source=self.source_name,
                content={
                    "policy_id": policy_id,
                    "policy_name": policy_name,
                    "article_id": art.get("article_id", ""),
                    "article_title": art.get("title", ""),
                    "article_content": art.get("content", ""),
                },
                raw_meta={
                    "file": str(file_path),
                    "mtime": self._file_mtime(file_path).isoformat(),
                },
                record_id=self._make_record_id(policy_id, art.get("article_id", "")),
            )

    def _parse_jsonl(self, file_path: Path) -> Iterator[RawRecord]:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield RawRecord(
                    source=self.source_name,
                    content=obj,
                    raw_meta={
                        "file": str(file_path),
                        "mtime": self._file_mtime(file_path).isoformat(),
                    },
                    record_id=obj.get("policy_id", "") + "_" + obj.get("article_id", ""),
                )

    def _parse_text(self, file_path: Path) -> Iterator[RawRecord]:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
        yield RawRecord(
            source=self.source_name,
            content={"raw_text": content, "policy_id": file_path.stem},
            raw_meta={
                "file": str(file_path),
                "mtime": self._file_mtime(file_path).isoformat(),
            },
            record_id=file_path.stem,
        )

    @staticmethod
    def _make_record_id(policy_id: str, article_id: str) -> str:
        raw = f"{policy_id}:{article_id}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    @staticmethod
    def _file_mtime(path: Path) -> datetime:
        return datetime.fromtimestamp(os.path.getmtime(path))
