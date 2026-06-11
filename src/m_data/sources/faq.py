"""业务 FAQ 数据源采集器。

M02 § 3.1: 从司内 FAQ 库导出：分类 + 问题 + 答案。
支持 JSON/JSONL/CSV 格式的 FAQ 文件。
"""

import csv
import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Iterator

from m_data.sources.base import DataSource, RawRecord

logger = logging.getLogger(__name__)


class FAQSource(DataSource):
    """从 FAQ 知识库文件中提取问答对。

    支持格式：
    - JSON: [{"category": "...", "question": "...", "answer": "..."}]
    - JSONL: 每行一个 JSON 对象
    - CSV: 带 category/question/answer 列
    """

    def __init__(self, data_path: str):
        self._data_path = Path(data_path)

    @property
    def source_name(self) -> str:
        return "faq_v2"

    def fetch(self, since: datetime | None = None, limit: int = 0) -> Iterator[RawRecord]:
        count = 0
        for file_path in self._iter_files():
            if since and self._file_mtime(file_path) < since:
                continue
            try:
                for record in self._parse_file(file_path):
                    yield record
                    count += 1
                    if 0 < limit <= count:
                        return
            except Exception as e:
                logger.warning("Failed to parse FAQ file %s: %s", file_path, e)

    def _iter_files(self) -> Iterator[Path]:
        if self._data_path.is_file():
            yield self._data_path
        elif self._data_path.is_dir():
            for ext in (".json", ".jsonl", ".csv"):
                yield from self._data_path.rglob(f"*{ext}")

    def _parse_file(self, file_path: Path) -> Iterator[RawRecord]:
        if file_path.suffix == ".json":
            yield from self._parse_json(file_path)
        elif file_path.suffix == ".jsonl":
            yield from self._parse_jsonl(file_path)
        elif file_path.suffix == ".csv":
            yield from self._parse_csv(file_path)

    def _parse_json(self, file_path: Path) -> Iterator[RawRecord]:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get("faqs", [])
        for item in items:
            yield self._build_record(item, file_path)

    def _parse_jsonl(self, file_path: Path) -> Iterator[RawRecord]:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield self._build_record(json.loads(line), file_path)

    def _parse_csv(self, file_path: Path) -> Iterator[RawRecord]:
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield self._build_record(row, file_path)

    def _build_record(self, item: dict, file_path: Path) -> RawRecord:
        question = item.get("question", "")
        record_id = hashlib.md5(question.encode()).hexdigest()[:12]
        return RawRecord(
            source=self.source_name,
            content={
                "category": item.get("category", ""),
                "question": question,
                "answer": item.get("answer", ""),
            },
            raw_meta={
                "file": str(file_path),
                "mtime": self._file_mtime(file_path).isoformat(),
            },
            record_id=record_id,
        )

    @staticmethod
    def _file_mtime(path: Path) -> datetime:
        return datetime.fromtimestamp(os.path.getmtime(path))
