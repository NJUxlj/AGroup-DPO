"""历史工单数据源采集器。

M02 § 3.1: 从工单系统导出"合规问答"子集。
支持 JSON/JSONL 格式的工单文件。
"""

import hashlib
import json
from utils.logger import CustomLogger
import os
from datetime import datetime
from pathlib import Path
from typing import Iterator

from m_data.sources.base import DataSource, RawRecord

log = CustomLogger.get_logger(__name__)


class TicketSource(DataSource):
    """
    从历史工单文件中提取合规问答对。

    工单记录结构：
    {
        "ticket_id": "TK-2026-0001",
        "category": "compliance_qa",
        "user_question": "...",
        "agent_answer": "...",
        "created_at": "2026-01-15T10:30:00"
    }

    仅提取 category 匹配 filter_category 的工单。
    """

    def __init__(self, data_path: str, filter_category: str = "compliance_qa"):
        """
        Args:
            data_path: 工单文件目录或单个文件路径。
            filter_category: 要提取的工单类别，默认 compliance_qa。
        """
        self._data_path = Path(data_path)
        self._filter_category = filter_category

    @property
    def source_name(self) -> str:
        return "ticket_v3"

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
                log.warning("Failed to parse ticket file %s: %s", file_path, e)

    def _iter_files(self) -> Iterator[Path]:
        if self._data_path.is_file():
            yield self._data_path
        elif self._data_path.is_dir():
            for ext in (".json", ".jsonl"):
                yield from self._data_path.rglob(f"*{ext}")

    def _parse_file(self, file_path: Path) -> Iterator[RawRecord]:
        if file_path.suffix == ".json":
            yield from self._parse_json(file_path)
        elif file_path.suffix == ".jsonl":
            yield from self._parse_jsonl(file_path)

    def _parse_json(self, file_path: Path) -> Iterator[RawRecord]:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        items = data if isinstance(data, list) else data.get("tickets", [])
        for item in items:
            record = self._build_record(item, file_path)
            if record is not None:
                yield record

    def _parse_jsonl(self, file_path: Path) -> Iterator[RawRecord]:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                record = self._build_record(item, file_path)
                if record is not None:
                    yield record

    def _build_record(self, item: dict, file_path: Path) -> RawRecord | None:
        category = item.get("category", "")
        if category != self._filter_category:
            return None
        ticket_id = item.get("ticket_id", "")
        record_id = hashlib.md5(ticket_id.encode()).hexdigest()[:12]
        return RawRecord(
            source=self.source_name,
            content={
                "ticket_id": ticket_id,
                "category": category,
                "user_question": item.get("user_question", ""),
                "agent_answer": item.get("agent_answer", ""),
                "created_at": item.get("created_at", ""),
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
