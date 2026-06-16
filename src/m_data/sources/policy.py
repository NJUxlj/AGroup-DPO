"""保险条款数据源采集器。

M02 § 3.1: 从条款库解析 PDF/HTML/JSON/JSONL/TXT → 结构化条款记录。
PDF 需要 PyPDF2 或 pdfplumber（可选依赖）；HTML 需要 beautifulsoup4（可选依赖）。
"""

import hashlib
import json
import logging
import os
import re
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
                    当文件扩展名已能确定类型时，该参数作为默认回退。
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
            for ext in (".json", ".jsonl", ".txt", ".pdf", ".html", ".htm"):
                yield from self._data_path.rglob(f"*{ext}")

    def _parse_file(self, file_path: Path) -> Iterator[RawRecord]:
        if file_path.suffix == ".json":
            # yield from 将子生成器（此处为 _parse_json）产出的每个值直接传递给当前生成器的调用者。
            # 其效果等同于：for record in self._parse_json(file_path): yield record
            yield from self._parse_json(file_path)
        elif file_path.suffix == ".jsonl":
            yield from self._parse_jsonl(file_path)
        elif file_path.suffix == ".txt":
            yield from self._parse_text(file_path)
        elif file_path.suffix == ".pdf":
            yield from self._parse_pdf(file_path)
        elif file_path.suffix == ".html" or file_path.suffix == ".htm":
            yield from self._parse_html(file_path)
        else:
            logger.warning("Unsupported file format, skipping: %s", file_path)

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

    def _parse_pdf(self, file_path: Path) -> Iterator[RawRecord]:
        """
        解析 PDF 条款文件，提取全文。

        优先使用 PyPDF2，其次 pdfplumber。均为可选依赖。
        每页作为一个独立 RawRecord 产出。
        """
        text_by_page: list[str] = []

        # 尝试 PyPDF2
        try:
            from PyPDF2 import PdfReader  # type: ignore[import-untyped]

            reader = PdfReader(str(file_path))
            text_by_page = [page.extract_text() or "" for page in reader.pages]
        except ImportError:
            pass

        # PyPDF2 失败则尝试 pdfplumber
        if not text_by_page:
            try:
                import pdfplumber  # type: ignore[import-untyped]

                with pdfplumber.open(str(file_path)) as pdf:
                    text_by_page = [page.extract_text() or "" for page in pdf.pages]
            except ImportError:
                logger.warning(
                    "PDF parsing requires PyPDF2 or pdfplumber. "
                    "Install with: pip install PyPDF2  (or pdfplumber). "
                    "Skipping: %s", file_path,
                )
                return
            except Exception as e:
                logger.warning("pdfplumber failed on %s: %s", file_path, e)
                return

        if not text_by_page:
            logger.warning("No extractable text in PDF: %s", file_path)
            return

        # 尝试从文本中识别 policy_id（如 "POL-XXX-001"）
        full_text = "\n".join(text_by_page)
        policy_id = self._guess_policy_id(full_text) or file_path.stem

        for page_num, text in enumerate(text_by_page, start=1):
            if not text.strip():
                continue
            yield RawRecord(
                source=self.source_name,
                content={
                    "raw_text": text,
                    "policy_id": policy_id,
                    "page": page_num,
                    "total_pages": len(text_by_page),
                },
                raw_meta={
                    "file": str(file_path),
                    "mtime": self._file_mtime(file_path).isoformat(),
                    "parser": "pdf",
                },
                record_id=self._make_record_id(policy_id, f"p{page_num}"),
            )

    def _parse_html(self, file_path: Path) -> Iterator[RawRecord]:
        """解析 HTML 条款文件，提取纯文本内容。

        优先使用 beautifulsoup4，否则使用简单正则去除标签。
        """
        raw_html = file_path.read_text(encoding="utf-8", errors="replace")

        # 尝试 BeautifulSoup
        try:
            from bs4 import BeautifulSoup  # type: ignore[import-untyped]

            soup = BeautifulSoup(raw_html, "html.parser")
            # 移除 script / style 标签
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
        except ImportError:
            # 回退：正则去除 HTML 标签
            text = re.sub(r"<[^>]+>", "", raw_html)
            text = re.sub(r"&[a-z]+;", " ", text)

        # 清洗：合并多余空白行
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if not text:
            logger.warning("No extractable text in HTML: %s", file_path)
            return

        policy_id = self._guess_policy_id(text) or file_path.stem

        # 尝试按 <h2>/<h3> 或「第X条」分段
        sections = re.split(r"\n(?=第[一二三四五六七八九十百千]+条|\d+\.\d+)", text)
        if len(sections) <= 1:
            # 无法分段，作为整体产出
            sections = [text]

        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
            # 取首行作为标题
            lines = section.split("\n", 1)
            title = lines[0].strip()[:80]
            content = lines[1].strip() if len(lines) > 1 else section

            yield RawRecord(
                source=self.source_name,
                content={
                    "article_title": title,
                    "article_content": content,
                    "policy_id": policy_id,
                    "section_index": i,
                },
                raw_meta={
                    "file": str(file_path),
                    "mtime": self._file_mtime(file_path).isoformat(),
                    "parser": "html",
                },
                record_id=self._make_record_id(policy_id, f"s{i}"),
            )

    @staticmethod
    def _guess_policy_id(text: str) -> str | None:
        """从文本内容中尝试提取 policy_id（如 POL-CRIT-001）。"""
        m = re.search(r"POL-[A-Z]+-\d+", text)
        return m.group(0) if m else None

    @staticmethod
    def _make_record_id(policy_id: str, article_id: str) -> str:
        raw = f"{policy_id}:{article_id}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    @staticmethod
    def _file_mtime(path: Path) -> datetime:
        return datetime.fromtimestamp(os.path.getmtime(path))
