"""测试 JSONL 导出器 (m_data/exporter.py)"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from m_data.exporter import Exporter


class TestExporter:
    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.jsonl")
            exporter = Exporter(path)
            samples = [
                {"prompt": "问题1", "chosen": "答案A", "rejected": "答案B"},
                {"prompt": "问题2", "chosen": "答案C", "rejected": "答案D"},
            ]
            written = exporter.write(samples)
            assert written == 2
            assert exporter.total_written == 2
            assert Path(path).exists()

            # 验证内容
            read = Exporter.read_samples(path)
            assert len(read) == 2
            assert read[0]["prompt"] == "问题1"

    def test_write_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stream.jsonl")
            exporter = Exporter(path)

            def gen():
                for i in range(2500):
                    yield {"id": i, "text": f"sample_{i}"}

            total = exporter.write_stream(gen(), chunk_size=1000)
            assert total == 2500
            assert exporter.total_written == 2500

            # 验证行数
            count = Exporter.count_lines(path)
            assert count == 2500

    def test_count_lines_empty_file(self):
        assert Exporter.count_lines("/nonexistent/file.jsonl") == 0

    def test_read_samples_with_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "limit.jsonl")
            exporter = Exporter(path)
            samples = [{"id": i} for i in range(100)]
            exporter.write(samples)

            read = Exporter.read_samples(path, limit=10)
            assert len(read) == 10
            assert read[0]["id"] == 0

    def test_shared_path_sync(self):
        with tempfile.TemporaryDirectory() as tmp:
            main_path = os.path.join(tmp, "main.jsonl")
            shared_path = os.path.join(tmp, "shared", "sync.jsonl")
            exporter = Exporter(main_path, shared_path=shared_path)

            samples = [{"test": True}]
            exporter.write(samples)

            assert Path(main_path).exists()
            assert Path(shared_path).exists()

            shared_data = Exporter.read_samples(shared_path)
            assert len(shared_data) == 1
            assert shared_data[0]["test"] is True

    def test_jsonl_format_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "format.jsonl")
            exporter = Exporter(path)

            sample = {
                "prompt": "测试问题？",
                "chosen": "正确回答：引用条款。",
                "rejected": "错误回答。",
                "source": "test",
                "version": "dpo_v1.2",
            }
            exporter.write([sample])

            with open(path, encoding="utf-8") as f:
                line = f.readline().strip()
                parsed = json.loads(line)
                assert parsed == sample

    def test_unicode_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "unicode.jsonl")
            exporter = Exporter(path)

            sample = {
                "prompt": "重疾险等待期内确诊是否赔付？",
                "chosen": "根据《保险法》第十六条，等待期内确诊一般不予赔付。",
                "rejected": "会赔付的。",
            }
            exporter.write([sample])

            read = Exporter.read_samples(path)
            assert read[0]["prompt"] == sample["prompt"]
            assert "保险法" in read[0]["chosen"]
