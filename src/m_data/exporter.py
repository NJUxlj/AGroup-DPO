"""JSONL 导出器。

M02 § 3.5: 将 DPO/SFT 样本写出为 JSONL 格式。
支持文件路径与共享存储路径双写。
"""

import json
from utils.logger import CustomLogger
import os
from pathlib import Path
from typing import Any, Iterator

log = CustomLogger.get_logger(__name__)


class Exporter:
    """将数据集样本写出为 JSONL 文件。

    支持：
    - 单文件写出（本地路径）
    - 双路径写出（本地 + 共享存储同步）
    - 增量追加
    """

    def __init__(self, output_path: str, shared_path: str = ""):
        """
        Args:
            output_path: 主输出路径（JSONL 文件）。
            shared_path: 可选的共享存储路径（双写）。
        """
        self._output_path = Path(output_path)
        self._shared_path = Path(shared_path) if shared_path else None
        self._count = 0

    def write(self, samples: list[dict[str, Any]]) -> int:
        """批量写出样本，返回写出条数。"""
        written = 0
        try:
            with open(self._output_path, "a", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    written += 1
            self._count += written
            log.info("Exported %d samples to %s", written, self._output_path)
        except Exception as e:
            log.error("Failed to export to %s: %s", self._output_path, e)

        # 双写到共享存储
        if self._shared_path and written > 0:
            try:
                os.makedirs(self._shared_path.parent, exist_ok=True)
                with open(self._shared_path, "a", encoding="utf-8") as f:
                    for sample in samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                log.info("Synced %d samples to %s", written, self._shared_path)
            except Exception as e:
                log.warning("Failed to sync to shared path %s: %s", self._shared_path, e)

        return written

    def write_stream(self, samples: Iterator[dict[str, Any]], chunk_size: int = 1000) -> int:
        """流式写出，分批写入，返回总条数。"""
        total = 0
        buf: list[dict[str, Any]] = []
        for sample in samples:
            buf.append(sample)
            if len(buf) >= chunk_size:
                total += self.write(buf)
                buf.clear()
        if buf:
            total += self.write(buf)
        return total

    @property
    def total_written(self) -> int:
        """已写出总条数。"""
        return self._count

    @staticmethod
    def count_lines(path: str) -> int:
        """统计 JSONL 文件行数。"""
        p = Path(path)
        if not p.exists():
            return 0
        with open(p, encoding="utf-8") as f:
            return sum(1 for _ in f)

    @staticmethod
    def read_samples(path: str, limit: int = 0) -> list[dict[str, Any]]:
        """读取 JSONL 文件中的样本。

        Args:
            path: JSONL 文件路径。
            limit: 最大读取条数，0 表示全部。

        Returns:
            样本列表。
        """
        samples = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
                if 0 < limit <= i + 1:
                    break
        return samples
