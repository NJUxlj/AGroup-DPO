"""数据源抽象基类与原始记录定义。

M02 § 3.1: 所有数据源采集器必须实现 DataSource 接口。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator


@dataclass
class RawRecord:
    """原始记录，字段未定型，后续由 Normalizer / PIIScrubber 清洗。

    Attributes:
        source: 数据来源标识，如 policy_v1 / faq_v2 / ticket_v3。
        content: 原始内容字典，各数据源字段不同。
        raw_meta: 采集元信息（时间戳、路径等）。
        record_id: 记录唯一标识，由采集器分配。
    """

    source: str
    content: dict[str, Any]
    raw_meta: dict[str, Any] = field(default_factory=dict)
    record_id: str = ""


class DataSource(ABC):
    """所有数据源采集器的统一抽象接口。

    每个数据源实现 fetch() 方法，以迭代器方式返回 RawRecord。
    支持增量采集（since 参数）与数量限制（limit 参数）。
    """

    @abstractmethod
    def fetch(self, since: datetime | None = None, limit: int = 0) -> Iterator[RawRecord]:
        """从数据源获取记录。

        Args:
            since: 增量采集起点，None 表示全量。
            limit: 最大记录数，0 表示不限制。

        Yields:
            RawRecord: 原始记录。
        """
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        """数据源标识名，用于溯源与统计。"""
        ...

    def validate_connectivity(self) -> bool:
        """连通性自检，子类可重写。默认返回 True。"""
        return True
