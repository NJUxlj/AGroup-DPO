"""推理后端抽象类 (FR-08)

定义 InferBackend ABC 及请求/响应数据类，屏蔽 vLLM / xinference 差异。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferRequest:
    """单条推理请求。

    Attributes:
        prompt: 输入文本
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度
        top_p: nucleus sampling 阈值
        stop: 停止词列表
        request_id: 可选请求 ID（用于追踪）
    """

    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[list[str]] = None
    request_id: Optional[str] = None


@dataclass
class InferResponse:
    """单条推理响应。

    Attributes:
        text: 生成的文本
        prompt_tokens: prompt token 数
        generated_tokens: 生成 token 数
        latency_ms: 首 token 时延（ms），不可用时为 0
        total_latency_ms: 全句时延（ms）
        request_id: 请求 ID
    """

    text: str
    prompt_tokens: int = 0
    generated_tokens: int = 0
    latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    request_id: Optional[str] = None


class InferBackend(ABC):
    """推理后端统一抽象。

    每个后端实现 load/infer/batch_infer/shutdown 四个方法。
    """

    @abstractmethod
    def load(self, model_path: str, **kwargs) -> None:
        """加载模型到 GPU/CPU。

        Args:
            model_path: 模型路径（HF 格式目录或模型名）
            **kwargs: 后端特定参数（如 tensor_parallel_size）
        """

    @abstractmethod
    def infer(self, req: InferRequest) -> InferResponse:
        """单条推理。

        Args:
            req: 推理请求

        Returns:
            InferResponse: 推理响应
        """

    def batch_infer(self, reqs: list[InferRequest]) -> list[InferResponse]:
        """批量推理。

        默认逐条调用 infer()，vLLM 后端可 override 利用 continuous batching。

        Args:
            reqs: 推理请求列表

        Returns:
            InferResponse 列表（顺序与输入一致）
        """
        return [self.infer(r) for r in reqs]

    @abstractmethod
    def shutdown(self) -> None:
        """释放模型资源。"""
