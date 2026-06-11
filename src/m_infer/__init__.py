"""M-INFER: 推理加速后端抽象层 (FR-08 上半)

提供 vLLM / xinference 双后端可切换封装，业务方通过 yaml 配置即可切换。
"""

from .base import InferBackend, InferRequest, InferResponse
from .factory import build_infer_backend
from .rag_handler import RAGRequest, RAGResponse, ContextDoc, create_rag_router
from .server import create_app

__all__ = [
    "InferBackend",
    "InferRequest",
    "InferResponse",
    "build_infer_backend",
    "RAGRequest",
    "RAGResponse",
    "ContextDoc",
    "create_rag_router",
    "create_app",
]
