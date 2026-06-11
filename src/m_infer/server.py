"""推理 HTTP 服务 (D-M05-05)

FastAPI 服务入口，暴露 /v1/insurance/qa 端点给司内 RAG 端调用。
支持：
  - 启动时自动加载 vLLM/xinference 后端
  - 健康检查 /health
  - 优雅关闭（释放 GPU 显存）
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from m_infer.factory import build_infer_backend
from m_infer.rag_handler import create_rag_router

logger = logging.getLogger(__name__)


# ---- 生命周期管理 ----

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期：启动加载模型，关闭释放资源。"""
    # 启动
    backend = app.state.backend
    logger.info("Infer backend ready: %s", type(backend).__name__)
    yield
    # 关闭
    logger.info("Shutting down infer backend...")
    backend.shutdown()
    logger.info("Infer backend shutdown complete")


# ---- 应用工厂 ----

def create_app(
    backend_name: str = "vllm",
    model_path: str = "",
    model_version: str = "",
    **backend_kwargs,
) -> FastAPI:
    """创建 FastAPI 应用实例。

    Args:
        backend_name: 推理后端名称（vllm / xinference）。
        model_path: 模型路径。
        model_version: 模型版本标识（写入响应）。
        **backend_kwargs: 传递给后端的额外参数（如 tensor_parallel_size）。

    Returns:
        配置完成的 FastAPI 实例。
    """
    # 加载推理后端
    logger.info("Loading infer backend: %s, model=%s", backend_name, model_path)
    backend = build_infer_backend(backend_name, model_path, **backend_kwargs)

    app = FastAPI(
        title="Ant-Group-DPO Insurance QA",
        description="保险问答推理服务 —— 与司内 RAG 端对接",
        version="1.0.0",
        lifespan=lifespan,
    )

    # 存到 app.state 供路由和生命周期使用
    app.state.backend = backend
    app.state.model_version = model_version or os.path.basename(model_path.rstrip("/"))

    # CORS（允许司内 RAG 端跨域访问）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 挂载 RAG 路由
    rag_router = create_rag_router(backend, model_version=app.state.model_version)
    app.include_router(rag_router)

    # 健康检查
    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "backend": backend_name,
            "model_version": app.state.model_version,
        }

    return app


# ---- CLI 入口 ----

def main():
    parser = argparse.ArgumentParser(description="Ant-Group-DPO Insurance QA Server")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "xinference"],
                        help="推理后端（默认 vllm）")
    parser.add_argument("--model-path", required=True,
                        help="模型路径（HF safetensors 目录）")
    parser.add_argument("--model-version", default="",
                        help="模型版本标识（默认取 model-path 最后一段）")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8080, help="监听端口（默认 8080）")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="张量并行数（默认 1）")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="GPU 显存利用率（默认 0.85）")
    parser.add_argument("--max-model-len", type=int, default=2048,
                        help="最大序列长度（默认 2048）")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # vLLM 0.22.1+ 需要此环境变量禁用 flashinfer 采样器
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    backend_kwargs = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
    }

    app = create_app(
        backend_name=args.backend,
        model_path=args.model_path,
        model_version=args.model_version,
        **backend_kwargs,
    )

    import uvicorn
    logger.info("Starting server on %s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
