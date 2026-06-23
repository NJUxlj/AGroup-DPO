"""推理 HTTP 服务 (D-M05-05)

FastAPI 服务入口，暴露 /v1/insurance/qa 端点给司内 RAG 端调用。
支持：
  - 启动时自动加载 vLLM/xinference 后端
  - 通过 configs/infer.yaml 配置驱动后端切换
  - 健康检查 /health
  - 优雅关闭（释放 GPU 显存）
"""

from __future__ import annotations

import argparse
from utils.logger import CustomLogger
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from m_infer.config import load_infer_config, resolve_infer_settings
from m_infer.factory import build_infer_backend
from m_infer.rag_handler import create_rag_router

log = CustomLogger.get_logger(__name__)


# ---- 生命周期管理 ----

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期：启动加载模型，关闭释放资源。"""
    backend = app.state.backend
    log.info("Infer backend ready: %s", type(backend).__name__)
    yield
    log.info("Shutting down infer backend...")
    backend.shutdown()
    log.info("Infer backend shutdown complete")


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
        **backend_kwargs: 传递给后端的额外参数。

    Returns:
        配置完成的 FastAPI 实例。
    """
    log.info("Loading infer backend: %s, model=%s", backend_name, model_path)
    backend = build_infer_backend(backend_name, model_path, **backend_kwargs)

    app = FastAPI(
        title="AGroup-DPO Insurance QA",
        description="保险问答推理服务 —— 与司内 RAG 端对接",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.backend = backend
    app.state.model_version = model_version or os.path.basename(model_path.rstrip("/"))

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    rag_router = create_rag_router(backend, model_version=app.state.model_version)
    app.include_router(rag_router)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "backend": backend_name,
            "model_version": app.state.model_version,
        }

    return app


def run_server(
    *,
    backend_name: str,
    model_path: str,
    model_version: str = "",
    host: str = "0.0.0.0",
    port: int = 8080,
    backend_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    """启动 uvicorn HTTP 推理服务。"""
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    app = create_app(
        backend_name=backend_name,
        model_path=model_path,
        model_version=model_version,
        **(backend_kwargs or {}),
    )

    import uvicorn
    log.info("Starting server on %s:%s", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


# ---- CLI 入口 ----

def main():
    parser = argparse.ArgumentParser(description="AGroup-DPO Insurance QA Server")
    parser.add_argument("--config", "-c", default=None,
                        help="推理配置文件（configs/infer.yaml）")
    parser.add_argument("--backend", default=None, choices=["vllm", "xinference"],
                        help="推理后端（覆盖 config）")
    parser.add_argument("--model-path", default=None,
                        help="模型路径（HF safetensors 目录）")
    parser.add_argument("--model-version", default="",
                        help="模型版本标识（默认取 model-path 最后一段）")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8080, help="监听端口（默认 8080）")

    # vLLM 参数（CLI 覆盖 config）
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=2048)

    # xinference 参数
    parser.add_argument("--xinference-endpoint", default="http://127.0.0.1:9997")
    parser.add_argument("--xinference-model-uid", default=None)

    args = parser.parse_args()

    CustomLogger.configure(level="INFO")

    try:
        if args.config:
            cfg = load_infer_config(args.config)
            backend_name, model_path, backend_kwargs = resolve_infer_settings(
                cfg,
                backend=args.backend,
                model_path=args.model_path,
            )
        else:
            if not args.model_path:
                parser.error("--model-path is required when --config is not provided")
            backend_name = args.backend or "vllm"
            model_path = args.model_path
            backend_kwargs = {}

        if backend_name == "vllm":
            backend_kwargs.setdefault("tensor_parallel_size", args.tensor_parallel_size)
            backend_kwargs.setdefault("gpu_memory_utilization", args.gpu_memory_utilization)
            backend_kwargs.setdefault("max_model_len", args.max_model_len)
        elif backend_name == "xinference":
            backend_kwargs.setdefault("server_endpoint", args.xinference_endpoint)
            if args.xinference_model_uid:
                backend_kwargs["model_uid"] = args.xinference_model_uid

        run_server(
            backend_name=backend_name,
            model_path=model_path,
            model_version=args.model_version,
            host=args.host,
            port=args.port,
            backend_kwargs=backend_kwargs,
        )
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
