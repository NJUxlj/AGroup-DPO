"""xinference 推理后端 (FR-08 备选)

通过 HTTP 与本地/远程 xinference 服务通信。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import requests

from .base import InferBackend, InferRequest, InferResponse

logger = logging.getLogger(__name__)


class XinferenceBackend(InferBackend):
    """xinference 推理后端。

    通过 HTTP /v1/completions 与 xinference 服务通信。

    Usage:
        # 前提：xinference 服务已启动并注册了模型
        backend = XinferenceBackend()
        backend.load("merged_models/...", server_endpoint="http://127.0.0.1:9997")
        resp = backend.infer(InferRequest(prompt="你好"))
    """

    def __init__(self) -> None:
        self._endpoint: str = ""
        self._model_uid: str = ""

    # ---- InferBackend 接口实现 ----

    def load(self, model_path: str, **kwargs) -> None:
        """连接 xinference 服务并注册模型。

        Args:
            model_path: 模型路径（HF 格式目录）
            **kwargs:
                server_endpoint: xinference 服务地址（默认 http://127.0.0.1:9997）
                model_uid: 已注册的模型 UID（若提供则跳过注册步骤）
                model_name: 注册时的模型名（默认 "insurance-dpo"）
        """
        server_endpoint = kwargs.get("server_endpoint", "http://127.0.0.1:9997")
        self._endpoint = server_endpoint.rstrip("/")

        model_uid = kwargs.get("model_uid")
        if model_uid is not None:
            # 使用已注册的模型
            self._model_uid = model_uid
            logger.info(
                "xinference connected: endpoint=%s, model_uid=%s",
                self._endpoint, self._model_uid,
            )
            return

        # 注册新模型
        model_name = kwargs.get("model_name", "insurance-dpo")
        logger.info(
            "xinference registering model: path=%s, endpoint=%s",
            model_path, self._endpoint,
        )
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{self._endpoint}/v1/models",
                json={
                    "model_name": model_name,
                    "model_path": model_path,
                    "model_type": "LLM",
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            self._model_uid = data.get("model_uid", model_name)
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"xinference service unreachable at {self._endpoint}. "
                "Start it with: xinference-local -H 0.0.0.0 -p 9997"
            )
        load_ms = (time.perf_counter() - t0) * 1000
        logger.info("xinference model registered in %.0fms, uid=%s", load_ms, self._model_uid)

    def infer(self, req: InferRequest) -> InferResponse:
        """单条 xinference 推理。"""
        if not self._endpoint or not self._model_uid:
            raise RuntimeError("XinferenceBackend: not connected. Call load() first.")

        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{self._endpoint}/v1/completions",
                json={
                    "model": self._model_uid,
                    "prompt": req.prompt,
                    "max_tokens": req.max_new_tokens,
                    "temperature": req.temperature,
                    "top_p": req.top_p,
                    "stop": req.stop,
                },
                timeout=60,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"xinference inference failed: {e}")

        total_ms = (time.perf_counter() - t0) * 1000
        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})

        return InferResponse(
            text=choice.get("text", choice.get("message", {}).get("content", "")),
            prompt_tokens=usage.get("prompt_tokens", 0),
            generated_tokens=usage.get("completion_tokens", 0),
            latency_ms=data.get("first_token_latency_ms", 0.0),
            total_latency_ms=total_ms,
            request_id=req.request_id,
        )

    def batch_infer(self, reqs: list[InferRequest]) -> list[InferResponse]:
        """批量推理（逐条请求，xinference 暂不支持 native batching）。"""
        return [self.infer(r) for r in reqs]

    def shutdown(self) -> None:
        """断开 xinference 连接。"""
        self._model_uid = ""
        self._endpoint = ""
        logger.info("xinference backend shutdown")
