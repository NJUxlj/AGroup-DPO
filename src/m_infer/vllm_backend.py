"""vLLM 推理后端 (FR-08 默认)

基于 vLLM PagedAttention + continuous batching，是推理主路径。
"""

from __future__ import annotations

from utils.logger import CustomLogger
import os
import time
from typing import Any, Optional

from .base import InferBackend, InferRequest, InferResponse

log = CustomLogger.get_logger(__name__)


class VLLMBackend(InferBackend):
    """vLLM 推理后端。

    封装 vLLM.LLM 的 load/generate，支持单卡和多卡 TP。

    Usage:
        backend = VLLMBackend()
        backend.load("merged_models/...", tensor_parallel_size=1)
        resp = backend.infer(InferRequest(prompt="你好"))
    """

    def __init__(self) -> None:
        self._llm: Any = None
        self._tokenizer: Any = None

    # ---- InferBackend 接口实现 ----

    def load(self, model_path: str, **kwargs) -> None:
        """加载 vLLM 模型。

        Args:
            model_path: 模型路径（HF 格式目录）
            **kwargs:
                tensor_parallel_size: 张量并行数（默认 1）
                gpu_memory_utilization: GPU 显存利用率（默认 0.85）
                max_model_len: 最大序列长度（默认 2048）
                enforce_eager: 是否禁用 CUDA graph（默认 False）
                dtype: 数据类型（默认 "bfloat16"）
        """
        # vLLM 0.22.1+ 默认依赖 flashinfer 采样器，设置此环境变量
        # 可强制使用 PyTorch 原生采样，避免 ModuleNotFoundError
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Install with: pip install vllm"
            )

        tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)
        gpu_memory_utilization = kwargs.get("gpu_memory_utilization", 0.85)
        max_model_len = kwargs.get("max_model_len", 2048)
        enforce_eager = kwargs.get("enforce_eager", False)
        dtype = kwargs.get("dtype", "bfloat16")

        log.info(
            "vLLM loading model: path=%s, tp=%s, max_len=%s, dtype=%s",
            model_path, tensor_parallel_size, max_model_len, dtype,
        )

        t0 = time.perf_counter()
        self._llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            dtype=dtype,
        )
        self._tokenizer = self._llm.get_tokenizer()
        load_ms = (time.perf_counter() - t0) * 1000
        log.info("vLLM model loaded in %.0fms", load_ms)

    def infer(self, req: InferRequest) -> InferResponse:
        """单条 vLLM 推理。"""
        if self._llm is None:
            raise RuntimeError("VLLMBackend: model not loaded. Call load() first.")

        from vllm import SamplingParams

        params = SamplingParams(
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_new_tokens,
            stop=req.stop,
        )

        t0 = time.perf_counter()
        outputs = self._llm.generate([req.prompt], params, use_tqdm=False)
        total_ms = (time.perf_counter() - t0) * 1000

        out = outputs[0]
        prompt_tokens = len(out.prompt_token_ids) if out.prompt_token_ids is not None else 0
        generated_tokens = (
            len(out.outputs[0].token_ids) if out.outputs[0].token_ids is not None else 0
        )

        first_token_ms = 0.0
        if out.metrics is not None:
            first_token_ms = getattr(out.metrics, "first_token_time", 0.0)
            if first_token_ms and first_token_ms > 0:
                first_token_ms *= 1000  # vLLM 0.22 returns seconds

        return InferResponse(
            text=out.outputs[0].text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            latency_ms=first_token_ms,
            total_latency_ms=total_ms,
            request_id=req.request_id,
        )

    def batch_infer(self, reqs: list[InferRequest]) -> list[InferResponse]:
        """批量推理，利用 vLLM continuous batching。"""
        if self._llm is None:
            raise RuntimeError("VLLMBackend: model not loaded. Call load() first.")

        if not reqs:
            return []

        from vllm import SamplingParams

        # 使用第一个请求的参数（批量推理要求统一 sampling params）
        first = reqs[0]
        params = SamplingParams(
            temperature=first.temperature,
            top_p=first.top_p,
            max_tokens=first.max_new_tokens,
            stop=first.stop,
        )

        prompts = [r.prompt for r in reqs]

        t0 = time.perf_counter()
        outputs = self._llm.generate(prompts, params, use_tqdm=False)
        total_ms = (time.perf_counter() - t0) * 1000

        results = []
        avg_per_sample = total_ms / len(reqs) if reqs else 0
        for i, out in enumerate(outputs):
            prompt_tokens = len(out.prompt_token_ids) if out.prompt_token_ids is not None else 0
            generated_tokens = (
                len(out.outputs[0].token_ids) if out.outputs[0].token_ids is not None else 0
            )
            first_token_ms = 0.0
            if out.metrics is not None:
                first_token_ms = getattr(out.metrics, "first_token_time", 0.0)
                if first_token_ms and first_token_ms > 0:
                    first_token_ms *= 1000

            results.append(InferResponse(
                text=out.outputs[0].text,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                latency_ms=first_token_ms,
                total_latency_ms=avg_per_sample,
                request_id=reqs[i].request_id,
            ))

        return results

    def shutdown(self) -> None:
        """释放 vLLM 模型资源。"""
        if self._llm is not None:
            del self._llm
            self._llm = None
            log.info("vLLM backend shutdown")
