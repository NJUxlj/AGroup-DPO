"""RAG 对接接口 (D-M05-06)

实现与司内普通 RAG 端的 HTTP 接口契约：
  POST /v1/insurance/qa

请求/响应 schema 严格对齐 M05 方案 § 3.5。
"""

from __future__ import annotations

from utils.logger import CustomLogger
import re
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from m_infer.base import InferBackend, InferRequest

log = CustomLogger.get_logger(__name__)


# ---- Pydantic 模型 ----

class ContextDoc(BaseModel):
    """RAG 端传入的上下文文档片段。"""
    id: str
    text: str


class RAGRequest(BaseModel):
    """POST /v1/insurance/qa 请求体（对齐 M05 § 3.5）。"""
    user_query: str = Field(..., description="用户问题")
    context_docs: list[ContextDoc] = Field(default_factory=list, description="RAG 检索的上下文文档")
    max_new_tokens: int = Field(default=512, ge=1, le=4096, description="最大生成长度")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="采样温度")


class RAGResponse(BaseModel):
    """POST /v1/insurance/qa 响应体（对齐 M05 § 3.5）。"""
    answer: str
    policy_refs: list[str] = Field(default_factory=list)
    first_token_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    model_version: str = ""
    request_id: str = ""


# ---- 策略引用提取 ----

_POLICY_REF_PATTERN = re.compile(r"(policy_\w+)")


def _extract_policy_refs(answer: str, context_docs: list[ContextDoc]) -> list[str]:
    """从答案中提取引用的策略文档 ID。

    策略：在 answer 中扫描 `policy_xxx` 字样，若该 ID 存在于 context_docs，
    则追加为引用。同时附加对应文档的首行摘要。
    """
    doc_map = {d.id: d.text for d in context_docs}
    mentioned_ids = set(_POLICY_REF_PATTERN.findall(answer))
    refs: list[str] = []
    for doc_id in mentioned_ids:
        if doc_id in doc_map:
            snippet = doc_map[doc_id][:60].replace("\n", " ").strip()
            refs.append(f"{doc_id} § {snippet}...")
    return refs


# ---- 路由构建 ----

def create_rag_router(
    backend: InferBackend,
    model_version: str = "",
) -> APIRouter:
    """创建 RAG 对接路由，绑定到给定的推理后端。

    Args:
        backend: 已加载模型的 InferBackend 实例。
        model_version: 模型版本标识（如 `qwen2_5_1_5b_insurance_dpo_v1.2`）。

    Returns:
        配置好的 fastapi.APIRouter。
    """
    router = APIRouter(tags=["insurance-qa"])

    @router.post("/v1/insurance/qa", response_model=RAGResponse)
    async def insurance_qa(
        req: RAGRequest,
        x_request_id: Optional[str] = Header(default=None, alias="X-Request-Id"),
    ):
        """保险问答接口 —— 司内 RAG 端的答案生成器。

        接收 RAG 检索到的上下文文档 + 用户问题，返回模型生成的答案。
        支持司内 RAG 端传入 X-Request-Id 请求头用于链路追踪（M05 § 3.5）。
        """
        request_id = x_request_id or str(uuid.uuid4())

        # 构造 prompt：若有上下文文档则拼接为 RAG 格式
        if req.context_docs:
            context_text = "\n\n".join(
                f"[{d.id}] {d.text}" for d in req.context_docs
            )
            prompt = (
                f"根据以下保险条款回答问题。\n\n"
                f"参考文档：\n{context_text}\n\n"
                f"问题：{req.user_query}\n"
                f"答案："
            )
        else:
            prompt = req.user_query

        infer_req = InferRequest(
            prompt=prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            request_id=request_id,
        )

        try:
            t0 = time.perf_counter()
            resp = backend.infer(infer_req)
            total_ms = (time.perf_counter() - t0) * 1000

            # 提取策略引用
            policy_refs = _extract_policy_refs(resp.text, req.context_docs)

            return RAGResponse(
                answer=resp.text,
                policy_refs=policy_refs,
                first_token_latency_ms=resp.latency_ms,
                total_latency_ms=total_ms,
                model_version=model_version,
                request_id=request_id,
            )
        except Exception as exc:
            log.error("insurance_qa inference failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    return router
