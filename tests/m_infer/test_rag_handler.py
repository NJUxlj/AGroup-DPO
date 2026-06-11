"""RAG handler 单元测试"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from m_infer.base import InferBackend, InferRequest, InferResponse
from m_infer.rag_handler import (
    RAGRequest,
    RAGResponse,
    ContextDoc,
    create_rag_router,
    _extract_policy_refs,
)


# ---- Mock 推理后端 ----

class _MockBackend(InferBackend):
    """模拟推理后端，返回预定义答案。"""

    def __init__(self):
        self._loaded = False

    def load(self, model_path: str, **kwargs) -> None:
        self._loaded = True

    def infer(self, req: InferRequest) -> InferResponse:
        # 如果 prompt 中包含 policy_xxx 字样，模型也会在答案中复现
        return InferResponse(
            text=f"[Answer to: {req.prompt[:30]}...] 根据 policy_001 的规定，等待期内确诊一般不予赔付。",
            prompt_tokens=10,
            generated_tokens=20,
            latency_ms=100.0,
            total_latency_ms=500.0,
            request_id=req.request_id,
        )

    def shutdown(self) -> None:
        self._loaded = False


# ---- Fixtures ----

@pytest.fixture
def backend():
    return _MockBackend()


@pytest.fixture
def client(backend):
    """创建 TestClient，挂载 RAG 路由。"""
    app = FastAPI()
    router = create_rag_router(backend, model_version="mock_v1.0")
    app.include_router(router)
    return TestClient(app)


# ---- 1. Pydantic 模型序列化/反序列化 ----

class TestRAGModels:
    def test_rag_request_minimal(self):
        req = RAGRequest(user_query="什么是保险？")
        assert req.user_query == "什么是保险？"
        assert req.context_docs == []
        assert req.max_new_tokens == 512
        assert req.temperature == 0.3

    def test_rag_request_full(self):
        req = RAGRequest(
            user_query="等待期是否赔付？",
            context_docs=[
                ContextDoc(id="policy_001", text="等待期内确诊不予赔付"),
                ContextDoc(id="policy_002", text="合同另有约定的除外"),
            ],
            max_new_tokens=256,
            temperature=0.5,
        )
        assert len(req.context_docs) == 2
        assert req.max_new_tokens == 256
        assert req.temperature == 0.5

    def test_rag_request_validation(self):
        """max_new_tokens 必须在 1-4096 范围内。"""
        with pytest.raises(Exception):
            RAGRequest(user_query="test", max_new_tokens=0)
        with pytest.raises(Exception):
            RAGRequest(user_query="test", max_new_tokens=5000)
        with pytest.raises(Exception):
            RAGRequest(user_query="test", temperature=3.0)

    def test_rag_response_defaults(self):
        resp = RAGResponse(answer="test answer")
        assert resp.answer == "test answer"
        assert resp.policy_refs == []
        assert resp.first_token_latency_ms == 0.0
        assert resp.total_latency_ms == 0.0
        assert resp.model_version == ""
        assert resp.request_id == ""


# ---- 2. 策略引用提取 ----

class TestExtractPolicyRefs:
    def test_no_context_docs(self):
        refs = _extract_policy_refs("答案中提到了 policy_001", [])
        assert refs == []

    def test_policy_id_in_answer_and_context(self):
        docs = [ContextDoc(id="policy_001", text="等待期内确诊不予赔付")]
        refs = _extract_policy_refs("根据 policy_001 的规定，不予赔付。", docs)
        assert len(refs) == 1
        assert "policy_001" in refs[0]

    def test_policy_id_not_in_context(self):
        docs = [ContextDoc(id="policy_002", text="其他条款")]
        refs = _extract_policy_refs("根据 policy_001 的规定...", docs)
        assert refs == []

    def test_multiple_refs(self):
        docs = [
            ContextDoc(id="policy_001", text="条款1内容"),
            ContextDoc(id="policy_002", text="条款2内容"),
        ]
        refs = _extract_policy_refs("根据 policy_001 和 policy_002 的规定", docs)
        assert len(refs) == 2


# ---- 3. HTTP 端点集成测试 ----

class TestRAGEndpoint:
    def test_health_check(self, client):
        """健康检查不应走 RAG 路由，这里只验证 RAG 路由是否挂载。"""
        resp = client.post("/v1/insurance/qa", json={"user_query": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_simple_query(self, client):
        resp = client.post("/v1/insurance/qa", json={
            "user_query": "保险等待期是什么？",
            "max_new_tokens": 128,
            "temperature": 0.3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"].startswith("[Answer to:")
        assert data["model_version"] == "mock_v1.0"
        assert data["request_id"] != ""
        assert data["total_latency_ms"] > 0

    def test_query_with_context(self, client):
        resp = client.post("/v1/insurance/qa", json={
            "user_query": "等待期确诊是否赔付？",
            "context_docs": [
                {"id": "policy_001", "text": "等待期内确诊不予赔付。"}
            ],
            "max_new_tokens": 256,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "policy_001" in data["policy_refs"][0]

    def test_empty_context_docs(self, client):
        resp = client.post("/v1/insurance/qa", json={
            "user_query": "测试问题",
            "context_docs": [],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["policy_refs"] == []

    def test_invalid_request(self, client):
        """缺少必填字段 user_query 应返回 422。"""
        resp = client.post("/v1/insurance/qa", json={})
        assert resp.status_code == 422

    def test_response_schema(self, client):
        """验证响应包含所有必要字段。"""
        resp = client.post("/v1/insurance/qa", json={"user_query": "test"})
        assert resp.status_code == 200
        data = resp.json()
        for key in ["answer", "policy_refs", "first_token_latency_ms",
                     "total_latency_ms", "model_version", "request_id"]:
            assert key in data, f"Missing key: {key}"


# ---- 4. create_rag_router ----

class TestCreateRagRouter:
    def test_returns_router(self, backend):
        router = create_rag_router(backend, model_version="v1.0")
        assert router is not None
        # 路由应包含 POST /v1/insurance/qa
        routes = [r.path for r in router.routes]
        assert "/v1/insurance/qa" in routes
