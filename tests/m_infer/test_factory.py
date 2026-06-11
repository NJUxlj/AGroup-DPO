"""m_infer 工厂与 base 单元测试"""
import pytest
from m_infer.base import InferBackend, InferRequest, InferResponse
from m_infer.factory import build_infer_backend


class TestInferDataclasses:
    def test_infer_request_defaults(self):
        """验证 InferRequest 默认值"""
        req = InferRequest(prompt="hello")
        assert req.prompt == "hello"
        assert req.max_new_tokens == 512
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.stop is None

    def test_infer_response_defaults(self):
        """验证 InferResponse 默认值"""
        resp = InferResponse(text="hi")
        assert resp.text == "hi"
        assert resp.prompt_tokens == 0
        assert resp.generated_tokens == 0


class TestInferFactory:
    def test_build_vllm_registered(self):
        """验证 vllm 后端在注册表中且可被工厂识别（不实际加载模型）"""
        # 仅验证注册表中存在，不实际 import vllm（macOS 可能未安装）
        from m_infer.registry import INFER_REGISTRY
        assert "vllm" in INFER_REGISTRY
        assert INFER_REGISTRY["vllm"] == "m_infer.vllm_backend:VLLMBackend"

    def test_build_xinference_registered(self):
        """验证 xinference 后端在注册表中"""
        from m_infer.registry import INFER_REGISTRY
        assert "xinference" in INFER_REGISTRY
        assert INFER_REGISTRY["xinference"] == "m_infer.xinference_backend:XinferenceBackend"

    def test_build_unknown_backend_raises(self):
        """验证未知后端抛 ValueError"""
        with pytest.raises(ValueError, match="unsupported infer backend"):
            build_infer_backend("unknown_xyz", "/tmp/model")

    def test_factory_list_contains_defaults(self):
        """验证工厂可列出的后端包含默认项"""
        from m_infer.registry import list_backends
        backends = list_backends()
        assert "vllm" in backends
        assert "xinference" in backends


class TestInferRequestId:
    def test_request_id_preserved(self):
        """验证 request_id 在请求中保持"""
        req = InferRequest(prompt="test", request_id="req-123")
        assert req.request_id == "req-123"
