"""m_infer 注册表单元测试"""
import pytest
from m_infer.registry import (
    INFER_REGISTRY, list_backends,
    register_backend, unregister_backend,
)


class TestInferRegistry:
    def test_default_backends_registered(self):
        """验证默认 vllm / xinference 已注册"""
        assert "vllm" in INFER_REGISTRY
        assert "xinference" in INFER_REGISTRY

    def test_list_backends(self):
        """验证 list_backends 返回包含默认后端"""
        backends = list_backends()
        assert "vllm" in backends
        assert "xinference" in backends

    def test_register_new_backend(self):
        """验证注册新后端"""
        register_backend("test", "m_infer.test:TestBackend")
        assert INFER_REGISTRY["test"] == "m_infer.test:TestBackend"
        unregister_backend("test")

    def test_register_duplicate_raises(self):
        """验证重复注册抛异常"""
        with pytest.raises(ValueError, match="already registered"):
            register_backend("vllm", "some.module:SomeClass")

    def test_unregister_missing_raises(self):
        """验证注销不存在后端抛异常"""
        with pytest.raises(KeyError, match="not found"):
            unregister_backend("nonexistent_backend_xyz")
