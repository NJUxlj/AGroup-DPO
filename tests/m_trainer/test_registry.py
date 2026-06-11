"""测试后端注册表与工厂。"""

import pytest

from m_trainer.registry import (
    BACKEND_REGISTRY,
    list_backends,
    register_backend,
    unregister_backend,
)


class TestRegistry:
    """后端注册表测试。"""

    def test_default_backends_registered(self):
        """默认 4 种后端均已注册。"""
        assert "deepspeed" in BACKEND_REGISTRY
        assert "fsdp" in BACKEND_REGISTRY
        assert "megatron" in BACKEND_REGISTRY
        assert "accelerate" in BACKEND_REGISTRY
        assert len(BACKEND_REGISTRY) >= 4

    def test_list_backends(self):
        """list_backends 返回所有后端名。"""
        names = list_backends()
        assert "deepspeed" in names
        assert "fsdp" in names
        assert isinstance(names, list)

    def test_register_custom_backend(self):
        """注册自定义后端成功。"""
        register_backend("test_custom", "some.module:TestBackend")
        assert "test_custom" in BACKEND_REGISTRY
        assert BACKEND_REGISTRY["test_custom"] == "some.module:TestBackend"
        # 清理
        unregister_backend("test_custom")

    def test_register_duplicate_raises(self):
        """重复注册抛出 ValueError。"""
        register_backend("test_dup", "a.b:C")
        with pytest.raises(ValueError, match="already registered"):
            register_backend("test_dup", "x.y:Z")
        unregister_backend("test_dup")

    def test_unregister_nonexistent_raises(self):
        """取消注册不存在的后端抛出 KeyError。"""
        with pytest.raises(KeyError, match="not found"):
            unregister_backend("nonexistent_backend_xyz")
