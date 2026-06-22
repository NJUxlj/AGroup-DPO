"""m_infer config 单元测试"""
import pytest
from m_infer.config import load_infer_config, resolve_infer_settings


class TestInferConfig:
    def test_load_infer_config(self, tmp_path):
        cfg_file = tmp_path / "infer.yaml"
        cfg_file.write_text(
            "infer:\n"
            "  backend: vllm\n"
            "  model_path: merged_models/test\n"
            "  vllm:\n"
            "    tensor_parallel_size: 2\n",
            encoding="utf-8",
        )
        cfg = load_infer_config(str(cfg_file))
        backend, model_path, kwargs = resolve_infer_settings(cfg)
        assert backend == "vllm"
        assert model_path == "merged_models/test"
        assert kwargs["tensor_parallel_size"] == 2

    def test_cli_override(self, tmp_path):
        cfg_file = tmp_path / "infer.yaml"
        cfg_file.write_text(
            "infer:\n"
            "  backend: vllm\n"
            "  model_path: from_config\n"
            "  xinference:\n"
            "    server_endpoint: http://127.0.0.1:9997\n"
            "    model_uid: null\n",
            encoding="utf-8",
        )
        cfg = load_infer_config(str(cfg_file))
        backend, model_path, kwargs = resolve_infer_settings(
            cfg, backend="xinference", model_path="cli_model",
        )
        assert backend == "xinference"
        assert model_path == "cli_model"
        assert "model_uid" not in kwargs

    def test_missing_infer_key(self, tmp_path):
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("other: {}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing top-level 'infer'"):
            load_infer_config(str(cfg_file))
