"""测试 m_merge.exporter —— LoRA 合并导出 (FR-06)。"""

from __future__ import annotations

import json
import os
import tempfile
from unittest import mock

import pytest
import torch

from m_merge.exporter import merge_and_export, _validate_adapter_is_lora


# ═══════════════════════════════════════════════════════════
# _validate_adapter_is_lora
# ═══════════════════════════════════════════════════════════

class TestValidateAdapterIsLora:
    """Adapter 类型校验。"""

    def test_valid_lora(self):
        """peft_type=LORA 时通过。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "adapter_config.json")
            with open(cfg_path, "w") as f:
                json.dump({"peft_type": "LORA"}, f)
            _validate_adapter_is_lora(tmp)  # 不抛异常

    def test_invalid_peft_type(self):
        """非 LORA 类型抛出 ValueError。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "adapter_config.json")
            with open(cfg_path, "w") as f:
                json.dump({"peft_type": "IA3"}, f)
            with pytest.raises(ValueError, match="期望 LORA adapter"):
                _validate_adapter_is_lora(tmp)

    def test_unknown_peft_type(self):
        """peft_type 字段缺失时也抛出。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "adapter_config.json")
            with open(cfg_path, "w") as f:
                json.dump({}, f)
            with pytest.raises(ValueError, match="期望 LORA adapter"):
                _validate_adapter_is_lora(tmp)

    def test_missing_config_no_error(self):
        """adapter_config.json 不存在时仅 warning，不抛异常。"""
        with tempfile.TemporaryDirectory() as tmp:
            _validate_adapter_is_lora(tmp)  # 不抛异常


# ═══════════════════════════════════════════════════════════
# merge_and_export — 路径校验
# ═══════════════════════════════════════════════════════════

class TestMergeAndExportPathValidation:
    """路径校验测试。"""

    def test_base_model_not_found(self):
        """基座模型路径不存在时抛 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="base model not found"):
            merge_and_export(
                base_model_path="/nonexistent/path",
                adapter_path="/tmp",
                export_dir="/tmp/out",
            )

    def test_adapter_not_found(self):
        """adapter 路径不存在时抛 FileNotFoundError。"""
        with tempfile.TemporaryDirectory() as base_dir:
            # base_dir exists but adapter doesn't
            with pytest.raises(FileNotFoundError, match="adapter not found"):
                merge_and_export(
                    base_model_path=base_dir,
                    adapter_path="/nonexistent/adapter",
                    export_dir="/tmp/out",
                )

    def test_peft_not_installed(self):
        """peft 未安装时抛 ImportError。"""
        with tempfile.TemporaryDirectory() as base_dir, \
             tempfile.TemporaryDirectory() as adapter_dir:
            # 创建 adapter_config.json 以通过类型校验
            cfg_path = os.path.join(adapter_dir, "adapter_config.json")
            with open(cfg_path, "w") as f:
                json.dump({"peft_type": "LORA"}, f)

            with mock.patch.dict("sys.modules", {"peft": None}):
                with pytest.raises(ImportError, match="peft is not installed"):
                    merge_and_export(
                        base_model_path=base_dir,
                        adapter_path=adapter_dir,
                        export_dir="/tmp/out",
                    )


# ═══════════════════════════════════════════════════════════
# merge_and_export — 设备策略
# ═══════════════════════════════════════════════════════════

class TestMergeAndExportDevice:
    """设备策略测试。"""

    def test_cuda_fallback_when_unavailable(self):
        """export_device='cuda' 但 CUDA 不可用时回退 CPU。"""
        with tempfile.TemporaryDirectory() as base_dir, \
             tempfile.TemporaryDirectory() as adapter_dir, \
             tempfile.TemporaryDirectory() as export_dir:
            cfg_path = os.path.join(adapter_dir, "adapter_config.json")
            with open(cfg_path, "w") as f:
                json.dump({"peft_type": "LORA"}, f)

            with mock.patch.object(torch.cuda, "is_available", return_value=False), \
                 mock.patch("m_merge.exporter.AutoModelForCausalLM") as mock_auto, \
                 mock.patch("m_merge.exporter.AutoTokenizer") as mock_tok, \
                 mock.patch("peft.PeftModel") as mock_peft:

                mock_base = mock.MagicMock()
                mock_auto.from_pretrained.return_value = mock_base
                mock_peft.from_pretrained.return_value = mock.MagicMock()
                mock_merged = mock.MagicMock()
                mock_peft.from_pretrained.return_value.merge_and_unload.return_value = mock_merged

                merge_and_export(
                    base_model_path=base_dir,
                    adapter_path=adapter_dir,
                    export_dir=export_dir,
                    export_device="cuda",
                )

                call_kwargs = mock_auto.from_pretrained.call_args.kwargs
                # CUDA 不可用时应不传 device_map
                assert "device_map" not in call_kwargs

    def test_cpu_mode_no_device_map(self):
        """export_device='cpu' 时不传 device_map。"""
        with tempfile.TemporaryDirectory() as base_dir, \
             tempfile.TemporaryDirectory() as adapter_dir, \
             tempfile.TemporaryDirectory() as export_dir:
            cfg_path = os.path.join(adapter_dir, "adapter_config.json")
            with open(cfg_path, "w") as f:
                json.dump({"peft_type": "LORA"}, f)

            with mock.patch("m_merge.exporter.AutoModelForCausalLM") as mock_auto, \
                 mock.patch("m_merge.exporter.AutoTokenizer") as mock_tok, \
                 mock.patch("peft.PeftModel") as mock_peft:

                mock_base = mock.MagicMock()
                mock_auto.from_pretrained.return_value = mock_base
                mock_peft.from_pretrained.return_value = mock.MagicMock()
                mock_merged = mock.MagicMock()
                mock_peft.from_pretrained.return_value.merge_and_unload.return_value = mock_merged

                merge_and_export(
                    base_model_path=base_dir,
                    adapter_path=adapter_dir,
                    export_dir=export_dir,
                    export_device="cpu",
                )

                call_kwargs = mock_auto.from_pretrained.call_args.kwargs
                assert "device_map" not in call_kwargs
                assert call_kwargs["low_cpu_mem_usage"] is True


# ═══════════════════════════════════════════════════════════
# merge_and_export — 完整流程
# ═══════════════════════════════════════════════════════════

class TestMergeAndExportFullPipeline:
    """完整合并流程测试（mock transformers + peft）。"""

    def test_successful_merge_returns_abs_path(self):
        """正常合并返回导出目录的绝对路径。"""
        with tempfile.TemporaryDirectory() as base_dir, \
             tempfile.TemporaryDirectory() as adapter_dir, \
             tempfile.TemporaryDirectory() as export_dir:
            cfg_path = os.path.join(adapter_dir, "adapter_config.json")
            with open(cfg_path, "w") as f:
                json.dump({"peft_type": "LORA"}, f)

            with mock.patch("m_merge.exporter.AutoModelForCausalLM") as mock_auto, \
                 mock.patch("m_merge.exporter.AutoTokenizer") as mock_tok, \
                 mock.patch("peft.PeftModel") as mock_peft:

                mock_base = mock.MagicMock()
                mock_auto.from_pretrained.return_value = mock_base
                mock_peft_model = mock.MagicMock()
                mock_peft.from_pretrained.return_value = mock_peft_model
                mock_merged = mock.MagicMock()
                mock_peft_model.merge_and_unload.return_value = mock_merged

                result = merge_and_export(
                    base_model_path=base_dir,
                    adapter_path=adapter_dir,
                    export_dir=export_dir,
                    export_device="cpu",
                )

                # 断言返回值为绝对路径
                assert os.path.isabs(result)
                assert result == os.path.abspath(export_dir)

    def test_merge_saves_model_and_tokenizer(self):
        """合并后调用 save_pretrained 保存模型和 tokenizer。"""
        with tempfile.TemporaryDirectory() as base_dir, \
             tempfile.TemporaryDirectory() as adapter_dir, \
             tempfile.TemporaryDirectory() as export_dir:
            cfg_path = os.path.join(adapter_dir, "adapter_config.json")
            with open(cfg_path, "w") as f:
                json.dump({"peft_type": "LORA"}, f)

            with mock.patch("m_merge.exporter.AutoModelForCausalLM") as mock_auto, \
                 mock.patch("m_merge.exporter.AutoTokenizer") as mock_tok, \
                 mock.patch("peft.PeftModel") as mock_peft:

                mock_base = mock.MagicMock()
                mock_auto.from_pretrained.return_value = mock_base
                mock_peft_model = mock.MagicMock()
                mock_peft.from_pretrained.return_value = mock_peft_model
                mock_merged = mock.MagicMock()
                mock_peft_model.merge_and_unload.return_value = mock_merged
                mock_tokenizer = mock.MagicMock()
                mock_tok.from_pretrained.return_value = mock_tokenizer

                merge_and_export(
                    base_model_path=base_dir,
                    adapter_path=adapter_dir,
                    export_dir=export_dir,
                    export_device="cpu",
                )

                mock_merged.save_pretrained.assert_called_once()
                mock_tokenizer.save_pretrained.assert_called_once_with(export_dir)

    def test_merge_cleans_up_intermediate_objects(self):
        """合并后释放中间对象（base + peft_model 被 del）。"""
        with tempfile.TemporaryDirectory() as base_dir, \
             tempfile.TemporaryDirectory() as adapter_dir, \
             tempfile.TemporaryDirectory() as export_dir:
            cfg_path = os.path.join(adapter_dir, "adapter_config.json")
            with open(cfg_path, "w") as f:
                json.dump({"peft_type": "LORA"}, f)

            with mock.patch("m_merge.exporter.AutoModelForCausalLM") as mock_auto, \
                 mock.patch("m_merge.exporter.AutoTokenizer") as mock_tok, \
                 mock.patch("peft.PeftModel") as mock_peft, \
                 mock.patch.object(torch.cuda, "empty_cache") as mock_empty_cache, \
                 mock.patch.object(torch.cuda, "is_available", return_value=True):

                mock_base = mock.MagicMock()
                mock_auto.from_pretrained.return_value = mock_base
                mock_peft_model = mock.MagicMock()
                mock_peft.from_pretrained.return_value = mock_peft_model
                mock_merged = mock.MagicMock()
                mock_peft_model.merge_and_unload.return_value = mock_merged

                merge_and_export(
                    base_model_path=base_dir,
                    adapter_path=adapter_dir,
                    export_dir=export_dir,
                    export_device="cuda",
                )

                # CUDA 可用时调用 empty_cache
                assert mock_empty_cache.call_count >= 2

    def test_export_size_passed_to_save(self):
        """export_size 正确传给 save_pretrained。"""
        with tempfile.TemporaryDirectory() as base_dir, \
             tempfile.TemporaryDirectory() as adapter_dir, \
             tempfile.TemporaryDirectory() as export_dir:
            cfg_path = os.path.join(adapter_dir, "adapter_config.json")
            with open(cfg_path, "w") as f:
                json.dump({"peft_type": "LORA"}, f)

            with mock.patch("m_merge.exporter.AutoModelForCausalLM") as mock_auto, \
                 mock.patch("m_merge.exporter.AutoTokenizer") as mock_tok, \
                 mock.patch("peft.PeftModel") as mock_peft:

                mock_base = mock.MagicMock()
                mock_auto.from_pretrained.return_value = mock_base
                mock_peft_model = mock.MagicMock()
                mock_peft.from_pretrained.return_value = mock_peft_model
                mock_merged = mock.MagicMock()
                mock_peft_model.merge_and_unload.return_value = mock_merged

                merge_and_export(
                    base_model_path=base_dir,
                    adapter_path=adapter_dir,
                    export_dir=export_dir,
                    export_size=3,  # 3GB
                    export_device="cpu",
                )

                _, call_kwargs = mock_merged.save_pretrained.call_args
                assert call_kwargs["max_shard_size"] == "3GB"


# ═══════════════════════════════════════════════════════════
# merge_and_export — torch_dtype 参数
# ═══════════════════════════════════════════════════════════

class TestMergeAndExportDtype:
    """torch_dtype 参数测试。"""

    def test_explicit_dtype_passed(self):
        """显式指定的 torch_dtype 传给 from_pretrained。"""
        with tempfile.TemporaryDirectory() as base_dir, \
             tempfile.TemporaryDirectory() as adapter_dir, \
             tempfile.TemporaryDirectory() as export_dir:
            cfg_path = os.path.join(adapter_dir, "adapter_config.json")
            with open(cfg_path, "w") as f:
                json.dump({"peft_type": "LORA"}, f)

            with mock.patch("m_merge.exporter.AutoModelForCausalLM") as mock_auto, \
                 mock.patch("m_merge.exporter.AutoTokenizer") as mock_tok, \
                 mock.patch("peft.PeftModel") as mock_peft:

                mock_base = mock.MagicMock()
                mock_auto.from_pretrained.return_value = mock_base
                mock_peft.from_pretrained.return_value = mock.MagicMock()
                mock_peft.from_pretrained.return_value.merge_and_unload.return_value = mock.MagicMock()

                merge_and_export(
                    base_model_path=base_dir,
                    adapter_path=adapter_dir,
                    export_dir=export_dir,
                    torch_dtype=torch.float16,
                    export_device="cpu",
                )

                call_kwargs = mock_auto.from_pretrained.call_args.kwargs
                assert call_kwargs["torch_dtype"] == torch.float16

    def test_cuda_defaults_to_bfloat16(self):
        """CUDA 模式下未指定 dtype 时默认 bfloat16。"""
        with tempfile.TemporaryDirectory() as base_dir, \
             tempfile.TemporaryDirectory() as adapter_dir, \
             tempfile.TemporaryDirectory() as export_dir:
            cfg_path = os.path.join(adapter_dir, "adapter_config.json")
            with open(cfg_path, "w") as f:
                json.dump({"peft_type": "LORA"}, f)

            with mock.patch("m_merge.exporter.AutoModelForCausalLM") as mock_auto, \
                 mock.patch("m_merge.exporter.AutoTokenizer") as mock_tok, \
                 mock.patch("peft.PeftModel") as mock_peft, \
                 mock.patch.object(torch.cuda, "is_available", return_value=True), \
                 mock.patch.object(torch.cuda, "empty_cache"):

                mock_base = mock.MagicMock()
                mock_auto.from_pretrained.return_value = mock_base
                mock_peft.from_pretrained.return_value = mock.MagicMock()
                mock_peft.from_pretrained.return_value.merge_and_unload.return_value = mock.MagicMock()

                merge_and_export(
                    base_model_path=base_dir,
                    adapter_path=adapter_dir,
                    export_dir=export_dir,
                    export_device="cuda",
                    torch_dtype=None,  # 默认行为
                )

                call_kwargs = mock_auto.from_pretrained.call_args.kwargs
                assert call_kwargs["torch_dtype"] == torch.bfloat16
