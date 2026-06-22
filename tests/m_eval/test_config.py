"""m_eval config 单元测试"""
import pytest
from m_eval.config import load_eval_config, resolve_eval_settings


class TestEvalConfig:
    def test_load_eval_config(self, tmp_path):
        cfg_file = tmp_path / "eval.yaml"
        cfg_file.write_text(
            "eval:\n"
            "  output_dir: reports/\n"
            "  max_new_tokens: 128\n"
            "  temperature: 0.2\n"
            "  datasets:\n"
            "    - name: medical_qa_1000\n"
            "      path: data/eval/medical_qa_1000.jsonl\n"
            "  thresholds:\n"
            "    bleu_4: 0.30\n",
            encoding="utf-8",
        )
        cfg = load_eval_config(str(cfg_file))
        settings = resolve_eval_settings(cfg)
        assert settings["max_new_tokens"] == 128
        assert settings["temperature"] == 0.2
        assert len(settings["datasets"]) == 1
        assert settings["thresholds"]["bleu_4"] == 0.30

    def test_missing_eval_key(self, tmp_path):
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("other: {}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing top-level 'eval'"):
            load_eval_config(str(cfg_file))
