"""m_eval reporter 单元测试"""
import json
import os
import tempfile
import pytest
from m_eval.reporter import EvalReporter
from m_eval.latency import LatencyStat


class TestEvalReporter:
    def test_to_dict_basic(self):
        reporter = EvalReporter(
            model_version="test-v1.0",
            infer_backend="vllm",
        )
        reporter.add_dataset("test_set", accuracy=0.85, bleu_4=0.34, rouge_l=0.48, n_samples=100)
        data = reporter.to_dict()
        assert data["model_version"] == "test-v1.0"
        assert data["infer_backend"] == "vllm"
        assert data["datasets"]["test_set"]["accuracy"] == 0.85

    def test_to_dict_with_latency(self):
        reporter = EvalReporter(model_version="test-v1.0")
        latency = LatencyStat(
            p50_first_token_ms=100.0,
            p95_total_ms=500.0,
            throughput_samples_per_s=10.0,
        )
        reporter.set_latency(latency)
        data = reporter.to_dict()
        assert "latency" in data
        assert data["latency"]["p50_first_token_ms"] == 100.0

    def test_write_json(self):
        reporter = EvalReporter(model_version="test-v1.0")
        reporter.add_dataset("test", accuracy=0.8, bleu_4=0.3, rouge_l=0.4, n_samples=10)
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "report")
            json_path, md_path = reporter.write(base)
            assert os.path.exists(json_path)
            assert os.path.exists(md_path)
            with open(json_path) as f:
                data = json.load(f)
            assert data["model_version"] == "test-v1.0"

    def test_write_markdown(self):
        reporter = EvalReporter(model_version="test-v1.0")
        reporter.add_dataset("medical_qa", accuracy=0.682, bleu_4=0.341, rouge_l=0.482, n_samples=1000)
        latency = LatencyStat(p50_total_ms=980.0, throughput_samples_per_s=9.8)
        reporter.set_latency(latency)
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "report")
            _, md_path = reporter.write(base)
            with open(md_path) as f:
                content = f.read()
            assert "# 评测报告" in content
            assert "medical_qa" in content
            assert "0.6820" in content

    def test_baseline_comparison(self):
        reporter = EvalReporter(
            model_version="dpo-v1.2",
            baseline_model="base-v1.0",
        )
        reporter.add_dataset("main", accuracy=0.75, bleu_4=0.35, rouge_l=0.50, n_samples=100)
        reporter.set_baseline(accuracy=0.65, rouge_l=0.45)
        data = reporter.to_dict()
        assert "baseline_comparison" in data
        assert data["baseline_comparison"]["baseline_model"] == "base-v1.0"
