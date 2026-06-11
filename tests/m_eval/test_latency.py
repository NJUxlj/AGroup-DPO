"""m_eval latency 单元测试"""
import pytest
from m_eval.latency import LatencyStat, aggregate_latency
from m_infer.base import InferResponse


class TestLatencyStat:
    def test_default_values(self):
        stat = LatencyStat()
        assert stat.p50_first_token_ms == 0.0
        assert stat.throughput_samples_per_s == 0.0

    def test_custom_values(self):
        stat = LatencyStat(
            p50_first_token_ms=100.0,
            p95_total_ms=500.0,
            throughput_samples_per_s=10.0,
        )
        assert stat.p50_first_token_ms == 100.0
        assert stat.p95_total_ms == 500.0
        assert stat.throughput_samples_per_s == 10.0


class TestAggregateLatency:
    def test_empty(self):
        stat = aggregate_latency([])
        assert stat.p50_total_ms == 0.0

    def test_single_response(self):
        resp = InferResponse(
            text="hello",
            latency_ms=50.0,
            total_latency_ms=200.0,
        )
        stat = aggregate_latency([resp])
        assert stat.p50_total_ms == 200.0
        assert stat.p50_first_token_ms == 50.0

    def test_multiple_responses(self):
        responses = [
            InferResponse(text=f"hello {i}", latency_ms=50.0 * i, total_latency_ms=200.0 * i)
            for i in range(1, 6)
        ]
        stat = aggregate_latency(responses)
        assert stat.p50_total_ms > 0
        assert stat.p95_total_ms > 0
        assert stat.throughput_samples_per_s > 0

    def test_zero_latency_in_responses(self):
        """有 latency_ms=0 的响应不应影响统计"""
        responses = [
            InferResponse(text="a", latency_ms=0.0, total_latency_ms=100.0),
            InferResponse(text="b", latency_ms=50.0, total_latency_ms=200.0),
        ]
        stat = aggregate_latency(responses)
        assert stat.p50_first_token_ms == 50.0
