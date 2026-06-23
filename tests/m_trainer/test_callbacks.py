"""训练指标回调测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from m_trainer.callbacks import (
    MetricsLogger,
    StepMetrics,
    TensorBoardCallback,
    WandbCallback,
    build_metrics_callbacks,
    parse_report_to,
)
from m_trainer.custom_trainer import TrainingConfig


class TestParseReportTo:
    def test_none(self):
        assert parse_report_to("none") == set()

    def test_tensorboard(self):
        assert parse_report_to("tensorboard") == {"tensorboard"}

    def test_all(self):
        assert parse_report_to("all") == {"tensorboard", "wandb"}

    def test_list(self):
        assert parse_report_to(["tensorboard", "wandb"]) == {"tensorboard", "wandb"}


class TestBuildMetricsCallbacks:
    def test_none_report(self):
        cfg = TrainingConfig(report_to="none")
        assert build_metrics_callbacks(cfg) == []

    def test_tensorboard_only(self):
        cfg = TrainingConfig(report_to="tensorboard", output_dir="/tmp/test_run")
        callbacks = build_metrics_callbacks(cfg)
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], TensorBoardCallback)

    def test_wandb_only(self):
        cfg = TrainingConfig(report_to="wandb", output_dir="/tmp/test_run")
        callbacks = build_metrics_callbacks(cfg)
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], WandbCallback)


class TestTensorBoardCallback:
    def test_on_log_writes_scalars(self):
        writer = MagicMock()
        cb = TensorBoardCallback(log_dir="/tmp/tb_test")
        cb._writer = writer

        cb.on_log(
            StepMetrics(
                loss=0.42,
                learning_rate=1e-4,
                global_step=10,
                grad_norm=1.5,
                gpu_mem_gb=12.0,
                throughput_samples_per_gpu=1.8,
                reward_margin=0.6,
                chosen_reward=1.2,
                rejected_reward=0.6,
            )
        )

        writer.add_scalar.assert_any_call("train/loss", 0.42, 10)
        writer.add_scalar.assert_any_call("train/grad_norm", 1.5, 10)
        writer.add_scalar.assert_any_call("train/gpu_mem_gb", 12.0, 10)
        writer.add_scalar.assert_any_call("train/throughput_samples_per_gpu", 1.8, 10)
        writer.add_scalar.assert_any_call("train/reward_margin", 0.6, 10)


class TestMetricsLogger:
    def test_throughput_window(self):
        logger = MetricsLogger([])
        logger._world_size = 2
        logger.record_batch_samples(4)
        logger.record_batch_samples(4)
        with patch("m_trainer.callbacks.time.perf_counter", return_value=2.0):
            logger._window_start = 0.0
            throughput = logger.throughput_samples_per_gpu()
        assert throughput == 4.0

    def test_on_log_resets_window(self):
        cb = MagicMock()
        logger = MetricsLogger([cb])
        logger.record_batch_samples(2)
        with patch("m_trainer.callbacks.time.perf_counter", side_effect=[0.0, 1.0, 1.0]):
            logger._window_start = 0.0
            logger.on_log(
                StepMetrics(loss=0.1, learning_rate=1e-5, global_step=1)
            )
        cb.on_log.assert_called_once()
        assert logger._samples_in_window == 0
