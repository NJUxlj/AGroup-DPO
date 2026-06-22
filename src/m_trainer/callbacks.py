"""
训练指标回调 (M03 § 3.6)

将 loss / grad_norm / gpu_mem / throughput / DPO reward margin 写入
TensorBoard 与/或 W&B。仅 rank 0 进程记录，避免分布式重复写入。
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from utils.logger import CustomLogger

if TYPE_CHECKING:
    from .custom_trainer import TrainingConfig

log = CustomLogger.get_logger(__name__)


def _is_main_process() -> bool:
    """分布式训练下仅 rank 0 上报指标。"""
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_rank() == 0
    except (ImportError, RuntimeError):
        pass
    return True


def _resolve_world_size() -> int:
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_world_size()
    except (ImportError, RuntimeError):
        pass
    return max(int(os.environ.get("WORLD_SIZE", "1")), 1)


def parse_report_to(value: Any) -> set[str]:
    """解析 LLaMA-Factory 风格的 report_to 配置。"""
    if value is None:
        return {"tensorboard"}
    if isinstance(value, bool) and not value:
        return set()
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "false"}:
            return set()
        if text in {"all", "both"}:
            return {"tensorboard", "wandb"}
        return {part.strip() for part in text.split(",") if part.strip()}
    if isinstance(value, (list, tuple)):
        return {str(item).strip().lower() for item in value if str(item).strip()}
    return {str(value).strip().lower()}


@dataclass
class StepMetrics:
    """单步（logging window）聚合指标。"""

    loss: float
    learning_rate: float
    global_step: int
    grad_norm: Optional[float] = None
    gpu_mem_gb: Optional[float] = None
    throughput_samples_per_gpu: Optional[float] = None
    reward_margin: Optional[float] = None
    chosen_reward: Optional[float] = None
    rejected_reward: Optional[float] = None
    extras: dict[str, float] = field(default_factory=dict)


class TrainingMetricsCallback(ABC):
    """训练指标回调基类。"""

    @abstractmethod
    def on_train_begin(self, config: "TrainingConfig") -> None: ...

    @abstractmethod
    def on_log(self, metrics: StepMetrics) -> None: ...

    @abstractmethod
    def on_train_end(self) -> None: ...


class TensorBoardCallback(TrainingMetricsCallback):
    """TensorBoard SummaryWriter 回调。"""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self._writer: Any = None

    def on_train_begin(self, config: "TrainingConfig") -> None:
        if not _is_main_process():
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            raise ImportError(
                "tensorboard is not installed. Install with: pip install tensorboard"
            ) from e

        os.makedirs(self.log_dir, exist_ok=True)
        self._writer = SummaryWriter(log_dir=self.log_dir)
        log.info("TensorBoard logging to %s", self.log_dir)

    def on_log(self, metrics: StepMetrics) -> None:
        if self._writer is None:
            return
        step = metrics.global_step
        self._writer.add_scalar("train/loss", metrics.loss, step)
        self._writer.add_scalar("train/learning_rate", metrics.learning_rate, step)
        if metrics.grad_norm is not None:
            self._writer.add_scalar("train/grad_norm", metrics.grad_norm, step)
        if metrics.gpu_mem_gb is not None:
            self._writer.add_scalar("train/gpu_mem_gb", metrics.gpu_mem_gb, step)
        if metrics.throughput_samples_per_gpu is not None:
            self._writer.add_scalar(
                "train/throughput_samples_per_gpu",
                metrics.throughput_samples_per_gpu,
                step,
            )
        if metrics.reward_margin is not None:
            self._writer.add_scalar("train/reward_margin", metrics.reward_margin, step)
        if metrics.chosen_reward is not None:
            self._writer.add_scalar("train/chosen_reward", metrics.chosen_reward, step)
        if metrics.rejected_reward is not None:
            self._writer.add_scalar("train/rejected_reward", metrics.rejected_reward, step)
        for key, value in metrics.extras.items():
            self._writer.add_scalar(f"train/{key}", value, step)

    def on_train_end(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None


class WandbCallback(TrainingMetricsCallback):
    """Weights & Biases 回调。"""

    def __init__(
        self,
        project: str,
        run_name: str,
        config: dict[str, Any],
    ):
        self.project = project
        self.run_name = run_name
        self.config = config
        self._wandb: Any = None
        self._run: Any = None

    def on_train_begin(self, config: "TrainingConfig") -> None:
        if not _is_main_process():
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb is not installed. Install with: pip install wandb"
            ) from e

        self._wandb = wandb
        self._run = wandb.init(
            project=self.project,
            name=self.run_name or None,
            config=self.config,
            reinit=True,
        )
        log.info("W&B run started: project=%s, name=%s", self.project, self.run_name)

    def on_log(self, metrics: StepMetrics) -> None:
        if self._wandb is None:
            return
        payload: dict[str, Any] = {
            "train/loss": metrics.loss,
            "train/learning_rate": metrics.learning_rate,
            "global_step": metrics.global_step,
        }
        if metrics.grad_norm is not None:
            payload["train/grad_norm"] = metrics.grad_norm
        if metrics.gpu_mem_gb is not None:
            payload["train/gpu_mem_gb"] = metrics.gpu_mem_gb
        if metrics.throughput_samples_per_gpu is not None:
            payload["train/throughput_samples_per_gpu"] = metrics.throughput_samples_per_gpu
        if metrics.reward_margin is not None:
            payload["train/reward_margin"] = metrics.reward_margin
        if metrics.chosen_reward is not None:
            payload["train/chosen_reward"] = metrics.chosen_reward
        if metrics.rejected_reward is not None:
            payload["train/rejected_reward"] = metrics.rejected_reward
        payload.update({f"train/{k}": v for k, v in metrics.extras.items()})
        self._wandb.log(payload, step=metrics.global_step)

    def on_train_end(self) -> None:
        if self._wandb is not None and self._run is not None:
            self._wandb.finish()
            self._wandb = None
            self._run = None


class MetricsLogger:
    """聚合多个回调，并提供指标采集辅助方法。"""

    def __init__(self, callbacks: list[TrainingMetricsCallback]):
        self.callbacks = callbacks
        self._window_start = time.perf_counter()
        self._samples_in_window = 0
        self._world_size = _resolve_world_size()

    @classmethod
    def from_config(cls, config: "TrainingConfig") -> "MetricsLogger":
        return cls(build_metrics_callbacks(config))

    def on_train_begin(self, config: "TrainingConfig") -> None:
        for cb in self.callbacks:
            cb.on_train_begin(config)

    def on_train_end(self) -> None:
        for cb in self.callbacks:
            cb.on_train_end()

    def record_batch_samples(self, batch_size: int) -> None:
        self._samples_in_window += batch_size

    def reset_window(self) -> None:
        self._window_start = time.perf_counter()
        self._samples_in_window = 0

    def throughput_samples_per_gpu(self) -> Optional[float]:
        elapsed = time.perf_counter() - self._window_start
        if elapsed <= 0 or self._samples_in_window <= 0:
            return None
        total_samples = self._samples_in_window * self._world_size
        return total_samples / elapsed / self._world_size

    def on_log(self, metrics: StepMetrics) -> None:
        if metrics.throughput_samples_per_gpu is None:
            metrics.throughput_samples_per_gpu = self.throughput_samples_per_gpu()
        for cb in self.callbacks:
            cb.on_log(metrics)
        self.reset_window()


def build_metrics_callbacks(config: "TrainingConfig") -> list[TrainingMetricsCallback]:
    """根据 TrainingConfig.report_to 构建回调列表。"""
    targets = parse_report_to(config.report_to)
    if not targets or not _is_main_process():
        return []

    callbacks: list[TrainingMetricsCallback] = []
    log_dir = config.logging_dir or os.path.join(config.output_dir, "runs")

    if "tensorboard" in targets:
        callbacks.append(TensorBoardCallback(log_dir=log_dir))

    if "wandb" in targets:
        run_name = config.run_name or os.path.basename(config.output_dir.rstrip("/"))
        wandb_config = {
            "stage": config.stage,
            "model_name_or_path": config.model_name_or_path,
            "distributed_backend": config.distributed_backend,
            "learning_rate": config.learning_rate,
            "per_device_batch_size": config.per_device_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "num_train_epochs": config.num_train_epochs,
            "seed": config.seed,
            "dpo_beta": config.dpo_beta,
        }
        callbacks.append(
            WandbCallback(
                project=config.wandb_project,
                run_name=run_name,
                config=wandb_config,
            )
        )

    return callbacks


def compute_grad_norm(model: Any) -> Optional[float]:
    """计算全局梯度 L2 范数。"""
    import torch

    params = model.parameters() if hasattr(model, "parameters") else []
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return None
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in grads]), 2)
    return float(total_norm.item())


def current_gpu_mem_gb() -> Optional[float]:
    """当前 GPU 已分配显存（GB）。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / (1024**3)
    except ImportError:
        return None
