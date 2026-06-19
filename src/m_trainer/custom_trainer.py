"""
CustomTrainer — 自研训练器 (M04)

封装完整的训练流程：模型加载 → 数据加载 → 分布式后端初始化 → 训练循环 → checkpoint。
支持 SFT 和 DPO 两种训练阶段，与 LLaMA-Factory 的训练配置格式兼容。

Usage:
    from m_trainer.custom_trainer import CustomTrainer
    trainer = CustomTrainer.from_yaml("configs/my_train.yaml")
    trainer.train()
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from utils.logger import CustomLogger

from .backends.base import TrainerConfig
from .backends.optimizer_factory import AdamWFactory
from .factory import build_backend

log = CustomLogger.get_logger(__name__)


# ---------------------------------------------------------------------------
# TrainingConfig — 扩展 TrainerConfig，加入模型/数据字段
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """完整训练配置，兼容 LLaMA-Factory YAML 格式。

    从 YAML 文件加载，包含模型路径、数据集、训练超参等所有必要信息。
    """

    # ---- 模型 ----
    model_name_or_path: str = ""
    trust_remote_code: bool = False

    # ---- 数据 ----
    dataset: str = ""
    dataset_dir: str = "data"
    template: str = "qwen"
    cutoff_len: int = 2048

    # ---- 训练阶段 ----
    stage: str = "sft"  # sft / dpo

    # ---- LoRA ----
    finetuning_type: str = "lora"  # lora / full
    lora_target: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # ---- TrainerConfig 字段 (内嵌) ----
    distributed_backend: str = "deepspeed"
    output_dir: str = "saves/custom"
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    num_train_epochs: float = 1.0
    max_steps: int = -1
    bf16: bool = True
    seed: int = 42
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 2
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"

    # ---- 后端专属配置 ----
    deepspeed_config: dict[str, Any] = field(default_factory=dict)
    fsdp_config: dict[str, Any] = field(default_factory=dict)
    megatron_config: dict[str, Any] = field(default_factory=dict)

    def to_trainer_config(self) -> TrainerConfig:
        """提取分布式训练相关字段为 TrainerConfig。"""
        return TrainerConfig(
            distributed_backend=self.distributed_backend,
            output_dir=self.output_dir,
            per_device_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            bf16=self.bf16,
            seed=self.seed,
            deepspeed_config=self.deepspeed_config,
            fsdp_config=self.fsdp_config,
            megatron_config=self.megatron_config,
        )


# ---------------------------------------------------------------------------
# 简易 JSONL Dataset
# ---------------------------------------------------------------------------


class JSONLinesDataset(Dataset):
    """从 JSONL 文件加载训练数据。

    支持两种格式：
      - SFT: {"instruction": ..., "input": ..., "output": ...}
      - DPO: {"prompt": ..., "chosen": ..., "rejected": ...}
    """

    def __init__(self, path: str):
        self.samples: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        if not self.samples:
            raise ValueError(f"No samples loaded from {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def _sft_collate(batch: list[dict], tokenizer, cutoff_len: int) -> dict[str, torch.Tensor]:
    """将 SFT 样本 tokenize 为 input_ids + labels。"""
    texts = []
    for item in batch:
        instruction = item.get("instruction", "")
        inp = item.get("input", "")
        output = item.get("output", "")
        if inp:
            prompt = f"<|im_start|>system\n你是AI财保助理。<|im_end|>\n<|im_start|>user\n{instruction}\n{inp}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>system\n你是AI财保助理。<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        texts.append(prompt + output + "<|im_end|>")

    enc = tokenizer(texts, truncation=True, max_length=cutoff_len, padding=True, return_tensors="pt")
    enc["labels"] = enc["input_ids"].clone()
    return enc


def _dpo_collate(batch: list[dict], tokenizer, cutoff_len: int) -> dict[str, torch.Tensor]:
    """将 DPO 样本 tokenize 为 DPO 训练所需字段。"""
    prompts, chosens, rejecteds = [], [], []
    for item in batch:
        prompts.append(item.get("prompt", ""))
        chosens.append(item.get("chosen", ""))
        rejecteds.append(item.get("rejected", ""))

    p_enc = tokenizer(prompts, truncation=True, max_length=cutoff_len, padding=True, return_tensors="pt")
    c_enc = tokenizer(chosens, truncation=True, max_length=cutoff_len, padding=True, return_tensors="pt")
    r_enc = tokenizer(rejecteds, truncation=True, max_length=cutoff_len, padding=True, return_tensors="pt")

    return {
        "prompt_input_ids": p_enc["input_ids"],
        "prompt_attention_mask": p_enc["attention_mask"],
        "chosen_input_ids": c_enc["input_ids"],
        "chosen_attention_mask": c_enc["attention_mask"],
        "rejected_input_ids": r_enc["input_ids"],
        "rejected_attention_mask": r_enc["attention_mask"],
    }


# ---------------------------------------------------------------------------
# CustomTrainer
# ---------------------------------------------------------------------------


class CustomTrainer:
    """自研训练器，封装完整的训练流程。

    支持通过 --backend 在 LLaMA-Factory 与自定义分布式后端之间切换。
    自定义后端通过 build_backend() 加载 deepspeed / fsdp / accelerate / megatron。

    Usage:
        trainer = CustomTrainer.from_yaml("configs/my_train.yaml")
        trainer.train()
    """

    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self.backend = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 工厂方法 ----

    @classmethod
    def from_yaml(cls, path: str) -> "CustomTrainer":
        """从 YAML 配置文件加载训练器。

        YAML 格式兼容 LLaMA-Factory 训练配置，额外字段会被忽略。
        """
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # 提取已知字段，忽略未知字段
        known = {f.name for f in TrainingConfig.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in raw.items() if k in known}
        return cls(TrainingConfig(**kwargs))

    # ---- 训练入口 ----

    def train(self) -> None:
        """执行完整训练流程。"""
        log.info("=" * 60)
        log.info("CustomTrainer: stage=%s, backend=%s", self.cfg.stage, self.cfg.distributed_backend)
        log.info("  model: %s", self.cfg.model_name_or_path)
        log.info("  output: %s", self.cfg.output_dir)
        log.info("=" * 60)

        self._load_model_and_tokenizer()
        dataloader = self._load_data()
        self._init_backend()

        self._run_training_loop(dataloader)

        self._save_checkpoint("final")
        log.info("Training complete. Model saved to %s", self.cfg.output_dir)

    # ---- 内部方法 ----

    def _load_model_and_tokenizer(self) -> None:
        """加载模型和 tokenizer。"""
        log.info("Loading model from %s ...", self.cfg.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name_or_path,
            trust_remote_code=self.cfg.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.bfloat16 if self.cfg.bf16 else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=self.cfg.trust_remote_code,
        )

        # 应用 LoRA
        if self.cfg.finetuning_type == "lora":
            self._apply_lora()

        self.model.to(self._device)
        self.model.train()
        log.info("Model loaded: %s params", sum(p.numel() for p in self.model.parameters()))

    def _apply_lora(self) -> None:
        """对模型应用 LoRA adapter。"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            log.warning("peft not installed, skipping LoRA")
            return

        target_modules = [m.strip() for m in self.cfg.lora_target.split(",")]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            target_modules=target_modules,
        )
        self.model = get_peft_model(self.model, lora_config)  # type: ignore[assignment]
        log.info("LoRA applied: rank=%s, targets=%s", self.cfg.lora_rank, target_modules)

    def _load_data(self) -> DataLoader:
        """加载训练数据集。"""
        data_path = os.path.join(self.cfg.dataset_dir, f"{self.cfg.dataset}.jsonl")
        if not os.path.exists(data_path):
            data_path = os.path.join(self.cfg.dataset_dir, self.cfg.dataset, "*.jsonl")
            raise FileNotFoundError(
                f"Dataset not found at {data_path}. "
                f"Expected {self.cfg.dataset_dir}/{self.cfg.dataset}.jsonl"
            )

        dataset = JSONLinesDataset(data_path)
        log.info("Loaded %d samples from %s", len(dataset), data_path)

        collate_fn = _sft_collate if self.cfg.stage == "sft" else _dpo_collate
        return DataLoader(
            dataset,
            batch_size=self.cfg.per_device_batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, self.tokenizer, self.cfg.cutoff_len),
        )

    def _init_backend(self) -> None:
        """初始化分布式后端。"""
        trainer_cfg = self.cfg.to_trainer_config()

        if self.cfg.distributed_backend == "deepspeed":
            # DeepSpeed: optimizer 由 engine 内部创建
            optimizer = None
        else:
            optimizer = AdamWFactory().build(
                self.model.parameters(),  # type: ignore[arg-type]
                {"learning_rate": self.cfg.learning_rate},
            )

        self.backend = build_backend(trainer_cfg)
        self.model, self.optimizer = self.backend.init(  # type: ignore[assignment]
            self.model, optimizer, trainer_cfg  # type: ignore[arg-type]
        )

    def _run_training_loop(self, dataloader: DataLoader) -> None:
        """主训练循环。"""
        total_steps = (
            self.cfg.max_steps
            if self.cfg.max_steps > 0
            else int(len(dataloader) * self.cfg.num_train_epochs)
        )
        grad_accum = self.cfg.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)

        # LR scheduler (DeepSpeed 内部管理 LR，跳过 torch scheduler)
        scheduler = None
        if self.cfg.distributed_backend != "deepspeed":
            optimizer_for_scheduler = getattr(self, "optimizer", None)
            if optimizer_for_scheduler is None and self.backend is not None:
                optimizer_for_scheduler = getattr(self.backend, "_optimizer", None)

            if optimizer_for_scheduler is not None:
                if self.cfg.lr_scheduler_type == "cosine":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer_for_scheduler, T_max=total_steps
                    )

        log.info("Training: total_steps=%d, grad_accum=%d, warmup=%d", total_steps, grad_accum, warmup_steps)

        global_step = 0
        total_loss = 0.0
        t_start = time.perf_counter()

        # Gradient accumulation counter (manual for DeepSpeed)
        accum_step = 0

        for epoch in range(int(math.ceil(self.cfg.num_train_epochs))):
            if global_step >= total_steps:
                break

            for batch in dataloader:
                if global_step >= total_steps:
                    break

                batch = {k: v.to(self._device) for k, v in batch.items()}

                if self.cfg.stage == "sft":
                    loss = self._sft_step(batch)
                else:
                    loss = self._dpo_step(batch)

                loss = loss / grad_accum
                self.backend.backward(loss)  # type: ignore[union-attr]
                accum_step += 1
                total_loss += loss.item()

                if accum_step >= grad_accum:
                    # Warmup LR
                    if global_step < warmup_steps and optimizer_for_scheduler is not None:
                        lr_scale = (global_step + 1) / max(warmup_steps, 1)
                        for pg in optimizer_for_scheduler.param_groups:
                            pg["lr"] = self.cfg.learning_rate * lr_scale

                    self.backend.step()  # type: ignore[union-attr]
                    if scheduler is not None:
                        scheduler.step()
                    self.backend.zero_grad()  # type: ignore[union-attr]
                    accum_step = 0
                    global_step += 1

                    # Logging
                    if global_step % self.cfg.logging_steps == 0:
                        elapsed = time.perf_counter() - t_start
                        avg_loss = total_loss / (self.cfg.logging_steps * grad_accum)
                        lr = optimizer_for_scheduler.param_groups[0]["lr"] if optimizer_for_scheduler else self.cfg.learning_rate
                        log.info(
                            "Step %d/%d | loss=%.4f | lr=%.2e | %.1fs",
                            global_step, total_steps, avg_loss, lr, elapsed,
                        )
                        total_loss = 0.0

                    # Checkpoint
                    if global_step % self.cfg.save_steps == 0:
                        self._save_checkpoint(f"step-{global_step}")

        log.info("Training loop finished: %d steps in %.1fs", global_step, time.perf_counter() - t_start)

    def _sft_step(self, batch: dict) -> torch.Tensor:
        """SFT 前向：标准 LM loss。"""
        outputs = self.model(**batch)  # type: ignore[operator]
        return outputs.loss

    def _dpo_step(self, batch: dict) -> torch.Tensor:
        """DPO 前向：简化 DPO loss。

        完整的 DPO loss 需要 reference model，这里实现核心逻辑。
        生产环境建议使用 TRL 的 DPOTrainer。
        """
        # Extract log-probs for chosen and rejected
        with torch.no_grad():
            chosen_out = self.model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
            )
            rejected_out = self.model(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"],
            )

        chosen_logps = self._compute_logp(chosen_out.logits, batch["chosen_input_ids"])
        rejected_logps = self._compute_logp(rejected_out.logits, batch["rejected_input_ids"])

        beta = 0.1
        logits = chosen_logps - rejected_logps
        loss = -torch.nn.functional.logsigmoid(beta * logits).mean()
        return loss

    @staticmethod
    def _compute_logp(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算序列的 log-probability。"""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        per_token_loss = per_token_loss.view(shift_labels.shape)
        return -per_token_loss.sum(dim=-1)

    def _save_checkpoint(self, tag: str) -> None:
        """保存 checkpoint。"""
        save_dir = os.path.join(self.cfg.output_dir, f"checkpoint-{tag}")
        os.makedirs(save_dir, exist_ok=True)

        # 保存 LoRA adapter 或完整模型
        if self.cfg.finetuning_type == "lora" and hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_dir)  # type: ignore[union-attr]
        elif hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_dir)  # type: ignore[union-attr]

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_dir)

        log.info("Checkpoint saved to %s", save_dir)
