"""
CustomTrainer — 自研训练器 (M03/M04)

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
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.logger import CustomLogger

from .backends.base import TrainerConfig
from .backends.optimizer_factory import AdamWFactory
from .callbacks import MetricsLogger, StepMetrics, compute_grad_norm, current_gpu_mem_gb
from .factory import build_backend

log = CustomLogger.get_logger(__name__)

IM_END = ""
QWEN_DEFAULT_SYSTEM = "你是AI财保助理，需严格依据条款作答。"

# LLaMA-Factory YAML 字段 → TrainingConfig 字段
_YAML_FIELD_ALIASES: dict[str, str] = {
    "per_device_train_batch_size": "per_device_batch_size",
    "dpo_beta": "dpo_beta",
    "pref_beta": "dpo_beta",
    "dpo_loss_type": "dpo_loss_type",
    "pref_loss": "dpo_loss_type",
}


# ---------------------------------------------------------------------------
# TrainingConfig — 扩展 TrainerConfig，加入模型/数据字段
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """完整训练配置，兼容 LLaMA-Factory YAML 格式。"""

    # ---- 模型 ----
    model_name_or_path: str = ""
    trust_remote_code: bool = False
    ref_model_name_or_path: str = ""

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
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # ---- DPO ----
    dpo_beta: float = 0.1
    dpo_loss_type: str = "sigmoid"
    dpo_max_prompt_length: int = 1024

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

    # ---- 可观测 (M03 § 3.6) ----
    report_to: str = "tensorboard"  # none / tensorboard / wandb / all
    logging_dir: str = ""
    wandb_project: str = "agroup-dpo"
    run_name: str = ""

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


def _load_deepspeed_config(raw: dict[str, Any]) -> dict[str, Any]:
    """从 YAML 顶层 deepspeed 字段加载 ZeRO 配置。"""
    ds_path = raw.get("deepspeed")
    if not ds_path:
        return {}
    path = Path(ds_path)
    if not path.is_file():
        log.warning("DeepSpeed config not found: %s", ds_path)
        return {}
    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    return loaded if isinstance(loaded, dict) else {}


def _normalize_yaml_config(raw: dict[str, Any]) -> dict[str, Any]:
    """将 LLaMA-Factory YAML 规范化为 TrainingConfig 字段。"""
    normalized: dict[str, Any] = {}

    for key, value in raw.items():
        if key == "trainer" and isinstance(value, dict):
            normalized.update(value)
            continue
        if key == "deepspeed":
            continue
        target = _YAML_FIELD_ALIASES.get(key, key)
        normalized[target] = value

    ds_cfg = _load_deepspeed_config(raw)
    if ds_cfg:
        normalized["deepspeed_config"] = ds_cfg

    report_to = normalized.get("report_to")
    if isinstance(report_to, list):
        normalized["report_to"] = ",".join(str(item) for item in report_to)

    return normalized


def resolve_dataset_path(dataset_dir: str, dataset_name: str) -> str:
    """解析数据集路径，优先读取 dataset_info.json（M02/M03 契约）。"""
    info_path = os.path.join(dataset_dir, "dataset_info.json")
    if os.path.isfile(info_path):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        entry = info.get(dataset_name)
        if isinstance(entry, dict):
            file_name = entry.get("file_name", f"{dataset_name}.jsonl")
            candidate = os.path.join(dataset_dir, file_name)
            if os.path.isfile(candidate):
                return candidate

    direct = os.path.join(dataset_dir, f"{dataset_name}.jsonl")
    if os.path.isfile(direct):
        return direct

    raise FileNotFoundError(
        f"Dataset '{dataset_name}' not found under {dataset_dir}. "
        f"Expected dataset_info.json mapping or {direct}"
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


def _format_qwen_sft_text(
    instruction: str,
    inp: str,
    output: str,
    system: str = QWEN_DEFAULT_SYSTEM,
) -> tuple[str, str]:
    """返回 (prompt, full_text)，prompt 部分在 SFT loss 中 mask。"""
    user_content = f"{instruction}\n{inp}" if inp else instruction
    prompt = (
        f"<|im_start|>system\n{system}{IM_END}\n"
        f"<|im_start|>user\n{user_content}{IM_END}\n"
        f"<|im_start|>assistant\n"
    )
    return prompt, prompt + output + IM_END


def _sft_collate(batch: list[dict], tokenizer, cutoff_len: int) -> dict[str, torch.Tensor]:
    """将 SFT 样本 tokenize 为 input_ids + labels（prompt 部分 mask 为 -100）。"""
    input_ids_list: list[list[int]] = []
    labels_list: list[list[int]] = []

    for item in batch:
        instruction = item.get("instruction", "")
        inp = item.get("input", "")
        output = item.get("output", "")
        system = item.get("system", QWEN_DEFAULT_SYSTEM)
        prompt, full_text = _format_qwen_sft_text(instruction, inp, output, system)

        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        encoded = tokenizer(
            full_text,
            truncation=True,
            max_length=cutoff_len,
            add_special_tokens=False,
        )
        ids = encoded["input_ids"]
        labels = ids.copy()
        mask_len = min(prompt_len, len(labels))
        for i in range(mask_len):
            labels[i] = -100

        input_ids_list.append(ids)
        labels_list.append(labels)

    max_len = max(len(x) for x in input_ids_list)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, (ids, lbls) in enumerate(zip(input_ids_list, labels_list)):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[i, : len(lbls)] = torch.tensor(lbls, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def _build_dpo_sequence(
    prompt: str,
    response: str,
    tokenizer,
    cutoff_len: int,
    max_prompt_length: int,
) -> tuple[list[int], list[int]]:
    """构造 DPO 序列：labels 仅在 response token 上保留 id，prompt 部分为 -100。"""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if not response_ids:
        response_ids = [tokenizer.eos_token_id or 0]

    if len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids

    if len(input_ids) > cutoff_len:
        overflow = len(input_ids) - cutoff_len
        input_ids = input_ids[overflow:]
        labels = labels[overflow:]

    return input_ids, labels


def _dpo_collate(
    batch: list[dict],
    tokenizer,
    cutoff_len: int,
    max_prompt_length: int,
) -> dict[str, torch.Tensor]:
    """将 DPO 样本 tokenize 为 prompt+response 拼接序列。"""
    chosen_ids_list: list[list[int]] = []
    chosen_labels_list: list[list[int]] = []
    rejected_ids_list: list[list[int]] = []
    rejected_labels_list: list[list[int]] = []

    for item in batch:
        prompt = item.get("prompt", "")
        chosen_ids, chosen_labels = _build_dpo_sequence(
            prompt, item.get("chosen", ""), tokenizer, cutoff_len, max_prompt_length
        )
        rejected_ids, rejected_labels = _build_dpo_sequence(
            prompt, item.get("rejected", ""), tokenizer, cutoff_len, max_prompt_length
        )
        chosen_ids_list.append(chosen_ids)
        chosen_labels_list.append(chosen_labels)
        rejected_ids_list.append(rejected_ids)
        rejected_labels_list.append(rejected_labels)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def _pad_sequences(
        ids_list: list[list[int]],
        labels_list: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = max(len(x) for x in ids_list)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, (ids, lbls) in enumerate(zip(ids_list, labels_list)):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            labels[i, : len(lbls)] = torch.tensor(lbls, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
        return input_ids, labels, attention_mask

    chosen_input_ids, chosen_labels, chosen_attention_mask = _pad_sequences(
        chosen_ids_list, chosen_labels_list
    )
    rejected_input_ids, rejected_labels, rejected_attention_mask = _pad_sequences(
        rejected_ids_list, rejected_labels_list
    )

    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_labels": chosen_labels,
        "chosen_attention_mask": chosen_attention_mask,
        "rejected_input_ids": rejected_input_ids,
        "rejected_labels": rejected_labels,
        "rejected_attention_mask": rejected_attention_mask,
    }


# ---------------------------------------------------------------------------
# CustomTrainer
# ---------------------------------------------------------------------------


class CustomTrainer:
    """自研训练器，封装完整的训练流程。"""

    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.model: Optional[nn.Module] = None
        self.ref_model: Optional[nn.Module] = None
        self.tokenizer = None
        self.backend = None
        self.optimizer: Any = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_yaml(cls, path: str) -> "CustomTrainer":
        """从 YAML 配置文件加载训练器（兼容 LLaMA-Factory 字段名）。"""
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        normalized = _normalize_yaml_config(raw)
        known = {f.name for f in TrainingConfig.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in normalized.items() if k in known}
        return cls(TrainingConfig(**kwargs))

    def train(self) -> None:
        """执行完整训练流程。"""
        log.info("=" * 60)
        log.info("CustomTrainer: stage=%s, backend=%s", self.cfg.stage, self.cfg.distributed_backend)
        log.info("  model: %s", self.cfg.model_name_or_path)
        log.info("  output: %s", self.cfg.output_dir)
        log.info("=" * 60)

        torch.manual_seed(self.cfg.seed)

        self._load_model_and_tokenizer()
        dataloader = self._load_data()
        self._init_backend()
        dataloader = self.backend.prepare_dataloader(dataloader)  # type: ignore[union-attr]

        metrics_logger = MetricsLogger.from_config(self.cfg)
        metrics_logger.on_train_begin(self.cfg)
        try:
            self._run_training_loop(dataloader, metrics_logger)
        finally:
            metrics_logger.on_train_end()

        self._save_checkpoint("final")
        log.info("Training complete. Model saved to %s", self.cfg.output_dir)

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

        if self.cfg.stage == "dpo" and self.cfg.finetuning_type != "lora":
            ref_path = self.cfg.ref_model_name_or_path or self.cfg.model_name_or_path
            log.info("Loading frozen reference model from %s", ref_path)
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                ref_path,
                torch_dtype=torch_dtype,
                trust_remote_code=self.cfg.trust_remote_code,
            )
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
            self.ref_model.to(self._device)

        if self.cfg.finetuning_type == "lora":
            self._apply_lora()

        self.model.to(self._device)
        self.model.train()
        log.info("Model loaded: %s params", sum(p.numel() for p in self.model.parameters()))

    def _apply_lora(self) -> None:
        """对模型应用 LoRA adapter（M03 策略 1：DPO 阶段重新注入 LoRA）。"""
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            log.warning("peft not installed, skipping LoRA")
            return

        target_modules = [m.strip() for m in self.cfg.lora_target.split(",") if m.strip()]
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
        data_path = resolve_dataset_path(self.cfg.dataset_dir, self.cfg.dataset)
        dataset = JSONLinesDataset(data_path)
        log.info("Loaded %d samples from %s", len(dataset), data_path)

        if self.cfg.stage == "sft":
            collate_fn = lambda b: _sft_collate(b, self.tokenizer, self.cfg.cutoff_len)
        else:
            collate_fn = lambda b: _dpo_collate(
                b,
                self.tokenizer,
                self.cfg.cutoff_len,
                self.cfg.dpo_max_prompt_length,
            )

        return DataLoader(
            dataset,
            batch_size=self.cfg.per_device_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def _init_backend(self) -> None:
        """初始化分布式后端。"""
        trainer_cfg = self.cfg.to_trainer_config()

        if self.cfg.distributed_backend == "deepspeed":
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

    def _run_training_loop(self, dataloader: DataLoader, metrics_logger: MetricsLogger) -> None:
        """主训练循环。"""
        total_steps = (
            self.cfg.max_steps
            if self.cfg.max_steps > 0
            else int(len(dataloader) * self.cfg.num_train_epochs)
        )
        grad_accum = self.cfg.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)
        backend_handles_accum = self.backend.handles_gradient_accumulation()  # type: ignore[union-attr]

        optimizer_for_scheduler = self.optimizer
        if optimizer_for_scheduler is None and self.backend is not None:
            optimizer_for_scheduler = getattr(self.backend, "_optimizer", None)

        scheduler = None
        if self.cfg.distributed_backend != "deepspeed" and optimizer_for_scheduler is not None:
            if self.cfg.lr_scheduler_type == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer_for_scheduler, T_max=max(total_steps - warmup_steps, 1)
                )

        log.info(
            "Training: total_steps=%d, grad_accum=%d, warmup=%d, backend_accum=%s",
            total_steps,
            grad_accum,
            warmup_steps,
            backend_handles_accum,
        )

        global_step = 0
        total_loss = 0.0
        dpo_margin_sum = 0.0
        dpo_chosen_sum = 0.0
        dpo_rejected_sum = 0.0
        t_start = time.perf_counter()
        accum_step = 0
        saved_checkpoints: list[str] = []

        for epoch in range(int(math.ceil(self.cfg.num_train_epochs))):
            if global_step >= total_steps:
                break

            for batch in dataloader:
                if global_step >= total_steps:
                    break

                batch = {k: v.to(self._device) for k, v in batch.items()}
                batch_size = batch["input_ids"].size(0) if self.cfg.stage == "sft" else batch["chosen_input_ids"].size(0)
                metrics_logger.record_batch_samples(batch_size)

                if self.cfg.stage == "sft":
                    loss, step_metrics = self._sft_step(batch)
                else:
                    loss, step_metrics = self._dpo_step(batch)
                    dpo_margin_sum += step_metrics.get("reward_margin", 0.0)
                    dpo_chosen_sum += step_metrics.get("chosen_reward", 0.0)
                    dpo_rejected_sum += step_metrics.get("rejected_reward", 0.0)

                if not backend_handles_accum:
                    loss = loss / grad_accum

                self.backend.backward(loss)  # type: ignore[union-attr]
                accum_step += 1
                total_loss += loss.item() * (1 if backend_handles_accum else grad_accum)

                should_step = backend_handles_accum or accum_step >= grad_accum
                if not should_step:
                    continue

                grad_norm = (
                    compute_grad_norm(self._trainable_module())
                    if (global_step + 1) % self.cfg.logging_steps == 0
                    else None
                )

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

                if global_step % self.cfg.logging_steps == 0:
                    elapsed = time.perf_counter() - t_start
                    avg_loss = total_loss / self.cfg.logging_steps
                    lr = (
                        optimizer_for_scheduler.param_groups[0]["lr"]
                        if optimizer_for_scheduler
                        else self.cfg.learning_rate
                    )
                    log.info(
                        "Step %d/%d | loss=%.4f | lr=%.2e | %.1fs",
                        global_step,
                        total_steps,
                        avg_loss,
                        lr,
                        elapsed,
                    )
                    step_log = StepMetrics(
                        loss=avg_loss,
                        learning_rate=lr,
                        global_step=global_step,
                        grad_norm=grad_norm,
                        gpu_mem_gb=current_gpu_mem_gb(),
                    )
                    if self.cfg.stage == "dpo":
                        window = self.cfg.logging_steps
                        step_log.reward_margin = dpo_margin_sum / window
                        step_log.chosen_reward = dpo_chosen_sum / window
                        step_log.rejected_reward = dpo_rejected_sum / window
                        dpo_margin_sum = 0.0
                        dpo_chosen_sum = 0.0
                        dpo_rejected_sum = 0.0
                    metrics_logger.on_log(step_log)
                    total_loss = 0.0

                if global_step % self.cfg.save_steps == 0:
                    tag = f"step-{global_step}"
                    self._save_checkpoint(tag)
                    saved_checkpoints.append(tag)
                    self._prune_checkpoints(saved_checkpoints)

        log.info("Training loop finished: %d steps in %.1fs", global_step, time.perf_counter() - t_start)

    def _trainable_module(self) -> nn.Module:
        """返回用于 grad_norm 等诊断的底层模块。"""
        module = self.model
        if module is None:
            raise RuntimeError("Model is not initialized")
        return getattr(module, "module", module)

    def _sft_step(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        """SFT 前向：标准 LM loss（仅 assistant 部分参与）。"""
        outputs = self.model(**batch)  # type: ignore[operator]
        return outputs.loss, {}

    def _dpo_step(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        """DPO 前向：标准 DPO loss（含 reference model 对比）。"""
        policy_chosen = self._compute_sequence_logp(
            self.model,
            batch["chosen_input_ids"],
            batch["chosen_labels"],
            batch["chosen_attention_mask"],
            reference=False,
        )
        policy_rejected = self._compute_sequence_logp(
            self.model,
            batch["rejected_input_ids"],
            batch["rejected_labels"],
            batch["rejected_attention_mask"],
            reference=False,
        )

        ref_model = self._reference_model()
        ref_chosen = self._compute_sequence_logp(
            ref_model,
            batch["chosen_input_ids"],
            batch["chosen_labels"],
            batch["chosen_attention_mask"],
            reference=True,
        )
        ref_rejected = self._compute_sequence_logp(
            ref_model,
            batch["rejected_input_ids"],
            batch["rejected_labels"],
            batch["rejected_attention_mask"],
            reference=True,
        )

        pi_logratios = policy_chosen - policy_rejected
        ref_logratios = ref_chosen - ref_rejected
        logits = pi_logratios - ref_logratios

        beta = self.cfg.dpo_beta
        if self.cfg.dpo_loss_type == "sigmoid":
            loss = -F.logsigmoid(beta * logits).mean()
        else:
            raise ValueError(f"Unsupported dpo_loss_type: {self.cfg.dpo_loss_type}")

        chosen_reward = float((beta * (policy_chosen - ref_chosen)).mean().item())
        rejected_reward = float((beta * (policy_rejected - ref_rejected)).mean().item())
        return loss, {
            "chosen_reward": chosen_reward,
            "rejected_reward": rejected_reward,
            "reward_margin": chosen_reward - rejected_reward,
        }

    def _reference_model(self) -> nn.Module:
        """返回 DPO reference model。

        LoRA 模式下通过 disable_adapter 复用基座权重作为 π_ref，避免双份全量模型。
        """
        if self.ref_model is not None:
            return self.ref_model
        if hasattr(self.model, "disable_adapter"):
            return self.model
        raise RuntimeError(
            "DPO requires a reference model. Use finetuning_type=lora or set ref_model_name_or_path."
        )

    def _compute_sequence_logp(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        reference: bool = False,
    ) -> torch.Tensor:
        """计算 response token 上的 log-probability 之和。"""
        from contextlib import nullcontext

        use_adapter_disable = (
            reference
            and self.ref_model is None
            and model is self.model
            and hasattr(model, "disable_adapter")
        )
        ctx = model.disable_adapter() if use_adapter_disable else nullcontext()
        with ctx:
            if reference:
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.clamp(min=0).unsqueeze(-1),
        ).squeeze(-1)

        loss_mask = (shift_labels != -100).float()
        return (token_log_probs * loss_mask).sum(dim=-1)

    def _save_checkpoint(self, tag: str) -> None:
        """保存 checkpoint。"""
        save_dir = os.path.join(self.cfg.output_dir, f"checkpoint-{tag}")
        os.makedirs(save_dir, exist_ok=True)

        model_to_save = getattr(self.model, "module", self.model)
        if hasattr(model_to_save, "save_pretrained"):
            model_to_save.save_pretrained(save_dir)  # type: ignore[union-attr]
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_dir)

        log.info("Checkpoint saved to %s", save_dir)

    def _prune_checkpoints(self, saved: list[str]) -> None:
        """保留最近 save_total_limit 个 checkpoint（M03 § 7.5）。"""
        limit = self.cfg.save_total_limit
        if limit <= 0 or len(saved) <= limit:
            return

        to_remove = saved[:-limit]
        for tag in to_remove:
            path = os.path.join(self.cfg.output_dir, f"checkpoint-{tag}")
            if os.path.isdir(path):
                import shutil

                shutil.rmtree(path, ignore_errors=True)
                log.info("Removed old checkpoint: %s", path)
        del saved[: len(to_remove)]
