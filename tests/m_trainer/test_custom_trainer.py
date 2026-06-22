"""CustomTrainer 与 M03 契约一致性测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from m_trainer.backends.base import TrainerConfig
from m_trainer.backends.deepspeed import build_zero3_config
from m_trainer.custom_trainer import (
    TrainingConfig,
    _normalize_yaml_config,
    resolve_dataset_path,
)
from m_trainer.factory import setup_distributed


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TestYamlNormalization:
    def test_per_device_train_batch_size_alias(self):
        raw = {"per_device_train_batch_size": 4, "stage": "sft"}
        normalized = _normalize_yaml_config(raw)
        assert normalized["per_device_batch_size"] == 4

    def test_pref_beta_alias(self):
        raw = {"pref_beta": 0.2, "pref_loss": "sigmoid"}
        normalized = _normalize_yaml_config(raw)
        assert normalized["dpo_beta"] == 0.2
        assert normalized["dpo_loss_type"] == "sigmoid"

    def test_trainer_section_merge(self):
        raw = {
            "stage": "dpo",
            "trainer": {"distributed_backend": "fsdp", "learning_rate": 1e-5},
        }
        normalized = _normalize_yaml_config(raw)
        assert normalized["distributed_backend"] == "fsdp"
        assert normalized["learning_rate"] == 1e-5
        assert normalized["stage"] == "dpo"

    def test_from_project_dpo_yaml(self):
        from m_trainer.custom_trainer import CustomTrainer

        cfg_path = PROJECT_ROOT / "configs" / "train_dpo_qwen2_5_1_5b_insurance.yaml"
        trainer = CustomTrainer.from_yaml(str(cfg_path))
        assert trainer.cfg.stage == "dpo"
        assert trainer.cfg.per_device_batch_size == 2
        assert trainer.cfg.gradient_accumulation_steps == 16
        assert trainer.cfg.dpo_beta == 0.1
        assert trainer.cfg.lora_rank == 16
        assert trainer.cfg.report_to == "tensorboard"


class TestDatasetResolution:
    def test_resolve_via_dataset_info(self):
        path = resolve_dataset_path(
            str(PROJECT_ROOT / "data" / "insurance"),
            "insurance_dpo_v1.2",
        )
        assert path.endswith("dpo_train_v1.2.jsonl")
        assert Path(path).is_file()


class TestDeepSpeedOptimizerConfig:
    def test_optimizer_in_zero3_config(self):
        cfg = TrainerConfig(learning_rate=5e-6)
        ds = build_zero3_config(cfg)
        assert ds["optimizer"]["type"] == "AdamW"
        assert ds["optimizer"]["params"]["lr"] == 5e-6


class TestDpoLossGrad:
    class _FakePeftModel(nn.Module):
        def __init__(self, vocab: int = 32, hidden: int = 16):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.proj = nn.Linear(hidden, vocab)
            self._adapter_enabled = True

        def disable_adapter(self):
            model = self

            class _Ctx:
                def __enter__(self_inner):
                    model._adapter_enabled = False
                    return model

                def __exit__(self_inner, *args):
                    model._adapter_enabled = True

            return _Ctx()

        def forward(self, input_ids, attention_mask=None):
            x = self.embed(input_ids)
            logits = self.proj(x)
            return type("Out", (), {"logits": logits})()

    def test_dpo_step_produces_grad(self):
        from m_trainer.custom_trainer import CustomTrainer

        trainer = CustomTrainer(TrainingConfig(stage="dpo", dpo_beta=0.1))
        trainer.model = self._FakePeftModel()
        trainer.ref_model = None

        batch = {
            "chosen_input_ids": torch.tensor([[1, 2, 3, 4]]),
            "chosen_labels": torch.tensor([[-100, -100, 3, 4]]),
            "chosen_attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "rejected_input_ids": torch.tensor([[1, 2, 5, 6]]),
            "rejected_labels": torch.tensor([[-100, -100, 5, 6]]),
            "rejected_attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }

        loss, _ = trainer._dpo_step(batch)
        loss.backward()

        assert loss.requires_grad
        assert trainer.model.proj.weight.grad is not None


class TestSetupDistributed:
    def test_setup_distributed_accelerate_smoke(self):
        pytest.importorskip("accelerate")
        model = nn.Linear(4, 2)
        cfg = TrainerConfig(distributed_backend="accelerate", learning_rate=1e-4)
        wrapped_model, optimizer, backend = setup_distributed(model, cfg)
        assert wrapped_model is not None
        assert optimizer is not None
        assert backend.handles_gradient_accumulation() is False
