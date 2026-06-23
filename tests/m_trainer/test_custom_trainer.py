"""CustomTrainer 与 M03 契约一致性测试。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from m_trainer.backends.base import TrainerConfig
from m_trainer.backends.deepspeed import build_zero3_config
from m_trainer.custom_trainer import (
    JSONLinesDataset,
    TrainingConfig,
    _build_dpo_sequence,
    _dpo_collate,
    _format_qwen_sft_text,
    _normalize_yaml_config,
    _sft_collate,
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

    def test_deepspeed_config_loaded(self):
        raw = {"deepspeed": "configs/deepspeed/zero3.json", "stage": "sft"}
        normalized = _normalize_yaml_config(raw)
        assert "deepspeed_config" in normalized
        assert normalized["deepspeed_config"]["zero_optimization"]["stage"] == 3

    def test_report_to_list_joined(self):
        raw = {"report_to": ["tensorboard", "wandb"]}
        normalized = _normalize_yaml_config(raw)
        assert normalized["report_to"] == "tensorboard,wandb"

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

    def test_from_project_sft_yaml(self):
        from m_trainer.custom_trainer import CustomTrainer

        cfg_path = PROJECT_ROOT / "configs" / "train_lora_qwen2_5_1_5b_insurance.yaml"
        trainer = CustomTrainer.from_yaml(str(cfg_path))
        assert trainer.cfg.stage == "sft"
        assert trainer.cfg.finetuning_type == "lora"
        assert trainer.cfg.lora_rank == 16
        assert trainer.cfg.num_train_epochs == 3.0


class TestTrainingConfig:
    def test_to_trainer_config(self):
        cfg = TrainingConfig(
            distributed_backend="accelerate",
            per_device_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=3e-5,
        )
        tc = cfg.to_trainer_config()
        assert isinstance(tc, TrainerConfig)
        assert tc.distributed_backend == "accelerate"
        assert tc.per_device_batch_size == 4
        assert tc.gradient_accumulation_steps == 2
        assert tc.learning_rate == 3e-5


class TestDatasetResolution:
    def test_resolve_via_dataset_info(self):
        path = resolve_dataset_path(
            str(PROJECT_ROOT / "data" / "insurance"),
            "insurance_dpo_v1.2",
        )
        assert path.endswith("dpo_train_v1.2.jsonl")
        assert Path(path).is_file()

    def test_resolve_sft_dataset(self):
        path = resolve_dataset_path(
            str(PROJECT_ROOT / "data" / "insurance"),
            "insurance_sft_v1",
        )
        assert path.endswith("insurance_sft_v1.jsonl")
        assert Path(path).is_file()

    def test_resolve_smoke_dataset(self):
        path = resolve_dataset_path(
            str(PROJECT_ROOT / "data" / "smoke"),
            "smoke_sft",
        )
        assert path.endswith("smoke_sft.jsonl")

    def test_missing_dataset_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_dataset_path(str(PROJECT_ROOT / "data" / "smoke"), "nonexistent_xyz")


class TestJSONLinesDataset:
    def test_loads_samples(self):
        path = PROJECT_ROOT / "data" / "smoke" / "smoke_sft.jsonl"
        ds = JSONLinesDataset(str(path))
        assert len(ds) > 0
        sample = ds[0]
        assert "instruction" in sample
        assert "output" in sample

    def test_empty_file_raises(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="No samples"):
            JSONLinesDataset(str(empty))


class TestQwenFormatting:
    def test_sft_prompt_masks_assistant_only(self):
        prompt, full = _format_qwen_sft_text(
            "请回答", "什么是保险？", "保险是一种风险管理工具。"
        )
        assert full.startswith(prompt)
        assert "assistant" in prompt
        assert "保险是一种风险管理工具" in full

    def test_sft_collate_masks_prompt_tokens(self):
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 0

        def fake_encode(text, **kwargs):
            ids = list(range(len(text.split())))
            return {"input_ids": ids}

        tokenizer.side_effect = fake_encode

        batch = [
            {"instruction": "Q1", "input": "", "output": "A1"},
            {"instruction": "Q2", "input": "ctx", "output": "A2"},
        ]
        out = _sft_collate(batch, tokenizer, cutoff_len=512)
        assert "input_ids" in out
        assert "labels" in out
        assert "attention_mask" in out
        assert out["input_ids"].shape[0] == 2

    def test_dpo_sequence_labels_only_on_response(self):
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2

        def fake_encode(text, **kwargs):
            n = max(1, len(text.split()))
            return {"input_ids": list(range(n))}

        tokenizer.side_effect = fake_encode

        ids, labels = _build_dpo_sequence("hello world", "good answer", tokenizer, 128, 64)
        assert len(ids) == len(labels)
        assert all(l == -100 for l in labels[: len(ids) - 2])
        assert labels[-2:] == ids[-2:]

    def test_dpo_collate_shapes(self):
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2

        def fake_encode(text, **kwargs):
            n = max(1, len(text.split()))
            return {"input_ids": list(range(n))}

        tokenizer.side_effect = fake_encode

        batch = [
            {"prompt": "Q?", "chosen": "yes good", "rejected": "no bad"},
        ]
        out = _dpo_collate(batch, tokenizer, cutoff_len=128, max_prompt_length=32)
        for key in (
            "chosen_input_ids",
            "chosen_labels",
            "chosen_attention_mask",
            "rejected_input_ids",
            "rejected_labels",
            "rejected_attention_mask",
        ):
            assert key in out
            assert out[key].dim() == 2


class TestDeepSpeedOptimizerConfig:
    def test_optimizer_in_zero3_config(self):
        cfg = TrainerConfig(learning_rate=5e-6)
        ds = build_zero3_config(cfg)
        assert ds["optimizer"]["type"] == "AdamW"
        assert ds["optimizer"]["params"]["lr"] == 5e-6


class TestSftLossGrad:
    class _FakeCausalLM(nn.Module):
        def __init__(self, vocab: int = 32, hidden: int = 16):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.proj = nn.Linear(hidden, vocab)

        def forward(self, input_ids, labels=None, attention_mask=None):
            x = self.embed(input_ids)
            logits = self.proj(x)
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            return type("Out", (), {"logits": logits, "loss": loss})()

    def test_sft_step_produces_grad(self):
        from m_trainer.custom_trainer import CustomTrainer

        trainer = CustomTrainer(TrainingConfig(stage="sft"))
        trainer.model = self._FakeCausalLM()
        trainer.ref_model = None

        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[-100, -100, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }

        loss, metrics = trainer._sft_step(batch)
        loss.backward()

        assert loss.requires_grad
        assert metrics == {}
        assert trainer.model.proj.weight.grad is not None


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

        loss, metrics = trainer._dpo_step(batch)
        loss.backward()

        assert loss.requires_grad
        assert trainer.model.proj.weight.grad is not None
        assert "chosen_reward" in metrics
        assert "rejected_reward" in metrics
        assert "reward_margin" in metrics

    def test_dpo_unsupported_loss_type(self):
        from m_trainer.custom_trainer import CustomTrainer

        trainer = CustomTrainer(TrainingConfig(stage="dpo", dpo_loss_type="hinge"))
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

        with pytest.raises(ValueError, match="Unsupported dpo_loss_type"):
            trainer._dpo_step(batch)


class TestSetupDistributed:
    def test_setup_distributed_accelerate_smoke(self):
        pytest.importorskip("accelerate")
        model = nn.Linear(4, 2)
        cfg = TrainerConfig(distributed_backend="accelerate", learning_rate=1e-4)
        wrapped_model, optimizer, backend = setup_distributed(model, cfg)
        assert wrapped_model is not None
        assert backend is not None
        assert backend.handles_gradient_accumulation() is False
        # setup_distributed 传入 optimizer=None，accelerate 后端不自动创建 optimizer
        assert optimizer is None
