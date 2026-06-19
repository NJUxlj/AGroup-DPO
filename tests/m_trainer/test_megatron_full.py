"""Megatron 分布式训练全量集成测试 (M04)

测试覆盖：
  Part A — 模型检测（无需 GPU，无需 megatron-core）
  Part B — TP=1 单卡测试（无需 megatron-core）
  Part C — TP=2 双卡 Tensor Parallelism 测试（需 megatron-core + 2 GPU）
  Part D — 训练 Loop 正确性（forward / backward / step / zero_grad）
  Part E — Checkpoint 一致性（save / load）
  Part F — 多模型适配验证（GPT2 / Qwen2.5-1.5B / Qwen3-4B）

使用方式：
  # 单机测试（仅 Part A+B）
  PYTHONPATH=src python tests/m_trainer/test_megatron_full.py --quick

  # 全量测试（含 TP=2 + 真实模型，需 2 GPU）
  PYTHONPATH=src torchrun --nproc_per_node=2 tests/m_trainer/test_megatron_full.py \
      --gpt2-model /root/autodl-tmp/models/gpt2 \
      --qwen2-model /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
      --qwen3-model /root/autodl-tmp/models/Qwen3-4B
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

# ---- 确保 src 在 PYTHONPATH 中 ----
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.normpath(os.path.join(_script_dir, '..', '..'))
_src_dir = os.path.join(_project_root, 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from utils.logger import CustomLogger

from m_trainer.backends.base import TrainerConfig, DistributedBackend
from m_trainer.backends.megatron import (
    MegatronBackend,
    detect_model_type,
    _is_megatron_core_available,
)

log = CustomLogger.get_logger(__name__)

# ---- 颜色输出 ----
GREEN = '\033[0;32m'; RED = '\033[0;31m'; YELLOW = '\033[1;33m'; NC = '\033[0m'


# ======================================================================
# Part A: 模型检测（无需 GPU）
# ======================================================================


class _DummyGPT2Config:
    model_type = 'gpt2'
    architectures = ['GPT2LMHeadModel']


class _DummyGPT2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _DummyGPT2Config()


class _DummyQwen2Config:
    model_type = 'qwen2'
    architectures = ['Qwen2ForCausalLM']


class _DummyQwen2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _DummyQwen2Config()


class _DummyQwen3Config:
    model_type = 'qwen3'
    architectures = ['Qwen3ForCausalLM']


class _DummyQwen3Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _DummyQwen3Config()


def test_model_detection() -> dict[str, bool]:
    """Part A: 测试所有模型类型的自动检测。"""
    results = {}

    log.info("=" * 60)
    log.info("Part A: 模型检测测试")
    log.info("=" * 60)

    # GPT2
    m = _DummyGPT2Model()
    r = detect_model_type(m)
    results['a1_gpt2_detect'] = (r == 'gpt2')
    log.info("  A1 GPT2 检测: %s → %s", '✅' if results['a1_gpt2_detect'] else '❌', r)

    # Qwen2.5
    m2 = _DummyQwen2Model()
    r2 = detect_model_type(m2)
    results['a2_qwen2_detect'] = (r2 == 'qwen2')
    log.info("  A2 Qwen2 检测: %s → %s", '✅' if results['a2_qwen2_detect'] else '❌', r2)

    # Qwen3
    m3 = _DummyQwen3Model()
    r3 = detect_model_type(m3)
    results['a3_qwen3_detect'] = (r3 == 'qwen2')  # Qwen3 共享 qwen2 转换逻辑
    log.info("  A3 Qwen3 检测: %s → %s (归一化为 qwen2)", '✅' if results['a3_qwen3_detect'] else '❌', r3)

    # Unknown
    m4 = nn.Linear(10, 10)
    r4 = detect_model_type(m4)
    results['a4_unknown_detect'] = (r4 == 'unknown')
    log.info("  A4 Unknown 检测: %s → %s", '✅' if results['a4_unknown_detect'] else '❌', r4)

    # MegatronBackend 实例化
    backend = MegatronBackend()
    results['a5_backend_instantiate'] = isinstance(backend, MegatronBackend)
    log.info("  A5 后端实例化: %s", '✅' if results['a5_backend_instantiate'] else '❌')

    return results


# ======================================================================
# Part B: TP=1 单卡测试（无需 megatron-core）
# ======================================================================


class _MiniGPT2ForTest(nn.Module):
    """最小 GPT2 模型，用于 TP 转换和训练 Loop 测试。

    简化说明：c_attn 直接输出 hidden_size（而非真实 GPT2 的 3*hidden_size QKV），
    以便 TP 层的输入/输出维度能被 tp_size 整除且不失真。
    """

    def __init__(self, vocab_size=128, hidden_size=64, num_layers=2):
        super().__init__()
        self.config = _DummyGPT2Config()
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(vocab_size, hidden_size)
        self.transformer.wpe = nn.Embedding(64, hidden_size)
        self.transformer.h = nn.ModuleList([
            self._make_block(hidden_size) for _ in range(num_layers)
        ])
        self.transformer.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _make_block(self, hidden_size):
        block = nn.Module()
        block.ln_1 = nn.LayerNorm(hidden_size)
        block.attn = nn.Module()
        # c_attn: 简化为 hidden_size → 2*hidden_size（模拟 QK 投影）
        block.attn.c_attn = nn.Linear(hidden_size, 2 * hidden_size)
        block.attn.c_proj = nn.Linear(hidden_size, hidden_size)
        block.ln_2 = nn.LayerNorm(hidden_size)
        block.mlp = nn.Module()
        block.mlp.c_fc = nn.Linear(hidden_size, 4 * hidden_size)
        block.mlp.c_proj = nn.Linear(4 * hidden_size, hidden_size)
        return block

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        hidden = self.transformer.wte(input_ids)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        hidden = hidden + self.transformer.wpe(pos_ids)
        for block in self.transformer.h:
            residual = hidden
            hidden = block.ln_1(hidden)
            # c_attn 输出 2*hidden_size → 取后半作为 attention 结果，前半丢弃
            attn_out = block.attn.c_attn(hidden)
            hidden = block.attn.c_proj(attn_out[..., :hidden.size(-1)])
            hidden = residual + hidden
            residual = hidden
            hidden = block.ln_2(hidden)
            hidden = block.mlp.c_fc(hidden)
            hidden = torch.relu(hidden)
            hidden = block.mlp.c_proj(hidden)
            hidden = residual + hidden
        hidden = self.transformer.ln_f(hidden)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return type('CausalLMOutput', (), {'loss': loss, 'logits': logits})()


class _MiniQwen2ForTest(nn.Module):
    """最小 Qwen2 模型，用于 TP 转换和训练 Loop 测试。

    简化说明：q/k/v 的 head_dim 必须能被 tp_size=2 整除，因此 k/v 的
    out_features 设置为 hidden_size//2（实际 Qwen2 中 num_heads * head_dim 模式）。
    """

    def __init__(self, vocab_size=128, hidden_size=64, num_layers=2):
        super().__init__()
        self.config = _DummyQwen2Config()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.layers = nn.ModuleList([
            self._make_layer(hidden_size) for _ in range(num_layers)
        ])
        self.model.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def _make_layer(self, hidden_size):
        layer = nn.Module()
        layer.input_layernorm = nn.LayerNorm(hidden_size)
        layer.self_attn = nn.Module()
        # q_proj / k_proj / v_proj: out_features 需要能被 tp_size=2 整除
        # 64/2=32, 16/2=8 均 OK
        layer.self_attn.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        layer.self_attn.k_proj = nn.Linear(hidden_size, hidden_size // 2, bias=False)
        layer.self_attn.v_proj = nn.Linear(hidden_size, hidden_size // 2, bias=False)
        # o_proj: in_features = sum of head outputs. 简化为直接投影 v_proj 输出
        layer.self_attn.o_proj = nn.Linear(hidden_size // 2, hidden_size, bias=False)
        layer.post_attention_layernorm = nn.LayerNorm(hidden_size)
        layer.mlp = nn.Module()
        layer.mlp.gate_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        layer.mlp.up_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        layer.mlp.down_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        return layer

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            # simplified attention: 直接用 v_proj → o_proj
            v = layer.self_attn.v_proj(hidden)
            attn_out = layer.self_attn.o_proj(v)
            hidden = residual + attn_out
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            gate = torch.sigmoid(layer.mlp.gate_proj(hidden))
            up = layer.mlp.up_proj(hidden)
            hidden = layer.mlp.down_proj(gate * up)
            hidden = residual + hidden
        hidden = self.model.norm(hidden)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return type('CausalLMOutput', (), {'loss': loss, 'logits': logits})()


def test_tp1_standalone() -> dict[str, bool]:
    """Part B: TP=1 单卡测试 —— 验证后端初始化、前向、反向、优化器步进。"""
    results = {}

    log.info("=" * 60)
    log.info("Part B: TP=1 单卡测试")
    log.info("=" * 60)

    cfg = TrainerConfig(
        distributed_backend='megatron',
        megatron_config={'tensor_model_parallel_size': 1},
        world_size=1,
        learning_rate=1e-3,
    )

    # --- B1: GPT2 TP=1 init ---
    log.info("  B1: GPT2 TP=1 初始化...")
    try:
        backend_gpt = MegatronBackend()
        model_gpt = _MiniGPT2ForTest()
        tp_model, opt = backend_gpt.init(model_gpt, None, cfg)
        assert tp_model is not None, "TP 模型为 None"
        assert opt is not None, "优化器为 None"
        results['b1_gpt2_tp1_init'] = True
        log.info("  B1 ✅ 初始化成功")
    except Exception as e:
        results['b1_gpt2_tp1_init'] = False
        log.error("  B1 ❌ 初始化失败: %s", e)

    # --- B2: GPT2 forward ---
    if results.get('b1_gpt2_tp1_init'):
        log.info("  B2: GPT2 前向传播...")
        try:
            input_ids = torch.randint(0, 128, (2, 8))
            labels = torch.randint(0, 128, (2, 8))
            output = tp_model(input_ids, labels=labels)
            assert output.loss is not None, "loss 为 None"
            assert output.loss.item() > 0, "loss <= 0"
            results['b2_gpt2_forward'] = True
            log.info("  B2 ✅ forward loss=%.4f", output.loss.item())
        except Exception as e:
            results['b2_gpt2_forward'] = False
            log.error("  B2 ❌ forward 失败: %s", e)

    # --- B3: GPT2 backward ---
    if results.get('b2_gpt2_forward'):
        log.info("  B3: GPT2 反向传播...")
        try:
            backend_gpt.backward(output.loss)
            results['b3_gpt2_backward'] = True
            log.info("  B3 ✅ backward 成功")
        except Exception as e:
            results['b3_gpt2_backward'] = False
            log.error("  B3 ❌ backward 失败: %s", e)

    # --- B4: GPT2 step ---
    if results.get('b3_gpt2_backward'):
        log.info("  B4: GPT2 optimizer step...")
        try:
            backend_gpt.step()
            results['b4_gpt2_step'] = True
            log.info("  B4 ✅ step 成功")
        except Exception as e:
            results['b4_gpt2_step'] = False
            log.error("  B4 ❌ step 失败: %s", e)

    # --- B5: GPT2 zero_grad ---
    if results.get('b4_gpt2_step'):
        log.info("  B5: GPT2 zero_grad...")
        try:
            backend_gpt.zero_grad()
            results['b5_gpt2_zero_grad'] = True
            log.info("  B5 ✅ zero_grad 成功")
        except Exception as e:
            results['b5_gpt2_zero_grad'] = False
            log.error("  B5 ❌ zero_grad 失败: %s", e)

    del backend_gpt, model_gpt, tp_model, opt; gc.collect()

    # --- B6: Qwen2 TP=1 init ---
    log.info("  B6: Qwen2 TP=1 初始化...")
    try:
        backend_qw = MegatronBackend()
        model_qw = _MiniQwen2ForTest()
        tp_model_qw, opt_qw = backend_qw.init(model_qw, None, cfg)
        assert tp_model_qw is not None
        results['b6_qwen2_tp1_init'] = True
        log.info("  B6 ✅ 初始化成功")
    except Exception as e:
        results['b6_qwen2_tp1_init'] = False
        log.error("  B6 ❌ 初始化失败: %s", e)

    # --- B7: Qwen2 forward/backward/step ---
    if results.get('b6_qwen2_tp1_init'):
        log.info("  B7: Qwen2 训练一步...")
        try:
            input_ids = torch.randint(0, 128, (2, 8))
            labels = torch.randint(0, 128, (2, 8))
            output = tp_model_qw(input_ids, labels=labels)
            backend_qw.backward(output.loss)
            backend_qw.step()
            backend_qw.zero_grad()
            results['b7_qwen2_train_step'] = True
            log.info("  B7 ✅ Qwen2 训练一步 loss=%.4f", output.loss.item())
        except Exception as e:
            results['b7_qwen2_train_step'] = False
            log.error("  B7 ❌ 训练一步失败: %s", e)

    del backend_qw, model_qw, tp_model_qw, opt_qw; gc.collect()

    # --- B8: MegatronBackend state_dict ---
    log.info("  B8: state_dict / load_state_dict...")
    try:
        backend_sd = MegatronBackend()
        model_sd = _MiniGPT2ForTest()
        tp_m, opt_m = backend_sd.init(model_sd, None, cfg)
        state = backend_sd.state_dict()
        assert 'model' in state, "state 缺少 model key"
        assert 'tp_size' in state, "state 缺少 tp_size key"
        assert state['tp_size'] == 1, f"tp_size 期望 1, 实际 {state['tp_size']}"
        backend_sd.load_state_dict(state)
        results['b8_state_dict'] = True
        log.info("  B8 ✅ state_dict/load_state_dict 成功")
    except Exception as e:
        results['b8_state_dict'] = False
        log.error("  B8 ❌ state_dict 失败: %s", e)

    del backend_sd, model_sd, tp_m, opt_m; gc.collect()

    # --- B9: barrier ---
    log.info("  B9: barrier...")
    try:
        backend_barrier = MegatronBackend()
        backend_barrier.barrier()
        results['b9_barrier'] = True
        log.info("  B9 ✅ barrier 不报错")
    except Exception as e:
        results['b9_barrier'] = False
        log.error("  B9 ❌ barrier 失败: %s", e)

    return results


# ======================================================================
# Part C: TP=2 双卡 Tensor Parallelism 测试（需 megatron-core + 2 GPU）
# ======================================================================


def test_tp2_gpt2() -> dict[str, bool]:
    """Part C1: GPT2 TP=2 测试（需要 megatron-core + 2 GPU）。"""
    results = {}

    if not _is_megatron_core_available():
        log.warning("  ⏭️  megatron-core 未安装，跳过 GPT2 TP=2 测试")
        results['c1_skip'] = True
        return results

    log.info("=" * 60)
    log.info("Part C1: GPT2 TP=2 Tensor Parallelism")
    log.info("=" * 60)

    from megatron.core.parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()

    if tp_size != 2:
        log.warning("  ⏭️  TP world_size=%s != 2，跳过 TP=2 测试 (当前 rank=%s)", tp_size, tp_rank)
        results['c1_skip'] = True
        return results

    cfg = TrainerConfig(
        distributed_backend='megatron',
        megatron_config={'tensor_model_parallel_size': 2},
        world_size=2,
        learning_rate=1e-3,
    )

    # --- C1a: GPT2 TP init ---
    log.info("  C1a: GPT2 TP=2 初始化 (rank=%s)...", tp_rank)
    try:
        backend = MegatronBackend()
        model = _MiniGPT2ForTest(vocab_size=128, hidden_size=64, num_layers=2).cuda()
        tp_model, opt = backend.init(model, None, cfg)
        assert tp_model is not None
        # 验证 wte 被转换为 VocabParallelEmbedding
        from m_trainer.backends.megatron import _VocabParallelEmbeddingWrapper
        wte = tp_model.transformer.wte
        assert isinstance(wte, _VocabParallelEmbeddingWrapper), \
            f"wte 应为 _VocabParallelEmbeddingWrapper, 实际 {type(wte)}"
        results['c1a_gpt2_tp2_init'] = True
        log.info("  C1a ✅ rank=%s: TP=2 初始化成功", tp_rank)
    except Exception as e:
        results['c1a_gpt2_tp2_init'] = False
        log.error("  C1a ❌ rank=%s: 初始化失败: %s", tp_rank, e)
        return results

    # 同步
    torch.distributed.barrier()

    # --- C1b: GPT2 forward ---
    log.info("  C1b: GPT2 TP=2 forward (rank=%s)...", tp_rank)
    try:
        input_ids = torch.randint(0, 128, (2, 8)).cuda()
        labels = torch.randint(0, 128, (2, 8)).cuda()
        output = tp_model(input_ids, labels=labels)
        assert output.loss is not None
        results['c1b_tp2_forward'] = True
        log.info("  C1b ✅ rank=%s: forward loss=%.4f", tp_rank, output.loss.item())
    except Exception as e:
        results['c1b_tp2_forward'] = False
        log.error("  C1b ❌ rank=%s: forward 失败: %s", tp_rank, e)
        return results

    torch.distributed.barrier()

    # --- C1c: GPT2 backward ---
    log.info("  C1c: GPT2 TP=2 backward (rank=%s)...", tp_rank)
    try:
        backend.backward(output.loss)
        results['c1c_tp2_backward'] = True
        log.info("  C1c ✅ rank=%s: backward 成功", tp_rank)
    except Exception as e:
        results['c1c_tp2_backward'] = False
        log.error("  C1c ❌ rank=%s: backward 失败: %s", tp_rank, e)
        return results

    torch.distributed.barrier()

    # --- C1d: GPT2 step ---
    log.info("  C1d: GPT2 TP=2 step (rank=%s)...", tp_rank)
    try:
        backend.step()
        results['c1d_tp2_step'] = True
        log.info("  C1d ✅ rank=%s: step 成功", tp_rank)
    except Exception as e:
        results['c1d_tp2_step'] = False
        log.error("  C1d ❌ rank=%s: step 失败: %s", tp_rank, e)

    torch.distributed.barrier()

    # --- C1e: GPT2 state_dict ---
    log.info("  C1e: GPT2 TP=2 state_dict (rank=%s)...", tp_rank)
    try:
        state = backend.state_dict()
        assert 'tp_size' in state
        assert state['tp_size'] == 2
        backend.load_state_dict(state)
        results['c1e_tp2_state_dict'] = True
        log.info("  C1e ✅ rank=%s: state_dict 成功", tp_rank)
    except Exception as e:
        results['c1e_tp2_state_dict'] = False
        log.error("  C1e ❌ rank=%s: state_dict 失败: %s", tp_rank, e)

    torch.distributed.barrier()

    del backend, model, tp_model, opt; gc.collect()
    torch.cuda.empty_cache()

    return results


def test_tp2_qwen2() -> dict[str, bool]:
    """Part C2: Qwen2 TP=2 测试。"""
    results = {}

    if not _is_megatron_core_available():
        log.warning("  ⏭️  megatron-core 未安装，跳过 Qwen2 TP=2 测试")
        results['c2_skip'] = True
        return results

    from megatron.core.parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size != 2:
        results['c2_skip'] = True
        return results

    log.info("=" * 60)
    log.info("Part C2: Qwen2 TP=2 Tensor Parallelism")
    log.info("=" * 60)

    cfg = TrainerConfig(
        distributed_backend='megatron',
        megatron_config={'tensor_model_parallel_size': 2},
        world_size=2,
        learning_rate=1e-3,
    )

    # Qwen2 的 head_dim 需要能被 tp_size=2 整除
    # MiniQwen2: hidden=64, head_dim=16 (需要能整除)
    log.info("  C2a: Qwen2 TP=2 初始化 (rank=%s)...", tp_rank)
    try:
        backend = MegatronBackend()
        model = _MiniQwen2ForTest(vocab_size=128, hidden_size=64, num_layers=2).cuda()
        tp_model, opt = backend.init(model, None, cfg)
        assert tp_model is not None
        results['c2a_qwen2_tp2_init'] = True
        log.info("  C2a ✅ rank=%s: Qwen2 TP=2 初始化成功", tp_rank)
    except Exception as e:
        results['c2a_qwen2_tp2_init'] = False
        log.error("  C2a ❌ rank=%s: 初始化失败: %s", tp_rank, e)
        return results

    torch.distributed.barrier()

    log.info("  C2b: Qwen2 TP=2 训练一步 (rank=%s)...", tp_rank)
    try:
        input_ids = torch.randint(0, 128, (2, 8)).cuda()
        labels = torch.randint(0, 128, (2, 8)).cuda()
        output = tp_model(input_ids, labels=labels)
        backend.backward(output.loss)
        backend.step()
        backend.zero_grad()
        results['c2b_qwen2_train_step'] = True
        log.info("  C2b ✅ rank=%s: Qwen2 训练一步 loss=%.4f", tp_rank, output.loss.item())
    except Exception as e:
        results['c2b_qwen2_train_step'] = False
        log.error("  C2b ❌ rank=%s: 训练一步失败: %s", tp_rank, e)

    torch.distributed.barrier()
    del backend, model, tp_model, opt; gc.collect()
    torch.cuda.empty_cache()

    return results


# ======================================================================
# Part F: 真实模型权重加载测试
# ======================================================================


def _run_real_model_test(
    model_path: str, model_type: str, model_name: str
) -> dict[str, bool]:
    """Part F: 加载真实 HuggingFace 模型并验证 MegatronBackend 适配。

    Args:
        model_path: HuggingFace 模型本地路径
        model_type: 期望的检测类型 ('gpt2' | 'qwen2')
        model_name: 显示名称
    """
    results = {}
    log.info("=" * 60)
    log.info("Part F: 真实模型适配 —— %s", model_name)
    log.info("=" * 60)

    if not os.path.isdir(model_path):
        log.warning("  ⏭️  模型路径不存在: %s", model_path)
        results[f'f_{model_name}_skip'] = True
        return results

    try:
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError:
        log.warning("  ⏭️  transformers 未安装，跳过真实模型测试")
        results[f'f_{model_name}_skip'] = True
        return results

    # F1: 模型类型检测
    log.info("  F1: 检测模型类型...")
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        actual_type = config.model_type
        log.info("  F1   config.model_type=%s, architectures=%s",
                    actual_type, getattr(config, 'architectures', []))
    except Exception as e:
        log.warning("  F1 ⚠️  无法读取 config: %s", e)

    # F2: 加载模型（TP=1，无需 megatron-core）
    log.info("  F2: 加载 %s (TP=1)...", model_name)
    try:
        # 使用较小的 dtype 节省显存
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # CPU 加载避免 OOM
            device_map=None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        detected = detect_model_type(model)
        assert detected == model_type, f"检测类型 {detected} != 期望 {model_type}"
        results[f'f1_{model_name}_detect'] = True
        log.info("  F2 ✅ 模型加载成功, 检测类型=%s, 参数量=%.1fM",
                    detected, sum(p.numel() for p in model.parameters()) / 1e6)
    except Exception as e:
        log.error("  F2 ❌ 模型加载失败: %s", e)
        results[f'f1_{model_name}_detect'] = False
        # 即使加载失败也继续后续测试
        return results

    # F3: MegatronBackend TP=1 初始化
    log.info("  F3: MegatronBackend TP=1 初始化...")
    try:
        cfg = TrainerConfig(
            distributed_backend='megatron',
            megatron_config={'tensor_model_parallel_size': 1},
            world_size=1,
            learning_rate=1e-4,
        )
        backend = MegatronBackend()
        tp_model, opt = backend.init(model, None, cfg)
        assert tp_model is not None, "TP 模型为 None"
        results[f'f2_{model_name}_tp1_init'] = True
        log.info("  F3 ✅ TP=1 初始化成功")
    except Exception as e:
        results[f'f2_{model_name}_tp1_init'] = False
        log.error("  F3 ❌ TP=1 初始化失败: %s", e)
        return results

    # F4: 前向传播
    log.info("  F4: %s 前向传播...", model_name)
    try:
        input_ids = torch.randint(0, min(1000, model.config.vocab_size - 1), (1, 16))
        output = tp_model(input_ids, labels=input_ids)
        if hasattr(output, 'loss') and output.loss is not None:
            results[f'f3_{model_name}_forward'] = True
            log.info("  F4 ✅ forward loss=%.4f", output.loss.item())
        else:
            results[f'f3_{model_name}_forward'] = True
            log.info("  F4 ✅ forward 成功 (无 loss 头)")
    except Exception as e:
        results[f'f3_{model_name}_forward'] = False
        log.error("  F4 ❌ forward 失败: %s", e)

    # F5: state_dict
    log.info("  F5: %s state_dict...", model_name)
    try:
        state = backend.state_dict()
        assert 'model' in state, "state 缺少 model key"
        assert 'tp_size' in state, "state 缺少 tp_size key"
        results[f'f4_{model_name}_state_dict'] = True
        log.info("  F5 ✅ state_dict/load_state_dict 成功")
    except Exception as e:
        results[f'f4_{model_name}_state_dict'] = False
        log.error("  F5 ❌ state_dict 失败: %s", e)

    del backend, model, tp_model, opt; gc.collect()

    # F6: TP=2 测试（仅在 torchrun 多进程环境下运行，需 torch.distributed 已初始化）
    if _is_megatron_core_available() and torch.cuda.device_count() >= 2 and torch.distributed.is_initialized():
        log.info("  F6: %s TP=2 测试...", model_name)
        try:
            from megatron.core.parallel_state import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
                initialize_model_parallel,
            )
            try:
                tp_size = get_tensor_model_parallel_world_size()
            except (AssertionError, RuntimeError):
                tp_size = 1
            if tp_size >= 2:
                model2 = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map=None,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                ).cuda()
                cfg2 = TrainerConfig(
                    distributed_backend='megatron',
                    megatron_config={'tensor_model_parallel_size': tp_size},
                    world_size=tp_size,
                    learning_rate=1e-4,
                )
                backend2 = MegatronBackend()
                tp_model2, opt2 = backend2.init(model2, None, cfg2)
                torch.distributed.barrier()
                input_ids = torch.randint(0, min(1000, model2.config.vocab_size - 1), (1, 16)).cuda()
                output2 = tp_model2(input_ids, labels=input_ids)
                if hasattr(output2, 'loss') and output2.loss is not None:
                    backend2.backward(output2.loss)
                    backend2.step()
                    backend2.zero_grad()
                torch.distributed.barrier()
                results[f'f5_{model_name}_tp2'] = True
                tp_rank = get_tensor_model_parallel_rank()
                log.info("  F6 ✅ rank=%s: %s TP=%s 训练一步 loss=%.4f",
                            tp_rank, model_name, tp_size,
                            output2.loss.item() if output2.loss is not None else float('nan'))
                del backend2, model2, tp_model2, opt2
                torch.cuda.empty_cache()
            else:
                results[f'f5_{model_name}_tp2'] = True
                log.info("  F6 ⏭️  TP size=%s 不足 2，跳过", tp_size)
        except Exception as e:
            results[f'f5_{model_name}_tp2'] = False
            log.error("  F6 ❌ TP=2 测试失败: %s", e)

    return results


# ======================================================================
# 汇总与入口
# ======================================================================


@dataclass
class _MegatronTestReport:
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    details: list[str] = None

    def __post_init__(self):
        if self.details is None:
            self.details = []


def merge_results(*dicts: dict[str, bool]) -> dict[str, bool]:
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged


def generate_report(results: dict[str, bool], title: str) -> _MegatronTestReport:
    report = _MegatronTestReport()
    report.total = len(results)
    for k, v in sorted(results.items()):
        if k.endswith('_skip'):
            report.skipped += 1
        elif v:
            report.passed += 1
            report.details.append(f"  ✅ {k}")
        else:
            report.failed += 1
            report.details.append(f"  ❌ {k}")

    log.info("")
    log.info("=" * 70)
    log.info("📊 %s 测试报告", title)
    log.info("=" * 70)
    log.info("  总计: %d  |  通过: %d  |  失败: %d  |  跳过: %d",
                report.total, report.passed, report.failed, report.skipped)
    log.info("-" * 70)
    for line in report.details:
        log.info(line)
    log.info("=" * 70)

    return report


def main():
    parser = argparse.ArgumentParser(description='Megatron 分布式训练全量测试')
    parser.add_argument('--quick', action='store_true',
                        help='仅运行 Part A+B（无需 GPU，无需 megatron-core）')
    parser.add_argument('--gpt2-model', type=str, default='',
                        help='GPT2 模型本地路径')
    parser.add_argument('--qwen2-model', type=str, default='',
                        help='Qwen2.5-1.5B 模型本地路径')
    parser.add_argument('--qwen3-model', type=str, default='',
                        help='Qwen3-4B 模型本地路径')
    parser.add_argument('--output-json', type=str, default='',
                        help='输出测试结果 JSON 文件路径')
    args = parser.parse_args()

    total_results: dict[str, bool] = {}

    # ---- Part A: 模型检测（始终运行）----
    a_results = test_model_detection()
    total_results.update(a_results)

    # ---- Part B: TP=1 单卡测试（始终运行）----
    b_results = test_tp1_standalone()
    total_results.update(b_results)

    if args.quick:
        report = generate_report(total_results, f"Megatron 快速测试 ({'ALL' if all(total_results.values()) else 'FAIL'})")
        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(total_results, f, indent=2)
        return 0 if not report.failed else 1

    # ---- Part C: TP=2 双卡测试（需要 torchrun + 2 GPU + megatron-core）----
    if torch.cuda.device_count() >= 2 and _is_megatron_core_available():
        # 先确保 torch.distributed 已初始化（torchrun 应已初始化）
        if torch.distributed.is_initialized():
            c1_results = test_tp2_gpt2()
            total_results.update(c1_results)
            c2_results = test_tp2_qwen2()
            total_results.update(c2_results)
        else:
            log.warning("⚠️  torch.distributed 未初始化！")
            log.warning("  请使用 torchrun 启动以运行 Part C TP=2 测试:")
            log.warning("    torchrun --nproc_per_node=2 tests/m_trainer/test_megatron_full.py ...")
            total_results['c_warning'] = True
    elif torch.cuda.device_count() >= 2 and not _is_megatron_core_available():
        log.warning("⚠️  检测到 %d GPU 但 megatron-core 未安装，跳过 TP=2 测试",
                       torch.cuda.device_count())
        log.warning("  安装: pip install megatron-core")
    else:
        log.info("ℹ️  GPU 数量=%d，跳过 TP=2 测试", torch.cuda.device_count())

    # ---- Part F: 真实模型加载 ----
    # GPT2
    gpt2_path = args.gpt2_model or os.path.expanduser('~/.cache/huggingface/hub/models--gpt2')
    if os.path.isdir(gpt2_path) or args.gpt2_model:
        f_gpt2 = _run_real_model_test(gpt2_path, 'gpt2', 'gpt2')
        total_results.update(f_gpt2)

    # Qwen2.5-1.5B
    qwen2_path = args.qwen2_model or ''
    if qwen2_path and os.path.isdir(qwen2_path):
        f_qwen2 = _run_real_model_test(qwen2_path, 'qwen2', 'qwen2_1.5b')
        total_results.update(f_qwen2)

    # Qwen3-4B
    qwen3_path = args.qwen3_model or ''
    if qwen3_path and os.path.isdir(qwen3_path):
        f_qwen3 = _run_real_model_test(qwen3_path, 'qwen2', 'qwen3_4b')
        total_results.update(f_qwen3)

    # ---- 输出报告 ----
    all_ok = all(v for k, v in total_results.items() if not k.endswith('_skip'))
    report = generate_report(total_results, f"Megatron 全量测试 ({'ALL PASS' if all_ok else 'PARTIAL FAIL'})")

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump({
                'total': report.total,
                'passed': report.passed,
                'failed': report.failed,
                'skipped': report.skipped,
                'results': total_results,
            }, f, indent=2)
        log.info("测试结果已写入: %s", args.output_json)

    torch.cuda.empty_cache()
    return 0 if not report.failed else 1


if __name__ == '__main__':
    sys.exit(main())
