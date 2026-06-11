"""M04 冒烟测试 —— 在 server2 上验证 DeepSpeed/accelerate 后端初始化。

验证项:
  1. DeepSpeedBackend 初始化（ZeRO3 + 小模型）
  2. DeepSpeed backward/step/zero_grad 流程
  3. AccelerateBackend 初始化 + 训练 loop
  4. MegatronBackend 正确触发 NotImplementedError
  5. 后端工厂 build_backend 端到端
"""

from __future__ import annotations

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 确保项目在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---- 简单 mock 模型 ----
class TinyModel(nn.Module):
    def __init__(self, vocab=1000, dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.linear = nn.Linear(dim, vocab)

    def forward(self, x, labels=None):
        h = self.embed(x)
        if h.dim() == 3:
            h = h.mean(dim=1)
        logits = self.linear(h)
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            return logits, loss
        return logits


def make_dummy_data(batch_size=4, seq_len=16, num_batches=3):
    """生成假的 token id 数据。"""
    return DataLoader(
        TensorDataset(
            torch.randint(0, 1000, (batch_size * num_batches, seq_len)),
            torch.randint(0, 1000, (batch_size * num_batches,)),
        ),
        batch_size=batch_size,
    )


# ============================================================
# 测试 1: DeepSpeedBackend 初始化
# ============================================================
def test_deepspeed_init():
    """验证 DeepSpeed ZeRO3 后端是否能正常初始化（单卡模式）。"""
    from m_trainer.backends.base import TrainerConfig
    from m_trainer.backends.deepspeed import DeepSpeedBackend

    # 单卡模式下也要设置分布式环境变量，否则 DeepSpeed 会尝试 MPI 初始化
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")

    logger.info("=" * 60)
    logger.info("测试 1: DeepSpeedBackend 初始化 + 训练 loop")
    logger.info("=" * 60)

    model = TinyModel().cuda()
    config = TrainerConfig(
        distributed_backend="deepspeed",
        per_device_batch_size=2,
        gradient_accumulation_steps=4,
        bf16=False,  # 不支持 bf16 的卡用 fp16
        world_size=1,
        deepspeed_config={
            "fp16": {"enabled": True},
        },
    )

    backend = DeepSpeedBackend()
    wrapped_model, engine = backend.init(model, optimizer=None, config=config)
    logger.info("  ✓ DeepSpeed 初始化成功")

    dataloader = make_dummy_data(batch_size=2, num_batches=2)
    dataloader = backend.prepare_dataloader(dataloader)

    for step, (x, labels) in enumerate(dataloader):
        x, labels = x.cuda(), labels.cuda()
        logits, loss = wrapped_model(x, labels=labels)
        logger.info(f"  Step {step}: loss={loss.item():.4f}")
        backend.backward(loss)
        backend.step()
        backend.zero_grad()

    logger.info("  ✓ DeepSpeed 训练 loop 完成")
    logger.info("  smoke test 1 通过 ✅\n")
    return True


# ============================================================
# 测试 2: AccelerateBackend 初始化
# ============================================================
def test_accelerate_init():
    """验证 accelerate 后端是否能正常初始化。"""
    from m_trainer.backends.base import TrainerConfig
    from m_trainer.backends.accelerate import AccelerateBackend

    logger.info("=" * 60)
    logger.info("测试 2: AccelerateBackend 初始化 + 训练 loop")
    logger.info("=" * 60)

    model = TinyModel()
    config = TrainerConfig(
        distributed_backend="accelerate",
        per_device_batch_size=2,
        gradient_accumulation_steps=1,
        bf16=False,
        world_size=1,
    )

    backend = AccelerateBackend()
    wrapped_model, wrapped_optimizer = backend.init(
        model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
        config=config,
    )
    logger.info("  ✓ Accelerate 初始化成功")

    dataloader = make_dummy_data(batch_size=2, num_batches=2)
    dataloader = backend.prepare_dataloader(dataloader)

    for step, (x, labels) in enumerate(dataloader):
        logits, loss = wrapped_model(x, labels=labels)
        logger.info(f"  Step {step}: loss={loss.item():.4f}")
        backend.backward(loss)
        backend.step()
        backend.zero_grad()

    logger.info("  ✓ Accelerate 训练 loop 完成")
    logger.info("  smoke test 2 通过 ✅\n")
    return True


# ============================================================
# 测试 3: MegatronBackend NotImplementedError
# ============================================================
def test_megatron_not_implemented():
    """验证 Megatron 后端在 1.5B 以下模型正确抛出 NotImplementedError。"""
    from m_trainer.backends.base import TrainerConfig
    from m_trainer.backends.megatron import MegatronBackend

    logger.info("=" * 60)
    logger.info("测试 3: MegatronBackend 正确触发 NotImplementedError")
    logger.info("=" * 60)

    model = TinyModel()
    config = TrainerConfig(distributed_backend="megatron", world_size=1)
    backend = MegatronBackend()

    try:
        backend.init(model, optimizer=None, config=config)
        logger.error("  ✗ 预期 NotImplementedError 但未抛出!")
        return False
    except NotImplementedError as e:
        logger.info(f"  ✓ 正确抛出 NotImplementedError: {e}")
        logger.info("  smoke test 3 通过 ✅\n")
        return True


# ============================================================
# 测试 4: 工厂端到端
# ============================================================
def test_factory_e2e():
    """验证 build_backend 工厂端到端工作。"""
    from m_trainer.backends.base import TrainerConfig
    from m_trainer.factory import build_backend

    logger.info("=" * 60)
    logger.info("测试 4: 工厂 build_backend 端到端")
    logger.info("=" * 60)

    for backend_name in ["deepspeed", "accelerate", "megatron"]:
        config = TrainerConfig(
            distributed_backend=backend_name,
            bf16=False,
            deepspeed_config={"fp16": {"enabled": True}} if backend_name == "deepspeed" else {},
        )
        backend = build_backend(config)
        logger.info(f"  ✓ build_backend('{backend_name}'): {type(backend).__name__}")

    logger.info("  smoke test 4 通过 ✅\n")
    return True


# ============================================================
# 主流程
# ============================================================
def main():
    logger.info("M04 Smoke Test Suite")
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    results = {}

    # 测试 4: 工厂（纯 Python 测试，不依赖 GPU）
    results["factory"] = test_factory_e2e()

    # 测试 2: accelerate（CPU 也可跑）
    results["accelerate"] = test_accelerate_init()

    # 测试 1: DeepSpeed（需要 CUDA）
    if torch.cuda.is_available():
        results["deepspeed"] = test_deepspeed_init()
    else:
        logger.warning("⚠ CUDA 不可用，跳过 DeepSpeed 冒烟测试")

    # 测试 3: Megatron（纯逻辑）
    results["megatron"] = test_megatron_not_implemented()

    # 汇总
    logger.info("=" * 60)
    logger.info("Smoke Test 汇总")
    logger.info("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("\n🎉 所有 M04 冒烟测试通过!")
    else:
        logger.error("\n❌ 有测试失败，请检查日志")
        sys.exit(1)


if __name__ == "__main__":
    main()
