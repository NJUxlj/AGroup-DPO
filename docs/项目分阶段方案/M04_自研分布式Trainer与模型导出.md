# M04 自研分布式 Trainer 与模型导出

> 阶段编号：M04
> 阶段名称：自研分布式 Trainer 与模型导出
> 预估工期：5 人天
> 关联文档：[项目方案.md § 5.4 M-TRAINER](../项目方案.md) | [项目方案.md § 5.5 M-MERGE](../项目方案.md) | [项目方案.md § 3.3 分布式训练后端选型](../项目方案.md)
> 上游阶段：M01
> 下游阶段：M03、M05
> 对应功能需求：FR-06、FR-07

---

## 1. 阶段定位

M04 是项目的**框架层阶段**，目标是交付一套可插拔的分布式训练后端抽象（FR-07），让同一份训练配置可以在 DeepSpeed / FSDP / Megatron / accelerate 四种后端间无缝切换。

同时承担 LoRA 与基座合并导出（FR-06），为下游推理（M05）提供可直接被 vLLM / xinference 加载的 HF safetensors 模型。

与 M03 的关系：

| 关注点 | M03（LLaMA-Factory） | M04（自研 Trainer） |
|--------|---------------------|---------------------|
| 数据加载 | ✅ | ❌ |
| 模型加载与 LoRA 注入 | ✅ | ❌ |
| DPO loss 计算 | ✅ | ❌ |
| 分布式后端适配 | ❌ | ✅ |
| 优化器分片 | ❌ | ✅ |
| checkpoint 策略 | 与 HF 兼容 | ✅（统一 state_dict） |
| Merge 导出 | ❌ | ✅（M-MERGE 子模块） |

---

## 2. 阶段目标

### 2.1 业务目标

| 维度 | 目标 | 衡量方式 |
|------|------|----------|
| 后端可插拔 | 同一份配置在 4 种后端上跑通 | 4 份冒烟日志 |
| 接口稳定 | LLaMA-Factory 与自研 backend 无冲突 | 端到端训练日志 |
| 模型可部署 | merge 后模型可被 vLLM 加载 | merge + 推理冒烟 |
| 代码可维护 | 关键模块单测覆盖率 | ≥ 60% |

### 2.2 技术目标

| 维度 | 目标 | 衡量方式 |
|------|------|----------|
| 抽象清晰 | `DistributedBackend` 接口纯 Python ABC | 类型检查 |
| 注册机制 | 工厂模式 + 字符串注册 | 单元测试 |
| 配置驱动 | 仅通过 yaml 字段切换 | 切换 demo |
| Checkpoint | 同后端内 `state_dict` 可加载；跨后端需 `merge_and_unload` 后重新加载（详见 § 7.5 / § 8 风险） | 同后端 `load_state_dict` 冒烟 | 单元测试 |
| 合并正确 | merge 后模型权重数学等价 | PEFT 库自带验证 |

---

## 3. 核心任务

### 3.1 抽象接口设计（FR-07）

**`DistributedBackend` 抽象类**：

```python
# m_trainer/backends/base.py
from abc import ABC, abstractmethod
from typing import Any

class DistributedBackend(ABC):
    """所有分布式后端的统一接口"""

    @abstractmethod
    def init(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             config: "TrainerConfig") -> tuple[torch.nn.Module, torch.optim.Optimizer]:
        """初始化分布式环境，返回 (wrapped_model, wrapped_optimizer)"""

    @abstractmethod
    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """对模型应用分片/并行策略"""

    @abstractmethod
    def wrap_optimizer(self, model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        """应用 ZeRO / FSDP 优化器包装"""

    @abstractmethod
    def prepare_dataloader(self, dataloader: torch.utils.data.DataLoader
                          ) -> torch.utils.data.DataLoader:
        """分发 sampler / 注入 rank-aware collator"""

    @abstractmethod
    def barrier(self) -> None:
        """跨卡同步"""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """统一状态字典，支持断点续训"""

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        ...
```

**`ShardingStrategy` 分片策略**：

```python
class ShardingStrategy(ABC):
    """分片策略：决定哪些层/参数被分片"""
    @abstractmethod
    def apply(self, model: torch.nn.Module) -> torch.nn.Module: ...
```

**`OptimizerFactory` 优化器工厂**：

```python
class OptimizerFactory(ABC):
    """优化器工厂：根据后端选择合适的 optimizer 包装"""
    @abstractmethod
    def build(self, params, base_cfg: dict) -> torch.optim.Optimizer: ...
```

### 3.2 四种后端适配器实现

| 后端 | 适配器类 | 关键实现 |
|------|----------|----------|
| DeepSpeed ZeRO3 | `DeepSpeedBackend` | 调用 `deepspeed.initialize(model, optimizer, config=ds_config)`，根据 yaml 生成 `ds_config`（zero_optimization.stage=3, offload_optimizer=cpu 等） |
| PyTorch FSDP | `FSDPBackend` | 使用 `torch.distributed.fsdp.fully_shard`（PyTorch 2.4+）或 `FullyShardedDataParallel`，auto_wrap_policy 基于 `nn.TransformerBlock` size |
| Megatron-LM | `MegatronBackend` | 注入 tensor_model_parallel_size 与 pipeline_model_parallel_size；1.5B 暂不启用 TP，仅保留接口 |
| accelerate | `AccelerateBackend` | 使用 `accelerator.prepare(model, optimizer, dataloader)`；用于单机多卡调试与 CI |

**DeepSpeed ZeRO3 配置生成器**：

```python
# m_trainer/backends/deepspeed.py
def build_zero3_config(cfg: "TrainerConfig") -> dict:
    return {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
            },
        },
        "bf16": {"enabled": True},
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": cfg.per_device_batch_size,
        "train_batch_size": (
            cfg.per_device_batch_size
            * cfg.gradient_accumulation_steps
            * cfg.world_size
        ),
    }
```

**FSDP wrap policy**：

```python
# m_trainer/backends/fsdp.py
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

def build_fsdp_model(model: nn.Module, cfg: "TrainerConfig") -> nn.Module:
    wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={nn.TransformerEncoderLayer, nn.TransformerDecoderLayer},
    )
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )
    return fully_shard(model, mp_policy=mp_policy, auto_wrap_policy=wrap_policy)
```

**Megatron 接口预留**：

```python
# m_trainer/backends/megatron.py
class MegatronBackend(DistributedBackend):
    """Megatron-LM 适配器：1.5B 暂不启用 TP/PP，仅保留接口"""
    def init(self, model, optimizer, config):
        # TODO: 模型规模 > 7B 时启用 tensor_model_parallel_size
        raise NotImplementedError(
            "Megatron backend is reserved for future scaling to 7B+; "
            "1.5B training should use DeepSpeed or FSDP."
        )
```

### 3.3 工厂模式 + 配置驱动切换

```python
# m_trainer/factory.py
import importlib

BACKEND_REGISTRY = {
    "deepspeed":  "m_trainer.backends.deepspeed:DeepSpeedBackend",
    "fsdp":       "m_trainer.backends.fsdp:FSDPBackend",
    "megatron":   "m_trainer.backends.megatron:MegatronBackend",
    "accelerate": "m_trainer.backends.accelerate:AccelerateBackend",
}

def build_backend(cfg: "TrainerConfig") -> DistributedBackend:
    name = cfg.distributed_backend
    if name not in BACKEND_REGISTRY:
        raise ValueError(
            f"unsupported backend: {name}, "
            f"choose from {list(BACKEND_REGISTRY)}"
        )
    module_path, cls_name = BACKEND_REGISTRY[name].split(":")
    cls = importlib.import_module(module_path).__dict__[cls_name]
    return cls()
```

**yaml 配置示例**：

```yaml
trainer:
  distributed_backend: deepspeed   # fsdp | megatron | accelerate
  output_dir: runs/qwen2_5_dpo_v1.2

deepspeed:
  zero_optimization:
    stage: 3
    offload_optimizer: { device: cpu, pin_memory: true }
    offload_param: { device: cpu, pin_memory: true }
  bf16:
    enabled: true
  gradient_accumulation_steps: 16
  train_micro_batch_size_per_gpu: 2
```

切换时仅修改 `trainer.distributed_backend` 字段，无需改动业务代码。

### 3.4 与 LLaMA-Factory 的协作关系

```
[yaml 配置]
    │
    ▼
[m_trainer.factory.build_backend] ──▶ [DeepSpeed/FSDP/... Backend]
    │
    ▼
[LLaMA-Factory Trainer]  ◀── wrapped_model / wrapped_optimizer
    │
    ▼
[训练循环]  ◀── 自研回调：loss 监控、checkpoint、ZeRO3 状态保存
```

**集成方式（协作模式）**：

```python
# src/train.py（具体训练循环与 loss 计算见 M03 § 7.4；本节只展示 M04 backend 的协作封装）
from m_trainer.factory import build_backend

def setup_distributed(model: torch.nn.Module, cfg: "TrainerConfig"
                      ) -> tuple[torch.nn.Module, torch.optim.Optimizer, "DistributedBackend"]:
    """M04 提供：构建 backend 并完成 model/optimizer 的分布式包装。

    各后端的 optimizer 创建位置不同：
      - DeepSpeed ZeRO3：optimizer 由 `deepspeed.initialize` 内部创建并分片，
        本函数返回的 optimizer 实际为 `DeepSpeedEngine`。
      - FSDP：optimizer 由 backend 内部创建，并通过 `fully_shard` 与 model 绑定。
      - accelerate：optimizer 与 model 由 `accelerator.prepare` 统一包装。
      - Megatron：当前 1.5B 模型触发 `NotImplementedError`（见 3.2）。
    因此本函数**不接受外部 optimizer**，由 backend 负责创建与包装。
    """
    backend = build_backend(cfg)
    model, optimizer = backend.init(model, optimizer=None, config=cfg)
    return model, optimizer, backend


# M03 训练入口会按如下顺序调用（完整代码见 M03 § 7.4）：
#   1) LLaMA-Factory 加载模型 + 注入 LoRA（返回 model）
#   2) setup_distributed(model, cfg.trainer)  -> (model, optimizer, backend)
#   3) LLaMA-Factory Trainer.train(model=model, ...)
#   4) M04 backend.save_checkpoint / load_state_dict 由 LLaMA-Factory 回调触发
```

> **约定**：`optimizer=None` 是显式语义——表示 optimizer 由 backend 内部创建，业务方不应在外部构造后传入。DeepSpeed engine 不是 `torch.optim.Optimizer` 子类，但语义上等价于"已包装好的优化器 + 训练 step 入口"。

### 3.5 LoRA 与基座合并导出（FR-06）

**合并算法**（PEFT 库 `merge_and_unload` 实现）：

\[
W_{\mathrm{merged}} = W_{\mathrm{base}} + \frac{\alpha}{r} \cdot B \cdot A
\]

**导出 CLI**：

```bash
# 使用 LLaMA-Factory 自带的 merge 脚本
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --adapter_name_or_path saves/qwen2_5_1_5b/insurance_dpo_v1.2/lora \
    --template qwen \
    --finetuning_type lora \
    --export_dir merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \
    --export_size 5 \
    --export_device cpu \
    --export_legacy_format false
```

**产物目录结构**：

```
merged_models/qwen2_5_1_5b_insurance_dpo_v1.2/
├── config.json
├── generation_config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── merges.txt / vocab.json
```

**导出格式**：

| 格式 | 适用场景 | 是否默认 |
|------|----------|----------|
| HuggingFace safetensors（单文件夹） | 直接被 vLLM / xinference / HF transformers 加载 | ✅ 默认 |
| 合并后单文件 .bin（兼容老格式） | 兼容部分老推理框架 | 备选 |
| LoRA adapter 单独保留 | 仅保留增量参数，便于回滚 | 与 merge 产物并存 |

**Merge 模块代码骨架**：

```python
# m_merge/exporter.py
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def merge_and_export(
    base_model_path: str,
    adapter_path: str,
    export_dir: str,
    export_size: int = 5,
    export_device: str = "cpu",
):
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
    )
    peft_model = PeftModel.from_pretrained(base, adapter_path)
    merged = peft_model.merge_and_unload()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    merged.save_pretrained(export_dir, max_shard_size=f"{export_size}GB")
    tokenizer.save_pretrained(export_dir)
```

---

## 4. 交付物清单

| 编号 | 交付物 | 路径 | 说明 |
|------|--------|------|------|
| D-M04-01 | `DistributedBackend` 抽象类 | `m_trainer/backends/base.py` | ABC 接口 |
| D-M04-02 | `ShardingStrategy` 抽象类 | `m_trainer/backends/sharding.py` | 分片策略 |
| D-M04-03 | `OptimizerFactory` 抽象类 | `m_trainer/backends/optimizer_factory.py` | 优化器工厂 |
| D-M04-04 | DeepSpeed 适配器 | `m_trainer/backends/deepspeed.py` | ZeRO3 主路径 |
| D-M04-05 | FSDP 适配器 | `m_trainer/backends/fsdp.py` | PyTorch 原生 |
| D-M04-06 | Megatron 适配器 | `m_trainer/backends/megatron.py` | 接口预留 |
| D-M04-07 | accelerate 适配器 | `m_trainer/backends/accelerate.py` | 单机多卡调试 |
| D-M04-08 | 后端工厂 | `m_trainer/factory.py` | 注册 + 实例化 |
| D-M04-09 | 注册表常量 | `m_trainer/registry.py` | BACKEND_REGISTRY |
| D-M04-10 | 合并导出模块 | `m_merge/exporter.py` | merge_and_export 函数 |
| D-M04-11 | Merge CLI 入口 | `m_merge/cli.py` | 命令行封装 |
| D-M04-12 | 4 后端 yaml 模板 | `configs/backends/{deepspeed,fsdp,megatron,accelerate}.yaml` | 配置示例 |
| D-M04-13 | 4 后端冒烟测试 | `tests/m_trainer/test_*_backend.py` | 各 1 个 E2E |
| D-M04-14 | 单测 | `tests/m_trainer/*.py` | 接口与工厂测试 |
| D-M04-15 | 阶段文档 | `docs/项目分阶段方案/M04_自研Trainer与模型导出.md` | 本文件 |
| D-M04-16 | 模块 README | `m_trainer/README.md` | 使用说明 |

---

## 5. 验收标准

| 类别 | 验收项 | 量化阈值 | 检查方式 |
|------|--------|----------|----------|
| 多后端支持 | 同一份 yaml + 同一份数据集在 4 种后端上均能启动训练 | 4/4 PASS | 冒烟日志 |
| 后端切换 | 仅修改 yaml 字段即可切换 | 无需改业务代码 | 切换 demo |
| 接口稳定 | LLaMA-Factory 与自研 backend 协作无冲突 | M03 端到端训练通过 | 训练日志 |
| Checkpoint 兼容 | 同后端内 `state_dict` 可加载；跨后端需 `merge_and_unload` 后重新加载 | 同后端 `load_state_dict` 不报错 | 单元测试 |
| Merge 后兼容 | merge 后 HF safetensors 可在 4 种后端上重新加载并继续训练 | `from_pretrained` 不报错 | 单元测试 |
| Merge 正确 | merge 后模型推理正常 + 权重数学等价 | 与 PEFT 库自带验证一致 | merge + 推理冒烟 |
| 单测覆盖 | `m_trainer/` 模块单测覆盖率 | ≥ 60% | pytest-cov |
| 性能 | DeepSpeed ZeRO3 vs FSDP 训练 throughput | 差异 ≤ 20% | benchmark 脚本 |
| 文档 | `m_trainer/README.md` 完整 | 含 4 后端切换示例 | 人工 review |

---

## 6. 依赖关系

### 6.1 上游依赖

```
┌──────────┐
│   M01    │  基础设施（4 种分布式后端 layer 镜像）
└────┬─────┘
     ▼
┌──────────┐
│   M04    │  ← 本阶段
└──────────┘
```

- M01：`copaw-dpo-{deepspeed,fsdp,megatron,accelerate}:latest` 镜像就绪
- M01：2 卡 NVLink 通信可用

### 6.2 下游依赖

```
       ┌──────────┐
       │   M04    │  ← 本阶段
       └────┬─────┘
            ├──────────────────┐
            ▼                  ▼
       ┌──────────┐      ┌──────────┐
       │   M03    │      │   M05    │
       │ LoRA+DPO │      │ 推理+评测│
       └──────────┘      └──────────┘
```

- M03 通过 M04 切换到 FSDP / accelerate 等备选后端。
- M05 消费 M04 产出的 `merged_models/...` 作为推理输入。

### 6.3 与 M03 的耦合（并行推进）

```
       ┌──────────┐      ┌──────────┐
       │   M03    │◀────▶│   M04    │
       │ LLaMA-F. │      │ 自研 Trnr│
       └──────────┘      └──────────┘
              ▲                │
              └──── 接口契约 ──┘
```

- M03 默认走 DeepSpeed ZeRO3（LLaMA-Factory 原生支持）。
- M03 通过 M04 提供的统一后端抽象切换到 FSDP。
- M04 的 `DistributedBackend` 接口需在 M03 启动 FSDP 备路径前稳定（建议先定接口，再并行实现）。

### 6.4 跨阶段接口契约

| 接口 | 契约 | 调用方 |
|------|------|--------|
| `build_backend(cfg)` | 返回实现 `DistributedBackend` ABC 的实例 | M03 → 本阶段 |
| `merged_models/.../` | HF safetensors 单文件夹，可被 vLLM 加载 | M05 → 本阶段 |
| yaml `trainer.distributed_backend` | 字符串枚举 `deepspeed/fsdp/megatron/accelerate` | 业务方 → 本阶段 |
| `BACKEND_REGISTRY` | 可被业务方注册自定义后端（高级用法） | 扩展方 → 本阶段 |

---

## 7. 详细技术规范

### 7.1 后端选型决策

| 后端 | 优势 | 劣势 | 适用场景 | 2×A100 可行性 |
|------|------|------|----------|---------------|
| DeepSpeed (ZeRO3) | 优化器状态/梯度/参数全分片，显存节省显著 | 配置复杂度中等 | 默认主选 | 完全可行 |
| PyTorch FSDP | PyTorch 原生，与 HF Trainer 集成度好 | 早期版本通信开销大 | DeepSpeed 的备选 | 完全可行 |
| Megatron-LM | 张量并行 + 流水线并行，性能极强 | 配置复杂，依赖 NVIDIA 生态 | 模型规模 > 7B 时 | 1.5B 略显重，但需保留接口 |
| HuggingFace accelerate | 上手快，适合小规模实验 | 大规模训练性能不如前三者 | 单卡/小规模调试 | 完全可行 |

**最终决策**：

- **主路径**：DeepSpeed (ZeRO3)，作为项目默认训练后端。
- **备选路径**：FSDP，作为 DeepSpeed 在 PyTorch 原生派系下的等价替代。
- **接口预留**：accelerate 用于单卡调试与 CI 烟雾测试；Megatron 用于后续扩展到 7B+ 模型时的预研接口。

### 7.2 训练框架协作关系图

```
┌─────────────────────────────────────────────┐
│             LLaMA-Factory                    │
│  - 数据加载（SFT/DPO 模板）                  │
│  - 模型加载（HF transformers + LoRA 注入）   │
│  - 训练循环（Hugging Face Trainer）          │
│  - LoRA / DPO loss 计算                      │
│  - yaml 配置驱动                             │
└─────────────────────────────────────────────┘
                    ↑ 适配
┌─────────────────────────────────────────────┐
│             自研 Trainer（M04）              │
│  - 4 种分布式后端统一接口                    │
│  - 优化器/调度器工厂                         │
│  - ZeRO3 配置生成器                          │
│  - FSDP wrap policy                          │
│  - Megatron 张量并行配置                     │
└─────────────────────────────────────────────┘
```

### 7.3 后端切换 demo

```bash
# 训练（默认 DeepSpeed）
copaw-dpo train-dpo --config configs/train_dpo_qwen2_5_1_5b_insurance.yaml

# 切换到 FSDP（仅一行 yaml 改动）
sed -i 's/distributed_backend: deepspeed/distributed_backend: fsdp/' \
    configs/train_dpo_qwen2_5_1_5b_insurance.yaml
copaw-dpo train-dpo --config configs/train_dpo_qwen2_5_1_5b_insurance.yaml

# 切换到 accelerate（单卡调试）
sed -i 's/distributed_backend: deepspeed/distributed_backend: accelerate/' \
    configs/train_dpo_qwen2_5_1_5b_insurance.yaml
copaw-dpo train-dpo --config configs/train_dpo_qwen2_5_1_5b_insurance.yaml
```

业务代码无任何改动。

### 7.4 Merge CLI 示例

```bash
# 使用 LLaMA-Factory 自带的 merge 脚本
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --adapter_name_or_path saves/qwen2_5_1_5b/insurance_dpo_v1.2/lora \
    --template qwen \
    --finetuning_type lora \
    --export_dir merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \
    --export_size 5 \
    --export_device cpu \
    --export_legacy_format false

# 或使用本阶段自研 merge 模块
python -m m_merge.cli \
    --base Qwen/Qwen2.5-1.5B-Instruct \
    --adapter saves/qwen2_5_1_5b/insurance_dpo_v1.2/lora \
    --output merged_models/qwen2_5_1_5b_insurance_dpo_v1.2
```

### 7.5 Checkpoint 统一策略

| 字段 | DeepSpeed | FSDP | accelerate | Megatron |
|------|-----------|------|------------|----------|
| optimizer state | ZeRO3 分片 | FSDP 全分片 | 不分片 | TP/PP 分片 |
| model state | ZeRO3 分片 | FSDP 全分片 | 全复制 | TP/PP 分片 |
| 其他状态 | engine.state_dict() | fsdp.state_dict() | accelerator.state | 各自接口 |

自研 backend 通过统一 `state_dict()` / `load_state_dict()` 接口屏蔽差异，checkpoint 在 4 种后端间不可直接互换（需 save_pretrained 重新分片），但 load 逻辑统一。

---

## 8. 风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| FSDP v2 API 在 PyTorch < 2.4 不可用 | 备选后端无法启动 | 锁定 PyTorch ≥ 2.4.0 |
| DeepSpeed 与 accelerate 单测冲突 | CI 不稳 | 测试时使用不同 worktree |
| Megatron 接口预留引发误用 | 用户在 1.5B 启用 TP | `NotImplementedError` 显式拒绝 + 文档说明 |
| Merge 后模型在 vLLM 加载失败 | 推理阻塞 | M04 与 M05 联合冒烟测试 |
| Checkpoint 跨后端无法加载 | 切换成本高 | 限制场景：仅同后端内续训，跨后端需 `merge_and_unload` 后重新加载 |
| FSDP 通信开销大 | 训练慢 | auto_wrap_policy 调优（按 layer size 分片） |

---

## 9. 阶段完成 checklist

- [ ] `m_trainer/` 模块完整代码（base / 4 adapters / factory / registry）
- [ ] `m_merge/` 模块完整代码（exporter / cli）
- [ ] 4 种后端冒烟测试日志全部通过
- [ ] 同一份 yaml 在 DeepSpeed 与 FSDP 上训练通过
- [ ] merge 后模型可被 vLLM 加载
- [ ] 单测覆盖率 ≥ 60%
- [ ] `m_trainer/README.md` 撰写完成
- [ ] `m_merge/README.md` 撰写完成
- [ ] 后端切换 demo 录制
