# 如何运行 SFT 微调

> 面向开源社区的 LoRA 监督微调教程  
> 对应模块：`src/m_trainer/` · 阶段 M03（FR-01）

---

## 目录

1. [SFT 在流水线中的位置](#1-sft-在流水线中的位置)
2. [前置条件](#2-前置条件)
3. [两种训练后端](#3-两种训练后端)
4. [方式一：LLaMA-Factory 训练（推荐生产）](#4-方式一llama-factory-训练推荐生产)
5. [方式二：CustomTrainer 训练（自研后端）](#5-方式二customtrainer-训练自研后端)
6. [配置文件说明](#6-配置文件说明)
7. [单卡 vs 多卡](#7-单卡-vs-多卡)
8. [监控与 checkpoint](#8-监控与-checkpoint)
9. [SFT 完成后：合并 LoRA](#9-sft-完成后合并-lora)
10. [烟雾测试 vs 完整训练](#10-烟雾测试-vs-完整训练)
11. [常见问题](#11-常见问题)

---

## 1. SFT 在流水线中的位置

SFT（Supervised Fine-Tuning，监督微调）是 DPO 对齐的**第一阶段**：

```
基座模型 (Qwen2.5-1.5B-Instruct)
    ↓  SFT（本文）
SFT 模型（具备保险领域指令遵循能力）
    ↓  merge LoRA
SFT merged 模型（作为 DPO 参考模型起点）
    ↓  DPO（见 How-to-run-DPO.md）
对齐后的 DPO 模型
```

SFT 的目标是让模型学会**按保险条款规范回答问题**，为后续 DPO 偏好优化提供良好的初始化。

---

## 2. 前置条件

### 2.1 环境

参考 [How-to-Deploy.md](./How-to-Deploy.md) 完成环境部署。

### 2.2 训练数据

SFT 数据由 M-DATA 流水线产出，或直接使用仓库内置文件：

```
data/insurance/
├── insurance_sft_v1.jsonl      # SFT 训练集
└── dataset_info.json           # LLaMA-Factory 数据集注册
```

若需重新生成：

```bash
copaw-dpo data --config configs/data/insurance_dpo_gen.yaml
```

### 2.3 基座模型

```bash
# 确认模型路径存在
ls /path/to/models/Qwen2.5-1.5B-Instruct/config.json
```

---

## 3. 两种训练后端

| 后端 | 入口 | 适用场景 |
|------|------|----------|
| **LLaMA-Factory** | `--backend llamafactory`（默认） | 生产训练，yaml 配置成熟 |
| **CustomTrainer** | `--backend deepspeed/fsdp/accelerate/megatron` | 自研分布式后端，多卡扩展 |

两者共用同一份 yaml 配置格式（兼容 LLaMA-Factory），通过 `--backend` 切换。

---

## 4. 方式一：LLaMA-Factory 训练（推荐生产）

### 4.1 完整训练

配置文件：[`configs/train_lora_qwen2_5_1_5b_insurance.yaml`](../../configs/train_lora_qwen2_5_1_5b_insurance.yaml)

```bash
# 双卡训练
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1 \
    copaw-dpo train --config configs/train_lora_qwen2_5_1_5b_insurance.yaml

# 等价于
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1 \
    llamafactory-cli train configs/train_lora_qwen2_5_1_5b_insurance.yaml
```

**修改模型路径**：编辑 yaml 中的 `model_name_or_path`：

```yaml
model_name_or_path: /path/to/models/Qwen2.5-1.5B-Instruct
```

### 4.2 关键超参数

| 超参 | 取值 | 说明 |
|------|------|------|
| `lora_rank` | 16 | 表达力与显存平衡 |
| `lora_alpha` | 32 | alpha/rank = 2 |
| `learning_rate` | 1.0e-4 | LoRA 常用区间 |
| `cutoff_len` | 2048 | 序列最大长度 |
| `per_device_train_batch_size` | 4 | 单卡 batch |
| `gradient_accumulation_steps` | 8 | 有效 batch = 4×8×GPU数 |
| `num_train_epochs` | 3.0 | 业务数据 3 epoch 通常足够 |

### 4.3 训练产物

```
saves/qwen2_5_1_5b/insurance_sft_v1/lora/
├── adapter_config.json
├── adapter_model.safetensors
├── checkpoint-500/
├── checkpoint-1000/
└── trainer_state.json
```

---

## 5. 方式二：CustomTrainer 训练（自研后端）

CustomTrainer 支持 4 种分布式后端，适合需要 DeepSpeed ZeRO / Megatron 等高级并行的场景。

### 5.1 单卡 accelerate（推荐调试）

```bash
copaw-dpo train \
    --config configs/smoke_custom_sft_insurance.yaml \
    --backend accelerate
```

### 5.2 双卡 DeepSpeed ZeRO-2

```bash
deepspeed --num_gpus=2 --module m_trainer.cli -- \
    --config configs/smoke_custom_sft_insurance_deepspeed_2gpu.yaml \
    --backend deepspeed
```

### 5.3 双卡 Megatron 张量并行

```bash
copaw-dpo train \
    --config configs/smoke_custom_sft_insurance_megatron_2gpu.yaml \
    --backend megatron
```

### 5.4 后端对比

| 后端 | 显存效率 | 适用模型 | 备注 |
|------|----------|----------|------|
| `accelerate` | 中 | 全部 | 单机调试首选 |
| `deepspeed` | 高（ZeRO-2/3） | 全部 | 生产默认 |
| `fsdp` | 高 | 全部 | PyTorch 原生 |
| `megatron` | 高（TP） | GPT-2, Qwen2.5, Qwen3 | 大模型张量并行 |

---

## 6. 配置文件说明

以 smoke 配置为例（[`configs/smoke_custom_sft_insurance.yaml`](../../configs/smoke_custom_sft_insurance.yaml)）：

```yaml
model_name_or_path: /path/to/models/Qwen2.5-1.5B-Instruct

stage: sft                    # 训练阶段：sft
finetuning_type: lora         # 微调方式：LoRA
lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05

dataset_dir: data/insurance   # 数据目录
dataset: insurance_sft_v1     # 对应 dataset_info.json 中的 key
template: qwen                # Qwen 对话模板

cutoff_len: 1024
output_dir: saves/smoke/custom_sft_insurance

per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 5.0e-5
max_steps: 20                 # smoke: 20 步；生产用 num_train_epochs
bf16: true
seed: 42

trainer:
  distributed_backend: accelerate   # CustomTrainer 专用
```

**LLaMA-Factory 与 CustomTrainer 的差异：**

- LLaMA-Factory 配置需加 `do_train: true`
- CustomTrainer 通过 `trainer.distributed_backend` 或 CLI `--backend` 指定后端
- CustomTrainer smoke 配置使用 `max_steps` 代替 `num_train_epochs`

---

## 7. 单卡 vs 多卡

### 7.1 单卡（smoke / 调试）

```bash
CUDA_VISIBLE_DEVICES=0 \
    copaw-dpo train --config configs/smoke_custom_sft_insurance.yaml --backend accelerate
```

### 7.2 双卡 LLaMA-Factory DDP

```bash
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1 \
    llamafactory-cli train configs/train_lora_qwen2_5_1_5b_insurance.yaml
```

### 7.3 多卡 NCCL 环境变量（如遇通信问题）

```bash
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=120
export PYTORCH_NCCL_BLOCKING_WAIT=1
```

---

## 8. 监控与 checkpoint

### 8.1 TensorBoard

生产配置中 `report_to: tensorboard`：

```bash
tensorboard --logdir saves/qwen2_5_1_5b/insurance_sft_v1/lora
```

关注指标：

- `train/loss` — 应持续下降
- `train/learning_rate` — cosine 调度曲线
- `train/grad_norm` — 不应出现 NaN 或爆炸

### 8.2 checkpoint 管理

```yaml
save_steps: 500
save_total_limit: 3    # 最多保留 3 个 checkpoint
```

CustomTrainer smoke 训练完成后 checkpoint 位于：

```
saves/smoke/custom_sft_insurance/checkpoint-final/
├── adapter_config.json
└── adapter_model.safetensors
```

---

## 9. SFT 完成后：合并 LoRA

DPO 训练需要将 SFT LoRA **合并进基座**，得到完整的 HF 模型作为参考模型起点（策略 1，项目默认）。

### 9.1 使用 copaw-dpo merge

```bash
copaw-dpo merge \
    --base /path/to/models/Qwen2.5-1.5B-Instruct \
    --adapter saves/qwen2_5_1_5b/insurance_sft_v1/lora \
    --output saves/qwen2_5_1_5b/insurance_sft_v1/lora_merged \
    --device cuda \
    --dtype bfloat16
```

### 9.2 使用 LLaMA-Factory export

```bash
llamafactory-cli export \
    --model_name_or_path /path/to/models/Qwen2.5-1.5B-Instruct \
    --adapter_name_or_path saves/qwen2_5_1_5b/insurance_sft_v1/lora \
    --export_dir saves/qwen2_5_1_5b/insurance_sft_v1/lora_merged \
    --export_size 5 \
    --export_device cpu
```

合并后的模型目录应包含 `model*.safetensors`、`config.json`、`tokenizer.json` 等。

> **重要**：DPO 配置的 `model_name_or_path` 必须指向此 merged 模型，而非原始基座或 LoRA adapter 目录。

---

## 10. 烟雾测试 vs 完整训练

| 维度 | Smoke Test | 完整训练 |
|------|------------|----------|
| 配置文件 | `configs/smoke_custom_sft_insurance.yaml` | `configs/train_lora_qwen2_5_1_5b_insurance.yaml` |
| 步数 | `max_steps: 20` | `num_train_epochs: 3.0` |
| LoRA rank | 8 | 16 |
| 数据量 | 全量（但只跑 20 步） | 全量 × 3 epoch |
| 耗时 | ~2–5 min（单卡） | ~2–4 h（双卡 A100） |
| 目的 | 验证训练链路 | 产出可用 SFT 模型 |
| 一键脚本 | `bash deploy/run_m_trainer_test.sh` | 手动启动 |

M01 阶段 SFT 烟雾（LLaMA-Factory，5 步）：

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/smoke_lora_qwen2_5_1_5b.yaml
```

---

## 11. 常见问题

| 问题 | 解决方案 |
|------|----------|
| `dataset not found` | 确认 `data/insurance/dataset_info.json` 存在且 `dataset` 名称匹配 |
| OOM（显存不足） | 减小 `per_device_train_batch_size` 或 `cutoff_len`；使用 DeepSpeed ZeRO-3 |
| loss 不下降 | 检查学习率（SFT 常用 1e-4 ~ 5e-5）；确认数据格式正确 |
| 多卡 DDP 崩溃 | smoke 阶段先用单卡；检查 NCCL 环境变量 |
| merge 后模型无法加载 | 确认 adapter 路径含 `adapter_model.safetensors` |
| CustomTrainer 找不到模块 | 设置 `export PYTHONPATH=src` |

---

**下一步**：[How-to-run-DPO.md](./How-to-run-DPO.md) — 在 SFT merged 模型上进行 DPO 偏好对齐训练。
