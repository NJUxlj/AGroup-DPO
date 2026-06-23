# 如何运行 DPO 对齐训练

> 面向开源社区的 Direct Preference Optimization 训练教程  
> 对应模块：`src/m_trainer/` · 阶段 M03（FR-02）  
> 理论背景：[DPO-Tutorial.md](./DPO-Tutorial.md)

---

## 目录

1. [DPO 在流水线中的位置](#1-dpo-在流水线中的位置)
2. [前置条件](#2-前置条件)
3. [LoRA + DPO 组合策略](#3-lora--dpo-组合策略)
4. [方式一：LLaMA-Factory DPO 训练](#4-方式一llama-factory-dpo-训练)
5. [方式二：CustomTrainer DPO 训练](#5-方式二customtrainer-dpo-训练)
6. [配置文件详解](#6-配置文件详解)
7. [DPO 超参数调优](#7-dpo-超参数调优)
8. [训练监控](#8-训练监控)
9. [训练完成后：合并与部署](#9-训练完成后合并与部署)
10. [烟雾测试 vs 完整训练](#10-烟雾测试-vs-完整训练)
11. [常见问题](#11-常见问题)

---

## 1. DPO 在流水线中的位置

DPO（Direct Preference Optimization）跳过传统 RLHF 的奖励模型 + PPO 阶段，**直接从偏好数据优化语言模型**：

```
SFT merged 模型（参考模型 π_ref）
    ↓  DPO 训练（本文）
DPO LoRA adapter
    ↓  merge
最终对齐模型 → 推理 / 评测
```

DPO 损失函数（Rafailov et al., 2023）：

\[
\mathcal{L}_{\mathrm{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right)\right]
\]

其中 \(y_w\) = chosen，\(y_l\) = rejected，\(\beta\) 控制偏离参考模型的强度。

---

## 2. 前置条件

### 2.1 环境

同 SFT，参考 [How-to-Deploy.md](./How-to-Deploy.md)。

### 2.2 SFT merged 模型

DPO **必须**以 SFT 后的模型作为参考模型起点。请先完成 SFT 并 merge：

```bash
# 1. SFT 训练（若尚未完成）
copaw-dpo train --config configs/train_lora_qwen2_5_1_5b_insurance.yaml

# 2. 合并 SFT LoRA
copaw-dpo merge \
    --base /path/to/models/Qwen2.5-1.5B-Instruct \
    --adapter saves/qwen2_5_1_5b/insurance_sft_v1/lora \
    --output saves/qwen2_5_1_5b/insurance_sft_v1/lora_merged
```

确认 merged 模型可加载：

```bash
ls saves/qwen2_5_1_5b/insurance_sft_v1/lora_merged/model*.safetensors
```

### 2.3 DPO 偏好数据

```
data/insurance/dpo_train_v1.2.jsonl    # (prompt, chosen, rejected) 三元组
data/insurance/dataset_info.json       # insurance_dpo_v1.2 已注册
```

数据生成见 [How-to-synthesize-training-data.md](./How-to-synthesize-training-data.md)。

---

## 3. LoRA + DPO 组合策略

| 策略 | 做法 | 项目采用 |
|------|------|----------|
| **策略 1（默认）** | SFT LoRA merge → 在 merged 模型上再注入新 DPO LoRA | ✅ |
| 策略 2 | SFT merge → 全参数 DPO | 显存压力大 |
| 策略 3 | 直接在 base 上做 DPO | 跳过 SFT，效果差 |

**策略 1 的流程：**

```
Base (Qwen2.5-1.5B)
  → SFT LoRA 训练
  → merge → SFT merged（冻结为参考模型）
  → 注入新 DPO LoRA（仅训练 DPO LoRA）
  → merge → 最终部署模型
```

优势：SFT 与 DPO 权重解耦，可单独回滚；显存友好。

---

## 4. 方式一：LLaMA-Factory DPO 训练

### 4.1 完整训练

配置文件：[`configs/train_dpo_qwen2_5_1_5b_insurance.yaml`](../../configs/train_dpo_qwen2_5_1_5b_insurance.yaml)

```bash
# 双卡训练
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1 \
    copaw-dpo train --config configs/train_dpo_qwen2_5_1_5b_insurance.yaml

# 等价于
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1 \
    llamafactory-cli train configs/train_dpo_qwen2_5_1_5b_insurance.yaml
```

**关键配置项**（务必确认）：

```yaml
model_name_or_path: saves/qwen2_5_1_5b/insurance_sft_v1/lora_merged   # SFT merged 模型

stage: dpo
finetuning_type: lora

dataset: insurance_dpo_v1.2

pref_beta: 0.1          # DPO β 参数（KL 强度）
pref_loss: sigmoid      # 标准 DPO 损失
```

### 4.2 训练产物

```
saves/qwen2_5_1_5b/insurance_dpo_v1.2/lora/
├── adapter_config.json
├── adapter_model.safetensors
├── checkpoint-200/
└── trainer_state.json
```

---

## 5. 方式二：CustomTrainer DPO 训练

### 5.1 单卡 smoke 测试

```bash
copaw-dpo train \
    --config configs/smoke_custom_dpo_insurance.yaml \
    --backend accelerate
```

Smoke 配置要点：

```yaml
model_name_or_path: /path/to/models/Qwen2.5-1.5B-Instruct   # smoke 直接用 base
stage: dpo
dataset: insurance_dpo_v1.2
dpo_max_prompt_length: 512
pref_beta: 0.1
pref_loss: sigmoid
max_steps: 20
```

> Smoke 测试为简化流程直接使用 base 模型；**生产 DPO 必须使用 SFT merged 模型**。

### 5.2 双卡 DeepSpeed

```bash
deepspeed --num_gpus=2 --module m_trainer.cli -- \
    --config configs/smoke_custom_dpo_insurance_deepspeed_2gpu.yaml \
    --backend deepspeed
```

### 5.3 双卡 Megatron

```bash
copaw-dpo train \
    --config configs/smoke_custom_dpo_insurance_megatron_2gpu.yaml \
    --backend megatron
```

### 5.4 一键测试脚本

```bash
# 单卡：SFT smoke → DPO smoke
bash deploy/run_m_trainer_test.sh

# 双卡：DeepSpeed + Megatron
bash deploy/run_m_trainer_dual_gpu_test.sh
```

---

## 6. 配置文件详解

完整生产配置 [`configs/train_dpo_qwen2_5_1_5b_insurance.yaml`](../../configs/train_dpo_qwen2_5_1_5b_insurance.yaml)：

```yaml
model_name_or_path: saves/qwen2_5_1_5b/insurance_sft_v1/lora_merged

stage: dpo
do_train: true
finetuning_type: lora
lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05

dataset_dir: data/insurance
dataset: insurance_dpo_v1.2
template: qwen
cutoff_len: 2048

output_dir: saves/qwen2_5_1_5b/insurance_dpo_v1.2/lora

per_device_train_batch_size: 2
gradient_accumulation_steps: 16    # 有效 batch = 2×16×2 = 64
learning_rate: 5.0e-6
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true

pref_beta: 0.1
pref_loss: sigmoid

flash_attn: fa2
report_to: tensorboard
seed: 42
```

---

## 7. DPO 超参数调优

| 超参 | 推荐值 | 说明 | 调优方向 |
|------|--------|------|----------|
| `learning_rate` | 5.0e-6 | DPO 远低于 SFT（5e-7 ~ 5e-6） | 过大 → 偏离参考模型太快，不稳定 |
| `pref_beta` (β) | 0.1 | KL 约束强度 | 过大 → 保守，接近参考模型；过小 → 激进 |
| `pref_loss` | sigmoid | 标准 DPO | 也可尝试 ipo / kto |
| `num_train_epochs` | 2.0 | DPO 不易过拟合 | 数据量大可减至 1 |
| `warmup_ratio` | 0.05 | 比 SFT 略高 | 避免初期梯度震荡 |
| effective batch | 64 | per_device × ga × GPU | 影响梯度估计稳定性 |

**经验法则：**

- DPO loss 应稳定下降，若震荡剧烈 → 降低 lr 或增大 β
- chosen-reward − rejected-reward margin ≥ 0.5 为验收目标
- 若模型输出变得过于保守（复读参考模型）→ 降低 β 或略增 lr
- 若模型偏离参考模型太远（胡言乱语）→ 增大 β 或降低 lr

---

## 8. 训练监控

### 8.1 TensorBoard

```bash
tensorboard --logdir saves/qwen2_5_1_5b/insurance_dpo_v1.2/lora
```

关注指标：

| 指标 | 期望趋势 |
|------|----------|
| `train/loss` | 稳定下降 |
| `train/rewards/chosen` | 上升 |
| `train/rewards/rejected` | 下降或持平 |
| `train/rewards/margins` | 上升（chosen − rejected） |
| `train/logps/chosen` vs `rejected` | chosen 对数概率应高于 rejected |

### 8.2 训练中人工抽检

每 200 step 保存 checkpoint 后，可快速 merge 并推理：

```bash
copaw-dpo merge --base saves/.../lora_merged \
    --adapter saves/.../insurance_dpo_v1.2/lora/checkpoint-200 \
    --output /tmp/dpo_ckpt200

copaw-dpo infer --backend vllm --model /tmp/dpo_ckpt200 \
    --prompts "重疾险等待期内确诊是否赔付？"
```

---

## 9. 训练完成后：合并与部署

### 9.1 合并 DPO LoRA

DPO LoRA 是在 SFT merged 模型之上训练的，合并时需要**以 SFT merged 为 base**：

```bash
copaw-dpo merge \
    --base saves/qwen2_5_1_5b/insurance_sft_v1/lora_merged \
    --adapter saves/qwen2_5_1_5b/insurance_dpo_v1.2/lora \
    --output merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \
    --device cuda \
    --dtype bfloat16
```

### 9.2 推理验证

```bash
copaw-dpo infer --backend vllm \
    --model merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \
    --prompts "保险等待期是什么？" "百万医疗险免赔额怎么算？"
```

### 9.3 启动 HTTP 推理服务

```bash
# 修改 configs/infer.yaml 中的 model_path
copaw-dpo infer --config configs/infer.yaml --host 0.0.0.0 --port 8080
```

### 9.4 一键 merge + 评测

```bash
bash deploy/run_merge_and_eval.sh
```

该脚本自动完成：DPO LoRA merge → 全量 1700 条评测集评估。

---

## 10. 烟雾测试 vs 完整训练

| 维度 | Smoke Test | 完整训练 |
|------|------------|----------|
| 配置 | `configs/smoke_custom_dpo_insurance.yaml` | `configs/train_dpo_qwen2_5_1_5b_insurance.yaml` |
| 步数 | `max_steps: 20` | `num_train_epochs: 2.0` |
| 参考模型 | base 模型（简化） | SFT merged 模型 |
| LoRA rank | 8 | 16 |
| 有效 batch | 2 | 64 |
| 耗时 | ~3–5 min | ~1–3 h |
| 目的 | 验证 DPO 链路 | 产出对齐模型 |

---

## 11. 常见问题

| 问题 | 解决方案 |
|------|----------|
| DPO loss 为 NaN | 降低 lr；检查数据中 chosen/rejected 是否完全相同 |
| 模型输出与 SFT 无差异 | β 过大或 lr 过小；检查 DPO 数据质量 |
| 模型质量退化 | β 过小或训练过久；减少 epoch 或增大 β |
| `model_name_or_path` 报错 | 确认指向 SFT **merged** 目录，非 LoRA adapter |
| merge 后推理乱码 | 确认 merge 的 base 是 SFT merged，adapter 是 DPO LoRA |
| OOM | DPO 需同时加载 policy + reference；减小 batch 或使用 DeepSpeed ZeRO-3 |
| LLaMA-Factory DPO 参数名 | 0.9.1 版本使用 `pref_beta` + `pref_loss`，非 `dpo_beta` |

---

**下一步**：[how-to-evaluate-model-performance.md](./how-to-evaluate-model-performance.md) — 对合并后的模型进行系统化评测。
