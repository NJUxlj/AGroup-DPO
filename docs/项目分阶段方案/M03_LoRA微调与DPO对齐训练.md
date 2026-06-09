# M03 LoRA 微调与 DPO 对齐训练

> 阶段编号：M03
> 阶段名称：LoRA 微调与 DPO 对齐训练
> 预估工期：6 人天
> 关联文档：[项目方案.md § 5.2 M-LORA](../项目方案.md) | [项目方案.md § 5.3 M-DPO](../项目方案.md) | [项目方案.md § 3.2 训练框架选型](../项目方案.md)
> 上游阶段：M01、M02
> 下游阶段：M05
> 对应功能需求：FR-01、FR-02、FR-05、FR-06

---

## 1. 阶段定位

M03 是**核心训练阶段**，是整个对齐模型的实际产出方。本阶段交付最终可用于部署的 LoRA 权重（后续 M05 经 merge 后得到 HF 模型）。

与 M04 的关系：

- M03 侧重 **LLaMA-Factory 的 yaml 配置与训练流程**（"训练什么"）。
- M04 侧重 **自研 Trainer 的分布式后端抽象**（"用什么后端跑"）。
- 二者并行推进，但 M04 的 `DistributedBackend` 接口需先稳定，M03 才能接入备选后端（FSDP）。

---

## 2. 阶段目标

### 2.1 业务目标

| 维度 | 目标 | 衡量方式 |
|------|------|----------|
| 对齐效果 | DPO chosen-reward - rejected-reward margin | ≥ 0.5（验收） |
| LoRA 收敛 | SFT loss 在最终 100 step 窗口内下降 | 相对初值 ≥ 50% |
| 模型可 merge | merge 后模型可正常推理 | merge 后 perplexity ≤ 原模型 1.2× |

### 2.2 技术目标

| 维度 | 目标 | 衡量方式 |
|------|------|----------|
| 主后端跑通 | DeepSpeed ZeRO3 在 2×A100 上训练成功 | 训练日志 |
| 备后端跑通 | FSDP 至少跑一次冒烟 | 训练日志 |
| 性能吞吐 | 训练 throughput | ≥ 1.5 samples/s/GPU |
| 可观测 | loss / grad_norm / gpu_mem | TensorBoard / W&B |
| 复现性 | 同 seed 下 loss 曲线误差 | ≤ 2% |

---

## 3. 核心任务

### 3.1 LLaMA-Factory 集成方式

LLaMA-Factory 提供 `LlamaFactory` 库 + `llamafactory-cli` 命令行工具。本模块以 `llamafactory-cli` 作为调用入口，yaml 作为配置驱动。

**职责划分**：

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
│             自研 Trainer（FR-07 / M04）      │
│  - 4 种分布式后端统一接口                    │
│  - 优化器/调度器工厂                         │
│  - ZeRO3 配置生成器                          │
│  - FSDP wrap policy                          │
│  - Megatron 张量并行配置                     │
└─────────────────────────────────────────────┘
```

LLaMA-Factory 负责"训练什么"，自研 Trainer 负责"用什么后端跑"。

### 3.2 LoRA SFT 微调配置（FR-01）

```yaml
# configs/train_lora_qwen2_5_1_5b_insurance.yaml
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
trust_remote_code: true

stage: sft
do_train: true
finetuning_type: lora
lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05

dataset_dir: data/insurance
dataset: insurance_sft_v1
# 数据来源：M02 产出的 insurance_sft_v1.jsonl（M02 6.3 接口契约）
# 文件位置：data/insurance/insurance_sft_v1.jsonl
template: qwen
cutoff_len: 2048
overwrite_cache: false
preprocessing_num_workers: 8

output_dir: saves/qwen2_5_1_5b/insurance_sft_v1/lora
logging_steps: 20
save_steps: 500
save_total_limit: 3
plot_loss: true
overwrite_output_dir: true

per_device_train_batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
tf32: true

flash_attn: fa2
report_to: tensorboard
seed: 42
```

**SFT 超参说明**：

| 超参 | 取值 | 说明 |
|------|------|------|
| lora_rank | 16 | 在 1.5B 模型上兼顾表达力与显存 |
| lora_alpha | 32 | alpha/rank = 2，标准配比 |
| learning_rate | 1.0e-4 | LoRA 常用区间 1e-4 ~ 5e-4 |
| cutoff_len | 2048 | 保险条款问答通常 < 1k，留 buffer |
| per_device_batch_size | 4 × 8 = 32 effective | 2×A100-80G 适配 |
| num_train_epochs | 3.0 | 业务数据规模较小，3 epoch 足够收敛 |

### 3.3 DPO 对齐训练配置（FR-02）

DPO 损失（Rafailov et al., 2023）：

\[
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim D}\left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right)\right]
\]

其中 \(y_w\) 为 chosen，\(y_l\) 为 rejected，\(\pi_{\mathrm{ref}}\) 为冻结的参考模型（一般为 SFT 后的模型），\(\beta\) 控制偏离参考模型的强度。

```yaml
# configs/train_dpo_qwen2_5_1_5b_insurance.yaml
# 注：DPO 起点 = M03 SFT 阶段 merge 后的 HF 模型（见 3.4 策略 1）。
#      SFT LoRA merge 命令参考 M04 3.5 节 llamafactory-cli export；也可用 M04 自研 m_merge 模块。
#      若 SFT LoRA 尚未 merge，可临时改为 base + adapter_name_or_path 双路径（LLaMA-Factory 支持）。
model_name_or_path: saves/qwen2_5_1_5b/insurance_sft_v1/lora_merged
trust_remote_code: true

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
logging_steps: 10
save_steps: 200
plot_loss: true

per_device_train_batch_size: 2
gradient_accumulation_steps: 16   # effective batch = 2 * 16 * 2 = 64
learning_rate: 5.0e-6              # DPO 学习率通常显著低于 SFT
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
tf32: true

# DPO 专属超参
loss_type: dpo                    # 可选: dpo / ipo / kto / simpo
dpo_beta: 0.1                     # KL 强度
dpo_loss_type: sigmoid            # 标准 DPO
dpo_max_prompt_length: 1024
dpo_max_length: 2048

flash_attn: fa2
report_to: tensorboard
seed: 42
```

**DPO 超参说明**：

| 超参 | 取值 | 说明 |
|------|------|------|
| learning_rate | 5.0e-6 | DPO 学习率通常显著低于 SFT，常用区间 5e-7 ~ 5e-6；取值越大越快偏离参考模型 |
| effective batch | 64 | 2×A100 + per_device=2 + ga=16 = 64 |
| dpo_beta | 0.1 | KL 强度，控制偏离参考模型的强度 |
| num_train_epochs | 2.0 | DPO 不易过拟合，2 epoch 通常足够 |
| warmup_ratio | 0.05 | 比 SFT 略高，避免初期不稳定 |

### 3.4 LoRA + DPO 的组合策略

| 策略 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| **策略 1（项目默认）** | 先将 SFT LoRA **merge** 进基座得到 SFT 模型 → 在 SFT 模型之上**再注入一层新的 DPO LoRA**，只训练 DPO LoRA | 显存友好；DPO LoRA 与 SFT 权重解耦，可单独 merge 与回滚 | 需要先做一次 merge（SFT 阶段输出即可用 `merge_and_unload`） |
| 策略 2 | 先 merge SFT LoRA → 全参数 DPO | 表达力强 | 显存压力大，与"LoRA+DPO"语义不符 |
| 策略 3 | 直接在 base + 单 LoRA 上做 DPO | 最简单 | 跳过 SFT，模型指令遵循能力弱 |

**项目默认采用策略 1**，对应 3.3 节 DPO yaml 中 `model_name_or_path: saves/qwen2_5_1_5b/insurance_sft_v1/lora_merged`（SFT LoRA merge 后的 HF 模型作为 DPO 起点），由 LLaMA-Factory 在 DPO 阶段重新注入一组新的 LoRA（`insurance_dpo_v1.2/lora`）并只训练这一组。

### 3.5 分布式训练启动命令（FR-05）

**主路径：DeepSpeed ZeRO3**

```bash
deepspeed --num_gpus=2 src/train.py \
    --config configs/train_dpo_qwen2_5_1_5b_insurance.yaml \
    --deepspeed configs/deepspeed/zero3.json
```

或通过 LLaMA-Factory CLI：

```bash
FORCE_TORCHRUN=1 llamafactory-cli train \
    configs/train_dpo_qwen2_5_1_5b_insurance.yaml
```

**备选路径：FSDP**

通过 M04 自研 Trainer 切换 `trainer.distributed_backend: fsdp`，具体接口见 M04 文档。

### 3.6 监控与可观测（NFR-08）

| 指标 | 采集方式 | 告警阈值 |
|------|----------|----------|
| 训练 loss | TensorBoard + 自研 callback 推送 W&B | 连续 200 step 不下降 |
| GPU 利用率 | `nvidia-smi dmon` + DCGM Exporter | < 30% 持续 10 min |
| GPU 显存 | DCGM Exporter | > 90% 持续 5 min |
| 训练 throughput | 自研 callback | < 1.0 samples/s/GPU 持续 20 min |
| grad norm | LLaMA-Factory 内置 | > 10 持续 50 step（异常） |

---

## 4. 交付物清单

| 编号 | 交付物 | 路径 | 说明 |
|------|--------|------|------|
| D-M03-01 | LoRA SFT yaml | `configs/train_lora_qwen2_5_1_5b_insurance.yaml` | SFT 配置 |
| D-M03-02 | DPO yaml | `configs/train_dpo_qwen2_5_1_5b_insurance.yaml` | DPO 配置 |
| D-M03-03 | DeepSpeed ZeRO3 配置 | `configs/deepspeed/zero3.json` | 2 卡 ZeRO3 模板 |
| D-M03-04 | 训练入口脚本 | `src/train.py` | 适配 LLaMA-Factory |
| D-M03-05 | SFT LoRA 权重 | `saves/qwen2_5_1_5b/insurance_sft_v1/lora/` | 中间产物 |
| D-M03-06 | DPO LoRA 权重 | `saves/qwen2_5_1_5b/insurance_dpo_v1.2/lora/` | 最终 LoRA 权重 |
| D-M03-07 | 训练日志 | `logs/m03_train_*.log` | 含 loss / grad_norm / gpu_mem |
| D-M03-08 | TensorBoard 链接 | `runs/qwen2_5_dpo_v1.2/` | loss 曲线 |
| D-M03-09 | W&B 实验链接 | （在线） | 同 seed 多实验对比 |
| D-M03-10 | 训练报告 | `reports/m03_training_report.md` | 收敛曲线 / 超参敏感性 |
| D-M03-11 | 单测 | `tests/m03_train_*.py` | yaml 解析 + 模型加载冒烟 |

---

## 5. 验收标准

| 类别 | 验收项 | 量化阈值 | 检查方式 |
|------|--------|----------|----------|
| SFT 收敛 | 最终 100 step 窗口内 loss 平均值 | 相对初值下降 ≥ 50% | TensorBoard |
| DPO 收敛 | chosen-reward - rejected-reward margin | ≥ 0.5 | 自研 callback |
| 模型可 merge | merge 后 perplexity | ≤ 原模型 1.2× | 评测脚本 |
| 主后端 | DeepSpeed ZeRO3 训练成功 | 训练日志 PASS | 脚本 |
| 备后端 | FSDP 至少跑一次冒烟 | 训练日志 PASS | 脚本 |
| 性能 | 训练 throughput | ≥ 1.5 samples/s/GPU | 自研 callback |
| 显存 | GPU 显存占用 | ≤ 75 GB / 卡 | nvidia-smi |
| 可复现 | 同 seed 下 loss 曲线误差 | ≤ 2% | 对比脚本 |
| 可观测 | TensorBoard / W&B 可访问 | URL 可打开 | 人工 |

---

## 6. 依赖关系

### 6.1 上游依赖

```
┌──────────┐  ┌──────────┐
│   M01    │  │   M02    │
│ 基础设施 │  │ DPO 数据 │
└────┬─────┘  └────┬─────┘
     │             │
     └──────┬──────┘
            ▼
       ┌──────────┐
       │   M03    │  ← 本阶段
       └──────────┘
```

- M01：训练镜像（`copaw-dpo-deepspeed:latest` / `copaw-dpo-fsdp:latest`）
- M02：`insurance_sft_v1.jsonl`（≥ 3000 条） + `dpo_train_v1.2.jsonl`（≥ 5000 条）

### 6.2 下游依赖

```
       ┌──────────┐
       │   M03    │  ← 本阶段
       └────┬─────┘
            ▼
       ┌──────────┐
       │   M05    │  推理 + 评测
       └──────────┘
```

- M05 消费 `lora_ckpt_dpo/` 作为 merge 输入。

### 6.3 与 M04 的耦合

```
       ┌──────────┐      ┌──────────┐
       │   M03    │◀────▶│   M04    │
       │ LLaMA-F. │      │ 自研 Trnr│
       └──────────┘      └──────────┘
```

- M03 默认走 DeepSpeed ZeRO3（LLaMA-Factory 原生支持）。
- M03 通过 M04 提供的统一后端抽象切换到 FSDP / accelerate。
- M04 的 `DistributedBackend` 接口需在 M03 启动 FSDP 备路径前稳定。

### 6.4 跨阶段接口契约

| 接口 | 契约 | 调用方 |
|------|------|--------|
| SFT 输入数据 | `insurance_sft_v1.jsonl` ≥ 3000 条（alpaca schema） | 本阶段 → M02 |
| DPO 输入数据 | `dpo_train_v1.2.jsonl` ≥ 5000 条 | 本阶段 → M02 |
| 输入基座 | `Qwen/Qwen2.5-1.5B-Instruct`（HF model id 或本地路径） | 本阶段 → M01 |
| 输出 SFT 模型 | `saves/qwen2_5_1_5b/insurance_sft_v1/lora_merged/`（merge 后 HF 模型） | 本阶段 DPO → 本阶段 SFT |
| 输出 DPO LoRA | `saves/qwen2_5_1_5b/insurance_dpo_v1.2/lora/`（PEFT 格式） | M05 → 本阶段 |
| 训练日志 | TensorBoard event file + JSON callback | 监控 |

---

## 7. 详细技术规范

### 7.1 训练流程

```
[阶段 A: SFT 训练]
   │
   ▼
[启动] deepspeed --num_gpus=2 src/train.py --config train_lora_qwen2_5_1_5b_insurance.yaml
   │
   ▼
[加载基座] Qwen/Qwen2.5-1.5B-Instruct
   │
   ▼
[注入 SFT LoRA]
   │
   ▼
[加载数据集] data/insurance/insurance_sft_v1.jsonl（M02 产出）
   │
   ▼
[DeepSpeed ZeRO3 init]
   │
   ▼
[SFT 训练循环] → saves/qwen2_5_1_5b/insurance_sft_v1/lora/

[阶段 B: SFT LoRA merge]（M03 必做；M04 3.5 提供 CLI 与 m_merge 模块）
   │
   ▼
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --adapter_name_or_path saves/qwen2_5_1_5b/insurance_sft_v1/lora \
    --template qwen --finetuning_type lora \
    --export_dir saves/qwen2_5_1_5b/insurance_sft_v1/lora_merged \
    --export_size 5 --export_device cpu
   │
   ▼
产出：saves/qwen2_5_1_5b/insurance_sft_v1/lora_merged/（HF 模型）

[阶段 C: DPO 训练]
   │
   ▼
[启动] deepspeed --num_gpus=2 src/train.py --config train_dpo_qwen2_5_1_5b_insurance.yaml
   │
   ▼
[加载基座] saves/qwen2_5_1_5b/insurance_sft_v1/lora_merged/（SFT 模型）
   │
   ▼
[注入 DPO LoRA]（与 SFT LoRA 不同名空间，互不干扰）
   │
   ▼
[加载数据集] data/insurance/dpo_train_v1.2.jsonl
   │
   ▼
[DeepSpeed ZeRO3 init]  ── 优化器/梯度/参数分片 + CPU offload
   │
   ▼
[DPO 训练循环]
   │  for step in 1..N:
   │      forward → DPO loss
   │      backward → DeepSpeed engine.backward
   │      optimizer step → ZeRO3 all-gather
   │      logging (loss / reward margin / throughput)
   │      save checkpoint (每 save_steps)
   ▼
[训练完成] saves/qwen2_5_1_5b/insurance_dpo_v1.2/lora/

[阶段 D: DPO LoRA merge]（M05 推理前最终合并，由 M04/M05 协作完成）
   │
   ▼
产出：merged_models/qwen2_5_1_5b_insurance_dpo_v1.2/（vLLM 可直接加载）
```

### 7.2 SFT 数据集 Schema（可选输入）

```json
{"instruction": "请回答用户的保险问题。", "input": "重疾险等待期内确诊是否赔付？", "output": "等待期内确诊一般不予赔付，合同另有约定的除外。具体参见条款 X 第 Y 条。", "system": "你是蚂蚁保险的智能客服，需严格依据条款作答。"}
```

### 7.3 LoRA 合并公式（合并导出在 M04，本阶段先记录）

合并公式（PEFT 库 `merge_and_unload` 实现）：

\[
W_{\mathrm{merged}} = W_{\mathrm{base}} + \frac{\alpha}{r} \cdot B \cdot A
\]

其中 \(W_{\mathrm{base}}\) 为基座 Qwen2.5-1.5B-Instruct 的原权重，\(A \in \mathbb{R}^{r \times d_{\mathrm{in}}}\)，\(B \in \mathbb{R}^{d_{\mathrm{out}} \times r}\) 为 LoRA 矩阵，\(\alpha\) 为缩放系数，\(r\) 为 rank。

### 7.4 训练入口代码骨架

```python
# src/train.py（M03 训练主入口；与 M04 § 3.4 `setup_distributed` 协作）
import os
import torch
from llamafactory import TrainerFactory           # LLaMA-Factory 提供
from m_trainer.factory import build_backend       # M04 自研（详见 M04 § 3.4）
from m_data.factory import build_dataloader       # 数据加载（M02 产出 → M03 输入）

def main(cfg):
    # 1. 加载训练配置（yaml → OmegaConf）
    trainer_cfg = cfg.trainer

    # 2. LLaMA-Factory 加载模型 + 注入 LoRA（与 3.2/3.3 yaml 一致）
    model, tokenizer = TrainerFactory.load_model(
        model_name_or_path=cfg.model_name_or_path,
        finetuning_type="lora",
        lora_target=cfg.lora_target,
    )

    # 3. M04 提供的分布式包装（封装 model/optimizer 创建与分片）
    backend = build_backend(trainer_cfg)
    model, optimizer = backend.init(model, optimizer=None, config=trainer_cfg)
    # 注：optimizer=None 是 M04 显式约定——由 backend 内部创建并包装，
    #     DeepSpeed 返回 DeepSpeedEngine（非 torch.optim.Optimizer 子类），
    #     FSDP/accelerate 由 backend 内部 new + 包装。详见 M04 § 3.4 约定。

    # 4. 数据加载（M02 产出的 insurance_sft_v1.jsonl 或 dpo_train_v1.2.jsonl）
    dataloader = build_dataloader(
        dataset_path=cfg.dataset_dir + "/" + cfg.dataset,
        template=cfg.template,
        batch_size=cfg.per_device_train_batch_size,
    )

    # 5. 训练循环（loss 计算在 m_train 下；此处仅示意）
    num_epochs = cfg.num_train_epochs
    for epoch in range(int(num_epochs)):
        for batch in dataloader:
            loss = compute_dpo_loss(model, batch)        # m_train.compute_dpo_loss
            backend.engine.backward(loss)                 # DeepSpeed engine
            backend.engine.step()
            log_metrics(loss, model)                      # TensorBoard / W&B

    # 6. 保存 LoRA（PEFT adapter 形式）
    save_lora(model, cfg.output_dir)

if __name__ == "__main__":
    cfg = load_config()  # yaml → OmegaConf
    main(cfg)
```

> **与 M04 的契约**：本节代码调用的 `build_backend(cfg.trainer)` / `backend.init(model, optimizer=None, ...)` 是 M04 § 3.4 暴露的接口。M04 § 3.4 已对 `setup_distributed` 的封装、各后端 optimizer 创建位置差异做完整说明，本节不重复。

### 7.5 检查点管理

- LLaMA-Factory 默认每 `save_steps` 保存一次 PEFT adapter（仅 LoRA 权重，非完整模型）
- `save_total_limit: 3`：仅保留最近 3 个 checkpoint
- 训练中断后可通过 `resume_from_checkpoint: saves/.../checkpoint-XXX` 续训

### 7.6 切换后端示例

修改 yaml 一行即可从 DeepSpeed 切到 FSDP：

```yaml
# configs/train_dpo_qwen2_5_1_5b_insurance.yaml
trainer:
  distributed_backend: fsdp   # 原为 deepspeed
```

业务代码无需任何改动。

---

## 8. 风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| DPO loss 收敛后回弹（过拟合） | 对齐效果差 | 降低 `num_train_epochs` / 增大 `dpo_beta` |
| 双 LoRA 显存爆炸 | OOM | 启用 `offload_param: cpu`；考虑策略 2（全参数 DPO） |
| DeepSpeed ZeRO3 与 FSDP 接口不一致 | 切换后报错 | M04 提供统一 `DistributedBackend` 抽象 |
| chosen-reward margin 长期 < 0.3 | 对齐未生效 | 检查数据集质量 / 调整 `dpo_beta` / 增大训练量 |
| 训练吞吐 < 1.0 samples/s/GPU | 性能不达标 | 启用 FlashAttention-2 / 增大 per_device_batch / 减少 dataloader worker |
| LLaMA-Factory 升级导致 yaml 字段变更 | 配置失效 | 锁版本 + CI 烟雾测试 |

---

## 9. 阶段完成 checklist

- [ ] LoRA SFT yaml 配置完成
- [ ] DPO yaml 配置完成
- [ ] SFT LoRA merge 完成（产出 `saves/.../insurance_sft_v1/lora_merged/`）
- [ ] DeepSpeed ZeRO3 在 2×A100 上跑通 SFT + DPO
- [ ] FSDP 至少跑一次冒烟
- [ ] chosen-reward - rejected-reward margin ≥ 0.5
- [ ] 训练 throughput ≥ 1.5 samples/s/GPU
- [ ] TensorBoard / W&B 可访问
- [ ] `saves/.../insurance_dpo_v1.2/lora/` 产出
- [ ] 训练报告 `reports/m03_training_report.md` 撰写完成
- [ ] `tests/m03_train_*.py` 单测通过
