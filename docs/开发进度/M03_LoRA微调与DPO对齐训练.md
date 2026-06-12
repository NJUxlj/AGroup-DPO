# M03 LoRA 微调与 DPO 对齐训练 — 进度报告

> 完成日期：2026-06-11
> 最后更新：2026-06-11 10:40
> 完成状态：**✅ 全部完成（全量 SFT + DPO 训练通过，模型已 Merge）**

---

## 1. 交付物完成情况

| 编号 | 交付物 | 状态 | 说明 |
|------|--------|------|------|
| D-M03-01 | `configs/train_lora_qwen2_5_1_5b_insurance.yaml` | ✅ 已创建 | 全量 SFT 训练配置，lora_rank=16 |
| D-M03-02 | `configs/train_dpo_qwen2_5_1_5b_insurance.yaml` | ✅ 已创建 | 全量 DPO 训练配置，pref_beta=0.1 |
| D-M03-03 | `configs/smoke_sft_insurance.yaml` | ✅ 已创建 | SFT 烟雾测试，max_steps=10 |
| D-M03-04 | `configs/smoke_dpo_insurance.yaml` | ✅ 已创建 | DPO 烟雾测试，max_steps=10 |
| D-M03-05 | `configs/deepspeed/zero3.json` | ✅ 已创建 | ZeRO3 + CPU offload |
| D-M03-06 | `data/insurance/dataset_info.json` | ✅ 已创建 | LLaMA-Factory 数据集映射 |
| D-M03-07 | SFT LoRA 权重 | ✅ 完成 | `saves/qwen2_5_1_5b/insurance_sft_v1/lora/` (71M) |
| D-M03-08 | DPO LoRA 权重 | ✅ 完成 | `saves/qwen2_5_1_5b/insurance_dpo_v1.2/lora/` |
| D-M03-09 | SFT Merge 模型 | ✅ 完成 | `saves/qwen2_5_1_5b/insurance_sft_v1/lora_merged/` |
| D-M03-10 | DPO Merge 模型 | ✅ 完成 | `merged_models/qwen2_5_1_5b_insurance_dpo_v1.2/` (2.9GB) |

---

## 2. 烟雾测试结果（2026-06-10）

### 2.1 SFT 烟雾测试 — ✅ PASS

| 项目 | 详情 |
|------|------|
| 配置文件 | `configs/smoke_sft_insurance.yaml` |
| 训练步数 | 10 steps |
| 最终 loss | 1.9127 |
| 吞吐量 | 3.728 samples/sec |
| 耗时 | ~2.68s |
| GPU | 单卡 RTX 5090 |
| 输出路径 | `saves/smoke/sft_insurance/` |

命令：
```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/smoke_sft_insurance.yaml
```

### 2.2 DPO 烟雾测试 — ✅ PASS

| 项目 | 详情 |
|------|------|
| 配置文件 | `configs/smoke_dpo_insurance.yaml` |
| 训练步数 | 10 steps |
| 最终 loss | 0.6908 |
| rewards/chosen | 正常追踪 |
| rewards/margins | 正常追踪 |
| 吞吐量 | 3.428 samples/sec |
| 耗时 | ~2.91s |
| GPU | 单卡 RTX 5090 |
| 输出路径 | `saves/smoke/dpo_insurance/` |

命令：
```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/smoke_dpo_insurance.yaml
```

---

## 3. 训练配置关键参数

### 3.1 SFT 配置

| 参数 | 烟雾测试 | 全量训练 |
|------|----------|----------|
| `stage` | sft | sft |
| `finetuning_type` | lora | lora |
| `lora_rank` | 8 | 16 |
| `lora_alpha` | 16 | 32 |
| `lora_target` | q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj | 同左 |
| `per_device_train_batch_size` | 1 | 2 |
| `gradient_accumulation_steps` | 1 | 4 |
| `max_steps` / `num_train_epochs` | 10 steps | 3 epochs |
| `learning_rate` | 5e-5 | 5e-5 |
| `lr_scheduler_type` | cosine | cosine |
| `warmup_ratio` | 0.1 | 0.1 |
| `bf16` | true | true |

### 3.2 DPO 配置

| 参数 | 烟雾测试 | 全量训练 |
|------|----------|----------|
| `stage` | dpo | dpo |
| `pref_beta` | 0.1 | 0.1 |
| `pref_loss` | sigmoid | sigmoid |
| `max_steps` / `num_train_epochs` | 10 steps | 2 epochs |
| `learning_rate` | 1e-5 | 1e-5 |
| 其他 LoRA 参数 | 同 SFT 烟雾 | 同 SFT 全量 |

---

## 4. 遇到并解决的问题

| # | 问题 | 解决方式 |
|---|------|----------|
| 1 | `trust_remote_code` 不被 LLaMA-Factory 0.9.1 识别 | 从所有 yaml 中移除该参数 |
| 2 | HuggingFace 不可达 | 改用本地模型路径 `/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct` |
| 3 | NCCL SIGABRT 在 2 GPU 模式下（RTX 5090 sm_120 兼容性问题） | 烟雾测试改用单 GPU (`CUDA_VISIBLE_DEVICES=0`) |
| 4 | `dpo_beta`/`dpo_loss_type`/`dpo_max_length` 不被识别 | 改为 LLaMA-Factory 0.9.1 的参数名：`pref_beta`/`pref_loss`，移除 max_length 参数 |
| 5 | DPO 数据集 `not applicable in current training stage` | 在 `dataset_info.json` 中添加 `"ranking": true` |

---

## 5. 全量训练结果（2026-06-11）

### 5.1 SFT 全量训练 — ✅ PASS

| 项目 | 详情 |
|------|------|
| 配置文件 | `configs/train_lora_qwen2_5_1_5b_insurance.yaml` |
| 样本数 | 7,096 条 |
| Epochs | 3 |
| 总步数 | 663 steps |
| 最终 loss | **0.0777** |
| 吞吐量 | 20.9 samples/sec |
| 训练时间 | 16 分 58 秒 |
| GPU | 单卡 RTX 5090 (8,139 MiB) |
| 输出路径 | `saves/qwen2_5_1_5b/insurance_sft_v1/lora/` |

命令：
```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/train_lora_qwen2_5_1_5b_insurance.yaml
```

### 5.2 DPO 全量训练 — ✅ PASS

| 项目 | 详情 |
|------|------|
| 配置文件 | `configs/train_dpo_qwen2_5_1_5b_insurance.yaml` |
| 基座模型 | SFT LoRA merge 后 HF 模型 |
| 样本数 | 7,223 条 |
| Epochs | 2 |
| 总步数 | 450 steps |
| 最终 loss | **0.0538** |
| rewards/margins | **9.47**（验收标准 ≥0.5，超额 18.9×） |
| rewards/accuracies | **1.0**（100% preference 正确率） |
| 吞吐量 | 9.5 samples/sec |
| 训练时间 | 25 分 27 秒 |
| GPU | 单卡 RTX 5090 (11,933 MiB) |
| 输出路径 | `saves/qwen2_5_1_5b/insurance_dpo_v1.2/lora/` |

命令：
```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/train_dpo_qwen2_5_1_5b_insurance.yaml
```

### 5.3 模型 Merge — ✅ 完成

| 步骤 | 命令 | 产物 |
|------|------|------|
| SFT LoRA Merge | `llamafactory-cli export --model ... --adapter .../sft/lora --export_dir .../lora_merged` | `saves/.../insurance_sft_v1/lora_merged/` |
| DPO LoRA Merge | `llamafactory-cli export --model .../lora_merged --adapter .../dpo/lora --export_dir merged_models/...` | `merged_models/qwen2_5_1_5b_insurance_dpo_v1.2/` (2.9GB) |

---

## 6. 已知缺口

| # | 缺口 | 说明 |
|---|------|------|
| 1 | ~~数据量严重不足~~ | ✅ 已解决：M02 产出 7,183 DPO + 7,048 SFT，超额达成 |
| 2 | **NCCL 2 GPU 训练** | RTX 5090 sm_120 与 NCCL 2.26 不兼容，全量训练使用单卡模式 |
| 3 | ~~全量训练未执行~~ | ✅ 已完成：SFT 3 epochs + DPO 2 epochs 全部跑通 |
| 4 | vLLM 与 LLaMA-Factory 版本冲突 | vLLM 从 llm 环境卸载（需 transformers>=4.48），推理留在独立 `vllm` env |

---

## 7. 环境信息速查

```
服务器：server2 (connect.bjb2.seetacloud.com:16531)
GPU：    2×NVIDIA GeForce RTX 5090 (32GB)
模型：   /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct
代码：   /root/autodl-tmp/agroup-dpo
数据：   /root/autodl-tmp/agroup-dpo/data/insurance/
SFT:     /root/autodl-tmp/agroup-dpo/saves/qwen2_5_1_5b/insurance_sft_v1/
DPO:     /root/autodl-tmp/agroup-dpo/saves/qwen2_5_1_5b/insurance_dpo_v1.2/
Merge:   /root/autodl-tmp/agroup-dpo/merged_models/qwen2_5_1_5b_insurance_dpo_v1.2/
```

---

## 8. 结论

M03 阶段**全部完成**。SFT 和 DPO 全量训练均已通过（单卡模式），最终对齐模型已 merge 产出（2.9GB safetensors）。

**关键成果**：
- DPO rewards/margins = **9.47**（远超 0.5 验收标准）
- rewards/accuracies = **100%**（模型 100% 正确偏好 chosen over rejected）
- SFT loss 从初值收敛至 0.0777
- 总训练时间：SFT 17min + DPO 25min = 42min

**下一步：M04 自研分布式 Trainer 或 M05 推理加速与评测。**
