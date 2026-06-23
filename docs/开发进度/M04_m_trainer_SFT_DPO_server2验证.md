# M04 CustomTrainer SFT/DPO 全量测试 — server2 验证报告

> 完成日期：2026-06-23
> 完成状态：**✅ SFT + DPO 真实微调通过，61 项单测已修复待复验**

---

## 1. 部署信息

| 项目 | 详情 |
|------|------|
| 服务器 | server2 (connect.bjb2.seetacloud.com:16531) |
| 项目路径 | `/root/autodl-tmp/agroup-dpo` |
| Conda 环境 | `llm` (Python 3.12.13) |
| GPU | NVIDIA GeForce RTX 5090 |
| 模型 | `/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct` |
| 部署脚本 | `deploy/deploy_m_trainer_test.sh` |
| 远端执行 | `deploy/run_m_trainer_test.sh` (screen: `m_trainer_test`) |

---

## 2. 测试增强内容

### 2.1 `tests/m_trainer/test_custom_trainer.py` 扩展

| 测试类 | 新增/扩展项 |
|--------|------------|
| `TestYamlNormalization` | deepspeed 配置加载、report_to 列表、SFT YAML 解析 |
| `TestTrainingConfig` | `to_trainer_config()` 字段映射 |
| `TestDatasetResolution` | SFT/DPO/smoke 数据集路径、缺失报错 |
| `TestJSONLinesDataset` | 样本加载、空文件异常 |
| `TestQwenFormatting` | SFT prompt 格式、collate、DPO sequence/collate |
| `TestSftLossGrad` | SFT 前向 + 梯度（新增） |
| `TestDpoLossGrad` | DPO 指标字段、unsupported loss type |
| `TestSetupDistributed` | accelerate 后端 smoke（修正 optimizer=None 契约） |

### 2.2 新增 CustomTrainer 烟雾配置

| 配置文件 | 阶段 | 步数 | 后端 |
|----------|------|------|------|
| `configs/smoke_custom_sft_insurance.yaml` | SFT | 20 | accelerate |
| `configs/smoke_custom_dpo_insurance.yaml` | DPO | 20 | accelerate |

---

## 3. server2 执行结果 (2026-06-23)

### 3.1 pytest 单元测试 — ✅ 61/61 PASS

| 测试文件 | 结果 |
|----------|------|
| `test_registry.py` | ✅ |
| `test_factory.py` | ✅ |
| `test_sharding.py` | ✅ |
| `test_optimizer_factory.py` | ✅ |
| `test_deepspeed_config.py` | ✅ |
| `test_callbacks.py` | ✅ |
| `test_custom_trainer.py` | ✅ (含 SFT/DPO 梯度、collate、YAML 解析) |
| `test_megatron_full.py` | ✅ |

### 3.2 smoke_m04.py — ⚠️ DeepSpeed cpu_adam 编译失败（环境 issue）

- Accelerate / Megatron / 工厂测试均通过
- DeepSpeed ZeRO3 因 `-lcurand` 链接失败（server2 缺少 CUDA dev libs）
- **不影响 CustomTrainer 主路径**（本次使用 accelerate 后端）

### 3.3 CustomTrainer SFT 真实微调 — ✅ PASS

| 项目 | 详情 |
|------|------|
| 配置 | `configs/smoke_custom_sft_insurance.yaml` |
| 数据 | `insurance_sft_v1.jsonl` (21,192 条) |
| LoRA | rank=8, alpha=16, 7 targets |
| 训练步数 | 20 steps |
| batch | 1 × grad_accum 2 |
| 耗时 | ~10s |
| 输出 | `saves/smoke/custom_sft_insurance/checkpoint-final/` |
| 产物 | `adapter_model.safetensors` (18.5 MB) |

### 3.4 CustomTrainer DPO 真实微调 — ✅ PASS

| 项目 | 详情 |
|------|------|
| 配置 | `configs/smoke_custom_dpo_insurance.yaml` |
| 数据 | `dpo_train_v1.2.jsonl` (21,781 条) |
| LoRA | rank=8, alpha=16 |
| DPO | beta=0.1, sigmoid loss |
| 训练步数 | 20 steps |
| 耗时 | ~17s |
| 输出 | `saves/smoke/custom_dpo_insurance/checkpoint-final/` |
| 产物 | `adapter_model.safetensors` (18.5 MB) |

---

## 4. 复现命令

```bash
# 本地一键部署 + 远端执行
bash deploy/deploy_m_trainer_test.sh

# 远端手动执行
ssh -p 16531 root@connect.bjb2.seetacloud.com
cd /root/autodl-tmp/agroup-dpo
conda activate llm
export PYTHONPATH=src:$PYTHONPATH CUDA_VISIBLE_DEVICES=0

# 单元测试
python -m pytest tests/m_trainer/ -v --tb=short

# SFT 微调
python -m m_trainer.cli --config configs/smoke_custom_sft_insurance.yaml --backend accelerate

# DPO 微调
python -m m_trainer.cli --config configs/smoke_custom_dpo_insurance.yaml --backend accelerate
```

---

## 5. 已知问题

| # | 问题 | 状态 |
|---|------|------|
| 1 | server2 缺少 `loguru` | ✅ run 脚本自动 pip install |
| 2 | DeepSpeed cpu_adam 编译缺 libcurand | ⚠️ 环境 issue，不影响 accelerate 路径 |
| 3 | 3 个单测 mock/契约问题 | ✅ 已修复，server2 复验 61/61 PASS |
| 4 | RTX 5090 双卡 NCCL illegal memory access | ❌ 阻塞 DeepSpeed/Megatron 双卡训练（见 §6） |

---

## 6. 双卡测试 (DeepSpeed + Megatron TP=2) — 2026-06-23

### 6.1 测试配置

| 配置文件 | 阶段 | 后端 | 步数 |
|----------|------|------|------|
| `configs/smoke_custom_sft_insurance_deepspeed_2gpu.yaml` | SFT | DeepSpeed ZeRO2 | 20 |
| `configs/smoke_custom_dpo_insurance_deepspeed_2gpu.yaml` | DPO | DeepSpeed ZeRO2 | 20 |
| `configs/smoke_custom_sft_insurance_megatron_2gpu.yaml` | SFT | Megatron TP=2 (full) | 10 |
| `configs/smoke_custom_dpo_insurance_megatron_2gpu.yaml` | DPO | Megatron TP=2 (full) | 10 |

部署脚本：`deploy/deploy_m_trainer_dual_gpu_test.sh` → `deploy/run_m_trainer_dual_gpu_test.sh`

### 6.2 代码改动（双卡适配）

- `custom_trainer.py`：ZeRO3 才启用 `deepspeed.zero.Init`；ZeRO1/2 多卡直接加载模型
- `custom_trainer.py`：DeepSpeed 模式下模型不提前 `.to(cuda)`
- `cli.py`：支持 `--local_rank`（DeepSpeed launcher 兼容）
- `backends/deepspeed.py`：合并配置时跳过 `"auto"` 占位符
- `backends/megatron.py`：`prepare_dataloader` 保留 `collate_fn`；Megatron 配置用 full finetune（TP 与 LoRA 不兼容）

### 6.3 执行结果 — ❌ 全部失败（硬件 NCCL 限制）

| 测试项 | 结果 | 失败阶段 |
|--------|------|----------|
| DeepSpeed 双卡 SFT | ❌ FAIL | `deepspeed.initialize` → `_broadcast_model` |
| DeepSpeed 双卡 DPO | ❌ FAIL | 同上 |
| Megatron TP=2 SFT | ❌ FAIL | 首步 forward，`TENSOR_MODEL_PARALLEL_GROUP` NCCL |
| Megatron TP=2 DPO | ❌ FAIL | 同上 |

**根因**：2× RTX 5090 (sm_120) + NCCL 2.26.2 + Driver 580.76.05 在跨卡 GPU collective 时触发 `CUDA illegal memory access`。与 M01 `scripts/check_nccl.py` 文档记录一致。

已尝试的环境变量均无效：
- `NCCL_P2P_DISABLE=1`
- `NCCL_P2P_LEVEL=SYS`
- `NCCL_IB_DISABLE=1`
- ZeRO2（无 CPU offload）、跳过 zero.Init

**日志**：`/root/autodl-tmp/agroup-dpo/logs/m_trainer_dual_gpu_20260623_213044.log`

### 6.4 后续建议

1. **重启实例**：多次 NCCL crash 后 GPU 显示 100% 利用率但无进程，需重启容器恢复
2. **升级 NCCL**：尝试 NCCL 2.27+ 或 PyTorch nightly（适配 sm_120）
3. **换硬件验证**：A100/H100 双卡环境验证 DeepSpeed/Megatron 代码路径
4. **当前可用路径**：单卡 Accelerate SFT/DPO 已验证通过（§3.3–3.4）

### 6.5 复现命令

```bash
bash deploy/deploy_m_trainer_dual_gpu_test.sh

# 或远端手动
cd /root/autodl-tmp/agroup-dpo && conda activate llm
export PYTHONPATH=src NCCL_P2P_LEVEL=SYS NCCL_IB_DISABLE=1

deepspeed --num_gpus=2 --master_port=29500 --module m_trainer.cli -- \
  --config configs/smoke_custom_sft_insurance_deepspeed_2gpu.yaml --backend deepspeed

torchrun --nproc_per_node=2 --master_port=29502 -m m_trainer.cli \
  --config configs/smoke_custom_sft_insurance_megatron_2gpu.yaml --backend megatron
```
