# 如何部署 AGroup-DPO

> 面向开源社区的环境搭建与端到端执行指南  
> 关联模块：M01 基础设施 · 统一 CLI `copaw-dpo`

---

## 目录

1. [项目做什么](#1-项目做什么)
2. [整体执行流程](#2-整体执行流程)
3. [硬件与软件要求](#3-硬件与软件要求)
4. [方式一：Conda 本地部署（推荐入门）](#4-方式一conda-本地部署推荐入门)
5. [方式二：Docker 分层镜像部署](#5-方式二docker-分层镜像部署)
6. [下载基座模型](#6-下载基座模型)
7. [统一 CLI 入口](#7-统一-cli-入口)
8. [一键部署脚本](#8-一键部署脚本)
9. [烟雾测试验收](#9-烟雾测试验收)
10. [端到端快速跑通](#10-端到端快速跑通)
11. [常见问题](#11-常见问题)
12. [延伸阅读](#12-延伸阅读)

---

## 1. 项目做什么

AGroup-DPO 是一条**保险领域 DPO 偏好对齐**的完整流水线，核心链路如下：

```
原始业务数据 → 自动生成 DPO/SFT 训练集 → LoRA SFT → LoRA DPO → 合并导出 → 推理服务 → 自动评测
```

五个子模块分工明确：

| 模块 | 路径 | 职责 |
|------|------|------|
| **M-DATA** | `src/m_data/` | 从条款/FAQ/工单生成 chosen/rejected 配对数据 |
| **M-TRAINER** | `src/m_trainer/` | LoRA SFT / DPO 训练（LLaMA-Factory + 自研分布式后端） |
| **M-MERGE** | `src/m_merge/` | LoRA adapter 与基座权重合并导出 |
| **M-INFER** | `src/m_infer/` | vLLM / xinference 推理与 HTTP 服务 |
| **M-EVAL** | `src/m_eval/` | Accuracy / BLEU-4 / ROUGE-L + 延迟统计 |

---

## 2. 整体执行流程

推荐按以下顺序执行（每步对应一份教程）：

```mermaid
flowchart LR
    A[部署环境] --> B[合成训练数据]
    B --> C[LoRA SFT]
    C --> D[合并 SFT LoRA]
    D --> E[LoRA DPO]
    E --> F[合并 DPO LoRA]
    F --> G[推理 / 评测]
```

| 步骤 | 教程 | 核心命令 |
|------|------|----------|
| 0. 部署 | 本文 | `pip install -e .` |
| 1. 数据 | [How-to-synthesize-training-data.md](./How-to-synthesize-training-data.md) | `copaw-dpo data --config ...` |
| 2. SFT | [How-to-run-SFT?](./How-to-run-SFT%3F) | `copaw-dpo train --config ...` |
| 3. DPO | [How-to-run-DPO.md](./How-to-run-DPO.md) | `copaw-dpo train --config ...` |
| 4. 评测 | [how-to-evaluate-model-performance.md](./how-to-evaluate-model-performance.md) | `copaw-dpo evaluate --config ...` |

DPO 理论背景可参考 [DPO-Tutorial.md](./DPO-Tutorial.md)。

---

## 3. 硬件与软件要求

### 3.1 硬件

| 项目 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | 1× RTX 4090 (24GB) | 2× A100-80G 或 2× RTX 4090/5090 |
| 显存 | 单卡可跑 smoke test | 双卡可跑完整 SFT + DPO |
| 磁盘 | ≥ 50 GB | ≥ 100 GB（含模型 + 数据 + checkpoint） |
| CUDA | ≥ 12.4 | 12.8 / 13.0 |
| 驱动 | ≥ 535 | ≥ 580（RTX 5090） |

### 3.2 软件版本（与 `requirements.txt` 对齐）

| 依赖 | 版本 |
|------|------|
| Python | 3.10 – 3.12 |
| PyTorch | 2.7.1+cu128 |
| transformers | 4.43.4 |
| LLaMA-Factory | 0.9.1 |
| DeepSpeed | 0.14.4 |
| vLLM | ≥ 0.8.5, < 0.9.0 |
| 基座模型 | Qwen2.5-1.5B-Instruct |

> **版本兼容提示**：vLLM 0.5.x 强依赖 torch 2.4.0，会覆盖 PyTorch 2.7.1；请使用 vLLM ≥ 0.8.5。若 vLLM ≥ 0.22.1，需设置 `export VLLM_USE_FLASHINFER_SAMPLER=0`。

---

## 4. 方式一：Conda 本地部署（推荐入门）

### 4.1 克隆仓库

```bash
git clone <repo-url> agroup-dpo
cd agroup-dpo
```

### 4.2 创建 Conda 环境

```bash
conda create -n llm python=3.12 -y
conda activate llm
```

### 4.3 安装 PyTorch

```bash
pip install --no-cache-dir \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
```

验证 GPU：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
```

### 4.4 安装项目依赖

**最小安装（训练 + 数据）：**

```bash
pip install -e .
pip install deepspeed==0.14.4
```

**完整安装（含推理 + 评测）：**

```bash
pip install -e ".[all]"
# 或分步安装
pip install -e ".[train,inference,evaluation]"
```

**数据流水线额外依赖（PolicyStore 条款检索修复）：**

```bash
pip install "pymilvus[milvus_lite]>=2.4.0" "sentence-transformers>=2.7.0"
```

国内下载 HuggingFace 模型时建议：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 4.5 验证安装

```bash
copaw-dpo --help
copaw-dpo train --help
PYTHONPATH=src python -m pytest tests/m_data/ -q
```

---

## 5. 方式二：Docker 分层镜像部署

项目采用**基础镜像 + 功能 layer** 的分层设计，按需构建，避免单个超大镜像。

```
copaw-dpo-base:latest
    ├── copaw-dpo-deepspeed:latest   # DeepSpeed ZeRO 训练
    ├── copaw-dpo-fsdp:latest        # PyTorch FSDP
    ├── copaw-dpo-megatron:latest    # Megatron-LM 张量并行
    ├── copaw-dpo-accelerate:latest  # 单机多卡调试
    ├── copaw-dpo-infer-vllm:latest  # vLLM 推理
    ├── copaw-dpo-infer-xinfer:latest # xinference 推理
    └── copaw-dpo-eval:latest        # 评测
```

### 5.1 构建镜像

```bash
# 基础镜像（约 25 min）
docker build -f deploy/Dockerfile.base -t copaw-dpo-base:latest .

# 训练后端 layer（约 5 min/个）
for backend in deepspeed fsdp megatron accelerate; do
    docker build -f deploy/Dockerfile.$backend -t copaw-dpo-$backend:latest .
done

# 推理 layer
docker build -f deploy/Dockerfile.infer-vllm   -t copaw-dpo-infer-vllm:latest .
docker build -f deploy/Dockerfile.infer-xinfer -t copaw-dpo-infer-xinfer:latest .

# 评测 layer
docker build -f deploy/Dockerfile.eval -t copaw-dpo-eval:latest .
```

### 5.2 docker-compose 启动

```bash
# DeepSpeed 训练
docker compose -f deploy/docker-compose.yml up train-deepspeed

# vLLM 推理服务
docker compose -f deploy/vllm-compose.yml up -d
curl http://localhost:8080/v1/models

# xinference 推理服务
docker compose -f deploy/xinference-compose.yml up -d
curl http://localhost:9997/v1/models
```

### 5.3 容器卷挂载约定

| 容器路径 | 宿主机路径 | 用途 |
|----------|------------|------|
| `/workspace/src` | `./src` | 源码 |
| `/workspace/configs` | `./configs` | 配置 |
| `/workspace/data` | `./data` | 训练/评测数据 |
| `/workspace/saves` | `./saves` | checkpoint |
| `/shared` | `/shared` | 跨节点共享存储（可选） |

详细说明见 [`deploy/README.md`](../../deploy/README.md)。

---

## 6. 下载基座模型

DPO 流水线默认使用 **Qwen2.5-1.5B-Instruct**：

```bash
# HuggingFace CLI
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir /path/to/models/Qwen2.5-1.5B-Instruct

# 或 ModelScope（国内）
modelscope download --model Qwen/Qwen2.5-1.5B-Instruct \
    --local_dir /path/to/models/Qwen2.5-1.5B-Instruct
```

下载完成后，在训练 yaml 中将 `model_name_or_path` 改为本地路径，例如：

```yaml
model_name_or_path: /path/to/models/Qwen2.5-1.5B-Instruct
```

---

## 7. 统一 CLI 入口

安装后可用 `copaw-dpo` 命令访问全部功能：

```
copaw-dpo
├── train      → 训练（LLaMA-Factory / CustomTrainer 后端）
├── data       → DPO/SFT 数据集生成
├── infer      → 推理（vLLM / xinference）
├── evaluate   → 评测（Accuracy / BLEU-4 / ROUGE-L）
└── merge      → LoRA adapter 合并导出
```

```bash
copaw-dpo --help
copaw-dpo train --help
copaw-dpo data --help
copaw-dpo infer --help
copaw-dpo evaluate --help
copaw-dpo merge --help
```

---

## 8. 一键部署脚本

`deploy/` 目录提供各阶段的自动化脚本，适合在 GPU 服务器上一键验收：

| 脚本 | 用途 |
|------|------|
| `deploy/run_m01_smoke.sh` | M01 环境 + 5 项烟雾测试（评测/vLLM/SFT/NCCL） |
| `deploy/run_dpo_data_smoke.sh` | M02 数据流水线全量集成测试 |
| `deploy/run_m_trainer_test.sh` | M04 CustomTrainer SFT/DPO 单卡测试 |
| `deploy/run_m_trainer_dual_gpu_test.sh` | DeepSpeed + Megatron 双卡测试 |
| `deploy/run_merge_and_eval.sh` | LoRA 合并 + 全量 1700 条评测 |
| `deploy/run_m_infer_test.sh` | 推理模块全面测试 |
| `deploy/run_m_eval_test.sh` | 评测模块全面测试 |

**远端服务器部署示例**（AutoDL / 云 GPU）：

```bash
# 在服务器上
cd /root/autodl-tmp/agroup-dpo
bash deploy/run_m01_smoke.sh
```

本地开发 → 远端执行的推荐流程：

```bash
# 本地 rsync 推送代码（排除 saves/ 和 .git/）
rsync -avz --delete \
    --exclude='saves/' --exclude='.git/' --exclude='__pycache__/' \
    ./agroup-dpo/ user@gpu-server:/root/autodl-tmp/agroup-dpo/

# 远端执行
ssh user@gpu-server "cd /root/autodl-tmp/agroup-dpo && bash deploy/run_m01_smoke.sh"
```

---

## 9. 烟雾测试验收

M01 阶段通过以下 5 项测试即表示环境就绪：

| # | 测试项 | 命令 | 通过标准 |
|---|--------|------|----------|
| 1 | 评测依赖 | `python scripts/smoke_eval.py` | 单元测试 PASS |
| 2 | vLLM 推理 | `python scripts/smoke_vllm.py --model <model_path>` | 输出非空文本 |
| 3 | xinference | `bash scripts/smoke_xinfer.sh` | HTTP 推理成功 |
| 4 | SFT 训练 | `llamafactory-cli train configs/smoke_lora_qwen2_5_1_5b.yaml` | 5 step loss 下降 |
| 5 | NCCL 通信 | `torchrun --nproc_per_node=2 scripts/check_nccl.py` | 跨卡 barrier 通过 |

NCCL 测试示例：

```bash
torchrun --nproc_per_node=2 --nnodes=1 \
    --master_addr=127.0.0.1 --master_port=29500 \
    scripts/check_nccl.py --size_mb 64
```

---

## 10. 端到端快速跑通

环境就绪后，可用 smoke 配置在 **30 分钟内** 跑通完整链路：

```bash
# 1. 生成训练数据（仓库已含示例数据，也可重新生成）
copaw-dpo data --config configs/data/insurance_dpo_gen.yaml --dry-run --verbose
copaw-dpo data --config configs/data/insurance_dpo_gen.yaml

# 2. SFT 烟雾训练（20 steps，单卡）
copaw-dpo train --config configs/smoke_custom_sft_insurance.yaml --backend accelerate

# 3. DPO 烟雾训练（20 steps，单卡）
copaw-dpo train --config configs/smoke_custom_dpo_insurance.yaml --backend accelerate

# 4. 合并 LoRA 并导出
copaw-dpo merge \
    --base /path/to/models/Qwen2.5-1.5B-Instruct \
    --adapter saves/smoke/custom_dpo_insurance/checkpoint-final \
    --output merged_models/qwen2_5_1_5b_insurance_dpo_v1.2

# 5. 推理验证
copaw-dpo infer --backend vllm \
    --model merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \
    --prompts "保险等待期是什么？"

# 6. 评测
copaw-dpo evaluate --config configs/eval.yaml \
    --model merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \
    --output reports/eval_report_smoke
```

完整生产训练请改用 `configs/train_lora_qwen2_5_1_5b_insurance.yaml` 和 `configs/train_dpo_qwen2_5_1_5b_insurance.yaml`，详见 SFT / DPO 教程。

---

## 11. 常见问题

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| `nvcc` 找不到 | Docker base 镜像 tag 错误 | 使用 `nvidia/cuda:12.4.1-cudnn8-devel-ubuntu22.04` |
| torchrun OOM / 通信失败 | `--shm-size` 太小 | Docker 加 `--shm-size=16g` |
| vLLM `Illegal memory access` | GPU 驱动过旧 | 升级驱动 ≥ 535 |
| bitsandbytes 无法加载 | 版本与 CUDA 不匹配 | 使用 `bitsandbytes==0.45.5`（含 cu128 wheel） |
| LLaMA-Factory 多卡 DDP 崩溃 | NCCL + 新架构 GPU 兼容 | smoke 阶段先用 `CUDA_VISIBLE_DEVICES=0` 单卡 |
| PolicyStore 初始化慢 | 首次下载 BGE 嵌入模型 | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
| vLLM 与 transformers 冲突 | 版本不兼容 | 保持 transformers 4.43.4 + vLLM 0.8.x |

---

## 12. 延伸阅读

| 文档 | 说明 |
|------|------|
| [README.md](../../README.md) | 项目总览 |
| [deploy/README.md](../../deploy/README.md) | Docker 镜像与 compose 详情 |
| [src/m_data/README.md](../../src/m_data/README.md) | 数据流水线模块 |
| [src/m_infer/README.md](../../src/m_infer/README.md) | 推理服务与 RAG 接口 |
| [src/m_eval/README.md](../../src/m_eval/README.md) | 评测指标与报告 |
| [docs/项目分阶段方案/](../../docs/项目分阶段方案/) | M01–M05 完整技术方案 |
