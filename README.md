# AGroup DPO

> 保险业务 DPO 训练 / 推理 / 评测一体化项目

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.7.1-red)](https://pytorch.org/)
[![LLaMA-Factory](https://img.shields.io/badge/LLaMA--Factory-0.9.1-orange)](https://github.com/hiyouga/LLaMA-Factory)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-0.14.4-green)](https://www.deepspeed.ai/)
[![vLLM](https://img.shields.io/badge/vLLM-0.8.5-blueviolet)](https://github.com/vllm-project/vllm)

---

## 项目简介

AGroup DPO 是一个面向保险业务场景的大模型偏好对齐工程化方案，基于 **Qwen2.5-1.5B-Instruct** 完成从数据生成、LoRA 微调、DPO 对齐训练、分布式训练框架、推理加速到自动化评测的全链路交付。

核心能力：
- **DPO 数据流水线**：从保险条款/FAQ/工单中自动构造 chosen/rejected 偏好对
- **多后端分布式训练**：DeepSpeed ZeRO3 / FSDP / Megatron / accelerate 统一抽象
- **推理后端可切换**：vLLM / xinference 双后端，配置驱动一键切换
- **自动化评测**：Accuracy / BLEU-4 / ROUGE-L + 推理时间统计

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      AGroup DPO 总体架构                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ M-DATA   │──▶│ M-LORA   │──▶│ M-DPO    │──▶│ M-TRAINER│    │
│  │ DPO 数据 │   │ LoRA 微调│   │ 对齐训练 │   │ 分布式   │    │
│  │ 流水线   │   │          │   │          │   │ Trainer  │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│       │              │              │              │           │
│       ▼              ▼              ▼              ▼           │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              模型权重 (HF safetensors)               │      │
│  └──────────────────────────────────────────────────────┘      │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                   │
│  │ M-MERGE  │──▶│ M-INFER  │──▶│ RAG 端   │                   │
│  │ 模型导出 │   │ 推理加速 │   │ 对接     │                   │
│  └──────────┘   └──────────┘   └──────────┘                   │
│                      │                                          │
│                      ▼                                          │
│                 ┌──────────┐                                    │
│                 │ M-EVAL   │                                    │
│                 │ 评测模块 │                                    │
│                 └──────────┘                                    │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  横切关注点：PII 脱敏 | 配置管理 | 日志/监控 | 合规校验         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | ≥ 3.10 | |
| PyTorch | 2.7.1+cu128 | |
| CUDA | ≥ 12.4 | |
| GPU | 2×A100-80G / 2×RTX 5090 | 训练需要 |
| LLaMA-Factory | 0.9.1 | 训练流水线编排 |
| DeepSpeed | 0.14.4 | 默认分布式后端 |

### 安装

```bash
# 克隆仓库
cd /root/autodl-tmp
git clone <repo-url> agroup-dpo
cd agroup-dpo

# 创建 conda 环境
conda create -n llm python=3.12
conda activate llm

# 安装核心依赖
pip install -e .

# 安装推理依赖（可选）
pip install -e ".[inference]"

# 安装评测依赖（可选）
pip install -e ".[evaluation]"

# 安装全部依赖
pip install -e ".[all]"
```

### 模型准备

```bash
# 下载基座模型
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct
```

---

## 模块总览

| 模块 | 路径 | 职责 |
|------|------|------|
| **M-DATA** | `src/m_data/` | DPO/SFT 数据集生成流水线（采集→脱敏→配对→校验→导出） |
| **M-TRAINER** | `src/m_trainer/` | 多后端分布式 Trainer（DeepSpeed/FSDP/Megatron/accelerate） |
| **M-INFER** | `src/m_infer/` | vLLM/xinference 双后端推理加速 + RAG 对接服务 |
| **M-EVAL** | `src/m_eval/` | Accuracy/BLEU/ROUGE 评测 + 推理时间统计 |
| **M-MERGE** | `src/m_merge/` | LoRA 权重与基座模型合并导出 |

---

## 核心功能

### 1. DPO 数据集生成 (M-DATA)

从保险业务语料自动构造 chosen/rejected 偏好对，支持三种策略：

```
[Collector] → [Normalizer] → [PIIScrubber] → [Filter]
                                                  │
                        ┌─────────────────────────┤
                        ▼                         ▼
                 [PairBuilder]              [SFTBuilder]
                 (DPO 配对)                 (SFT 样本)
                        │                         │
                        ▼                         ▼
                  [Validator]              [SFT Validator]
                        │                         │
                        ▼                         ▼
                  [Exporter]                [Exporter]
```

**三种配对策略**：

| 策略 | 标识 | 说明 |
|------|------|------|
| 规则硬负例 | `rule_based` | 基于保险条款强制构造合规/违规答案对 |
| LLM-as-Judge | `llm_judge` | Qwen2.5-7B 对比 RAG 答案与专家标注 |
| 检索差异 | `retrieval_diff` | 完整索引 vs 截断索引 RAG 答案对比 |

**数据源**：保险条款 PDF/HTML、业务 FAQ、历史工单

**质量保障**：PII 脱敏、6 类规则校验、chosen/rejected 相似度去重

```bash
# 生成 DPO 数据集
PYTHONPATH=src python -m m_data.cli \
    --config configs/data/insurance_dpo_gen.yaml

# 干跑模式（不写文件）
PYTHONPATH=src python -m m_data.cli \
    --config configs/data/insurance_dpo_gen.yaml \
    --dry-run --verbose
```

### 2. 多后端分布式训练 (M-TRAINER)

同一份训练配置可在 4 种分布式后端间切换，无需改动业务代码：

| 后端 | 适用场景 | 状态 |
|------|----------|------|
| **DeepSpeed ZeRO3** | 默认主路径，显存节省显著 | ✅ 生产 |
| **PyTorch FSDP** | PyTorch 原生备选 | ✅ 可用 |
| **Megatron-LM** | 7B+ 模型预研接口 | 🔧 预留 |
| **accelerate** | 单机多卡调试/CI | ✅ 可用 |

```bash
# LoRA SFT 微调（默认 DeepSpeed）
copaw-dpo train-sft --config configs/train_lora_qwen2_5_1_5b_insurance.yaml

# DPO 对齐训练（切换到 FSDP）
copaw-dpo train-dpo \
    --config configs/train_dpo_qwen2_5_1_5b_insurance.yaml \
    --override trainer.distributed_backend=fsdp
```

### 3. 双后端推理加速 (M-INFER)

vLLM 与 xinference 可切换封装，配置文件 `infer.backend: vllm | xinference` 一键切换：

```python
from m_infer import build_infer_backend, InferRequest

backend = build_infer_backend("vllm", "merged_models/qwen2_5_1_5b_insurance_dpo_v1.2")
resp = backend.infer(InferRequest(prompt="保险等待期是什么？", max_new_tokens=128))
print(resp.text)
```

**HTTP 推理服务**（与司内 RAG 端对接）：

```bash
python -m m_infer.server \
    --backend vllm \
    --model-path merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \
    --host 0.0.0.0 --port 8080

# 保险问答
curl -X POST http://127.0.0.1:8080/v1/insurance/qa \
  -H "Content-Type: application/json" \
  -d '{"user_query": "等待期内确诊是否赔付？", "context_docs": [], "max_new_tokens": 256, "temperature": 0.3}'
```

### 4. 自动化评测 (M-EVAL)

```bash
copaw-dpo evaluate \
    --model merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \
    --eval-data data/eval/medical_qa_1000.jsonl \
    --output reports/eval_report_dpo_v1.2
```

评测指标与输出：

| 指标 | 说明 |
|------|------|
| Accuracy | 分类问答严格匹配 + 开放式 LLM-as-Judge |
| BLEU-4 | sacrebleu 计算 n-gram=4 |
| ROUGE-L | 最长公共子序列重合度 |
| 推理时间 | p50/p95/p99 + throughput |

输出 JSON + Markdown 双格式评测报告。

---

## 项目目录

```
AGroup-DPO/
├── src/
│   ├── m_data/               # DPO 数据集生成流水线
│   │   ├── sources/          #   数据源（条款/FAQ/工单）
│   │   ├── prompts/          #   LLM-as-Judge 模板
│   │   ├── pipeline.py       #   流水线编排
│   │   ├── pair_builder.py   #   chosen/rejected 配对
│   │   ├── sft_builder.py    #   SFT 样本构造
│   │   ├── validator.py      #   规则校验
│   │   ├── normalizer.py     #   文本规范化
│   │   ├── pii_scrubber.py   #   PII 脱敏
│   │   └── pii_patterns.py   #   共享 PII 正则
│   ├── m_trainer/            # 分布式 Trainer
│   │   └── backends/         #   DeepSpeed/FSDP/Megatron/accelerate
│   ├── m_infer/              # 推理加速后端
│   │   ├── vllm_backend.py   #   vLLM 实现
│   │   ├── xinference_backend.py  # xinference 实现
│   │   ├── server.py         #   FastAPI 推理服务
│   │   └── rag_handler.py    #   RAG 对接路由
│   ├── m_eval/               # 评测模块
│   │   ├── metrics.py        #   Accuracy/BLEU/ROUGE
│   │   ├── latency.py        #   推理时间统计
│   │   └── reporter.py       #   报告产出
│   └── m_merge/              # 模型合并导出
├── configs/                  # YAML 训练/推理/评测配置
│   ├── data/                 #   数据流水线配置
│   ├── backends/             #   分布式后端配置
│   └── deepspeed/            #   DeepSpeed ZeRO 配置
├── data/
│   ├── insurance/            # 保险训练数据
│   │   ├── dpo_train_v1.2.jsonl
│   │   └── insurance_sft_v1.jsonl
│   ├── eval/                 # 评测数据集
│   └── smoke/                # 烟雾测试数据
├── deploy/                   # Docker 镜像 + 部署脚本
│   ├── Dockerfile.base       #   基础训练镜像
│   ├── Dockerfile.{deepspeed,fsdp,megatron,accelerate}  # 后端 layer
│   ├── Dockerfile.infer-{vllm,xinfer}  # 推理 layer
│   ├── Dockerfile.eval       #   评测 layer
│   └── deploy_to_server2.sh  #   一键部署脚本
├── scripts/                  # 烟雾测试 + 工具脚本
├── tests/                    # 单元测试（43 用例）
├── docs/                     # 项目方案 + 阶段设计文档
├── pyproject.toml            # 项目配置
└── requirements.txt          # 完整依赖清单
```

---

## 开发阶段

项目按 5 个阶段递进式开发，已完成全链路交付：

| 阶段 | 名称 | 对应模块 | 状态 |
|------|------|----------|------|
| **M01** | 基础设施与环境准备 | deploy/, Docker 镜像 | ✅ |
| **M02** | DPO 数据集生成流水线 | m_data/ | ✅ |
| **M03** | LoRA 微调与 DPO 对齐训练 | configs/, m_trainer/ | ✅ |
| **M04** | 自研分布式 Trainer | m_trainer/ backends/ | ✅ |
| **M05** | 推理加速与评测 | m_infer/, m_eval/, m_merge/ | ✅ |

---

## 部署

### 本地推送至远端

```bash
# 一键部署到 server2（rsync + 远端烟雾测试）
bash deploy/deploy_to_server2.sh
```

流程：`rsync 推送 → 远端 screen 后台执行 M01 烟雾测试`

### Docker 镜像

分层设计，按需拉取：

```
copaw-dpo-base:latest          # 基础训练镜像 (~12 GB)
├── copaw-dpo-deepspeed:latest # + DeepSpeed ZeRO3
├── copaw-dpo-fsdp:latest      # + PyTorch FSDP
├── copaw-dpo-accelerate:latest # + accelerate
├── copaw-dpo-megatron:latest  # + Megatron-LM
├── copaw-dpo-infer-vllm:latest  # + vLLM
├── copaw-dpo-infer-xinfer:latest # + xinference
└── copaw-dpo-eval:latest      # + 评测工具
```

```bash
# 构建全部镜像
docker build -f deploy/Dockerfile.base -t copaw-dpo-base:latest .
for backend in deepspeed fsdp megatron accelerate; do
    docker build -f deploy/Dockerfile.$backend -t copaw-dpo-$backend:latest .
done
```

---

## 测试

```bash
# 运行全部单元测试
PYTHONPATH=src python -m pytest tests/ -v

# 分模块测试
PYTHONPATH=src python -m pytest tests/m_data/ -v     # DPO 数据流水线
PYTHONPATH=src python -m pytest tests/m_trainer/ -v  # 分布式 Trainer
PYTHONPATH=src python -m pytest tests/m_infer/ -v    # 推理后端
PYTHONPATH=src python -m pytest tests/m_eval/ -v     # 评测模块
```

当前测试覆盖：Validator、PIIScrubber、PairBuilder、Exporter、Trainer Backend、Infer Backend、RAG Handler、Metrics、Latency、Reporter 等核心模块。

---

## 核心依赖

| 类别 | 关键包 | 版本 |
|------|--------|------|
| 训练 | torch, transformers, peft, trl, llamafactory | 2.7.1, 4.43.4, 0.11.1, 0.9.6, 0.9.1 |
| 分布式 | deepspeed, accelerate | 0.14.4, ≥0.34.0 |
| 推理 | vllm, xinference, fastapi, uvicorn | ≥0.8.5, ≥0.15.4, 0.111.0, 0.30.1 |
| 评测 | sacrebleu, rouge-score, nltk, jiwer | 2.4.2, 0.1.2, 3.8.1, 3.0.3 |
| 工具 | pyyaml, omegaconf, loguru, pydantic | 6.0.1, 2.3.0, 0.7.2, 2.8.2 |

完整依赖见 `requirements.txt` 和 `pyproject.toml`。

---

## 文档索引

| 文档 | 路径 |
|------|------|
| 项目总体方案 | [`docs/项目方案.md`](docs/项目方案.md) |
| 项目需求 | [`docs/项目需求.md`](docs/项目需求.md) |
| M01 基础设施 | [`docs/项目分阶段方案/M01_基础设施与环境准备.md`](docs/项目分阶段方案/M01_基础设施与环境准备.md) |
| M02 DPO 数据流水线 | [`docs/项目分阶段方案/M02_DPO数据集生成流水线.md`](docs/项目分阶段方案/M02_DPO数据集生成流水线.md) |
| M03 LoRA+DPO 训练 | [`docs/项目分阶段方案/M03_LoRA微调与DPO对齐训练.md`](docs/项目分阶段方案/M03_LoRA微调与DPO对齐训练.md) |
| M04 分布式 Trainer | [`docs/项目分阶段方案/M04_自研分布式Trainer与模型导出.md`](docs/项目分阶段方案/M04_自研分布式Trainer与模型导出.md) |
| M05 推理加速与评测 | [`docs/项目分阶段方案/M05_推理加速与评测.md`](docs/项目分阶段方案/M05_推理加速与评测.md) |
| 开发进度 | [`docs/开发进度/`](docs/开发进度/) |
| m_data 模块 | [`src/m_data/README.md`](src/m_data/README.md) |
| m_infer 模块 | [`src/m_infer/README.md`](src/m_infer/README.md) |
| m_eval 模块 | [`src/m_eval/README.md`](src/m_eval/README.md) |
| 部署说明 | [`deploy/README.md`](deploy/README.md) |

---

## 团队

**AGroup DPO Team**

---

## 许可证

Proprietary — 内部项目，保留所有权利。
