<p align="center">
  <img src="docs/screenshots/title.svg" alt="AGroup DPO" width="700">
</p>

<p align="center">
  <b>保险业务场景下的 DPO 偏好对齐训练项目</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.7.1-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/LLaMA--Factory-0.9.1-orange" alt="LLaMA-Factory">
  <img src="https://img.shields.io/badge/DeepSpeed-0.14.4-green" alt="DeepSpeed">
  <img src="https://img.shields.io/badge/vLLM-0.8.5-blueviolet" alt="vLLM">
</p>

---

## 这个项目做什么

用 Qwen2.5-1.5B-Instruct 做保险领域的 DPO 偏好对齐。整体流程是：先从保险条款、FAQ、工单里自动生成训练数据，然后 LoRA 微调 + DPO 对齐训练，最后部署推理服务并评测效果。

主要做了几件事：

- **数据流水线**：把保险条款/FAQ/工单自动转成 DPO 需要的 chosen/rejected 配对数据
- **训练**：LoRA SFT → LoRA DPO，支持 DeepSpeed ZeRO3 / FSDP 多后端切换
- **推理**：vLLM 和 xinference 双后端，改配置就能切换
- **评测**：Accuracy、BLEU-4、ROUGE-L + 推理耗时统计

---

## 跑起来

### 环境

| 依赖 | 版本 | 
|------|------|
| Python | ≥ 3.10 |
| PyTorch | 2.7.1+cu128 |
| CUDA | ≥ 12.4 |
| GPU | 2×A100-80G 或 2×RTX 4090/5090 |

### 安装

```bash
cd /root/autodl-tmp
git clone <repo-url> agroup-dpo
cd agroup-dpo

conda create -n llm python=3.12
conda activate llm
pip install -e .
```

### 下载模型

```bash
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct
```

### 生成训练数据

```bash
PYTHONPATH=src python -m m_data.cli \
    --config configs/data/insurance_dpo_gen.yaml

# 或者先干跑看看
PYTHONPATH=src python -m m_data.cli \
    --config configs/data/insurance_dpo_gen.yaml \
    --dry-run --verbose
```

**实际运行结果 (server6, 2×RTX 4090)：**

![DPO 数据生成](docs/screenshots/dpo_data_gen.png)

```
============================================================
Pipeline Summary
============================================================
  DPO samples:  7247
  SFT samples:  7048
  Elapsed:      0.7s
  Validator:    7247/7247 passed (100.0%)
============================================================
```

### 训练

```bash
# LoRA SFT
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train \
    configs/train_lora_qwen2_5_1_5b_insurance.yaml

# DPO 对齐（基于 SFT 合并后的模型）
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train \
    configs/train_dpo_qwen2_5_1_5b_insurance.yaml
```

**DPO 训练实际结果 (server6, smoke test, 10 steps)：**

![DPO 训练](docs/screenshots/dpo_train.png)

### 推理

```python
from m_infer import build_infer_backend, InferRequest

backend = build_infer_backend("vllm", "merged_models/qwen2_5_1_5b_insurance_dpo_v1.2")
resp = backend.infer(InferRequest(prompt="保险等待期是什么？", max_new_tokens=128))
print(resp.text)
```

### 评测

```bash
copaw-dpo evaluate \
    --model merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \
    --eval-data data/eval/insurance_qa_500.jsonl \
    --output reports/eval_report
```

---

## 数据怎么生成的

从 3 类数据源出发，走一条 6 步流水线：

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

三种配对方式：

| 策略 | 怎么做的 |
|------|----------|
| 规则硬负例 | 基于条款强制构造合规/违规答案对 |
| LLM-as-Judge | 用 Qwen2.5-7B 对比 RAG 答案和专家标注 |
| 检索差异 | 完整索引 vs 截断索引的 RAG 答案对比 |

数据源：保险条款（PDF/HTML）、业务 FAQ、历史工单

校验规则：prompt 长度、chosen/rejected 长度、PII 检测、条款引用检查、相似度去重

---

## 训练怎么跑的

4 种分布式后端，同一份配置切着用：

| 后端 | 场景 | 
|------|------|
| DeepSpeed ZeRO3 | 默认，显存省得多 |
| PyTorch FSDP | PyTorch 原生备选 |
| accelerate | 单机多卡调试 |
| Megatron-LM | 7B+ 模型预留 |

---

## 模块说明

| 模块 | 在哪 | 干什么 |
|------|------|------|
| M-DATA | `src/m_data/` | 数据生成（采集→脱敏→配对→校验→导出） |
| M-TRAINER | `src/m_trainer/` | 分布式训练后端 |
| M-INFER | `src/m_infer/` | vLLM/xinference 推理 |
| M-EVAL | `src/m_eval/` | 评测（Accuracy/BLEU/ROUGE） |
| M-MERGE | `src/m_merge/` | LoRA 权重合并导出 |

---

## 目录结构

```
AGroup-DPO/
├── src/
│   ├── m_data/             # 数据流水线
│   │   ├── sources/        #   数据源（条款/FAQ/工单）
│   │   ├── prompts/        #   LLM-as-Judge 模板
│   │   ├── pipeline.py     #   流水线编排
│   │   ├── pair_builder.py #   chosen/rejected 配对
│   │   ├── validator.py    #   规则校验
│   │   └── pii_scrubber.py #   PII 脱敏
│   ├── m_trainer/          # 分布式 Trainer
│   │   └── backends/       #   DeepSpeed/FSDP/Megatron/accelerate
│   ├── m_infer/            # 推理后端
│   │   ├── vllm_backend.py
│   │   ├── xinference_backend.py
│   │   ├── server.py       #   FastAPI 服务
│   │   └── rag_handler.py  #   RAG 对接
│   ├── m_eval/             # 评测
│   └── m_merge/            # 模型合并
├── configs/                # YAML 配置
│   ├── data/               #   数据流水线配置
│   ├── backends/           #   分布式后端配置
│   └── deepspeed/          #   DeepSpeed ZeRO 配置
├── data/
│   ├── insurance/          # 保险训练数据
│   ├── eval/               # 评测数据
│   └── smoke/              # 烟雾测试
├── deploy/                 # Docker + 部署脚本
├── scripts/                # 工具脚本
├── tests/                  # 单元测试
└── docs/                   # 方案文档
```

---

## 开发进度

分 5 个阶段做完的：

| 阶段 | 内容 | 
|------|------|
| M01 | 基础设施与环境准备 |
| M02 | DPO 数据集生成流水线 |
| M03 | LoRA 微调 + DPO 对齐训练 |
| M04 | 自研分布式 Trainer |
| M05 | 推理加速与评测 |

详细文档见 [`docs/项目分阶段方案/`](docs/项目分阶段方案/)

---

## 部署

### 推到远端

```bash
bash deploy/deploy_to_server2.sh
```

### Docker

分层镜像，按需构建：

```bash
docker build -f deploy/Dockerfile.base -t copaw-dpo-base:latest .
docker build -f deploy/Dockerfile.deepspeed -t copaw-dpo-deepspeed:latest .
docker build -f deploy/Dockerfile.infer-vllm -t copaw-dpo-infer-vllm:latest .
```

---

## 测试

```bash
PYTHONPATH=src python -m pytest tests/ -v
PYTHONPATH=src python -m pytest tests/m_data/ -v    # 数据流水线
PYTHONPATH=src python -m pytest tests/m_trainer/ -v # 训练后端
```

---

## 文档

| 文档 | 路径 |
|------|------|
| 项目方案 | [`docs/项目方案.md`](docs/项目方案.md) |
| M01 基础设施 | [`docs/项目分阶段方案/M01_基础设施与环境准备.md`](docs/项目分阶段方案/M01_基础设施与环境准备.md) |
| M02 数据流水线 | [`docs/项目分阶段方案/M02_DPO数据集生成流水线.md`](docs/项目分阶段方案/M02_DPO数据集生成流水线.md) |
| M03 训练 | [`docs/项目分阶段方案/M03_LoRA微调与DPO对齐训练.md`](docs/项目分阶段方案/M03_LoRA微调与DPO对齐训练.md) |
| M04 Trainer | [`docs/项目分阶段方案/M04_自研分布式Trainer与模型导出.md`](docs/项目分阶段方案/M04_自研分布式Trainer与模型导出.md) |
| M05 推理评测 | [`docs/项目分阶段方案/M05_推理加速与评测.md`](docs/项目分阶段方案/M05_推理加速与评测.md) |

---

AGroup DPO Team · Proprietary
