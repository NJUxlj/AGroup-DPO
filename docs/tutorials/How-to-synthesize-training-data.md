# 如何合成训练数据

> 面向开源社区的 DPO / SFT 数据生成教程  
> 对应模块：`src/m_data/` · 阶段 M02

---

## 目录

1. [概述](#1-概述)
2. [前置条件](#2-前置条件)
3. [流水线架构](#3-流水线架构)
4. [数据源准备](#4-数据源准备)
5. [配置文件详解](#5-配置文件详解)
6. [三种配对策略](#6-三种配对策略)
7. [运行命令](#7-运行命令)
8. [输出格式](#8-输出格式)
9. [校验与条款引用修复](#9-校验与条款引用修复)
10. [自定义与扩展](#10-自定义与扩展)
11. [质量验收](#11-质量验收)
12. [常见问题](#12-常见问题)

---

## 1. 概述

AGroup-DPO 的数据流水线（**M-DATA**）从保险业务原始数据中，自动构造 DPO 训练所需的 `(prompt, chosen, rejected)` 三元组，以及 SFT 训练所需的 `(instruction, input, output)` 样本。

**核心能力：**

- 三类数据源：保险条款、FAQ、历史工单
- 三种配对策略：规则硬负例、LLM-as-Judge、检索差异
- 六类质量校验 + PII 脱敏
- PolicyStore 条款检索修复（Milvus Lite + BGE 嵌入）

**无需人工标注**即可从结构化业务数据批量产出万级训练样本。

---

## 2. 前置条件

### 2.1 环境

```bash
conda activate llm
pip install -e .
pip install "pymilvus[milvus_lite]>=2.4.0" "sentence-transformers>=2.7.0"
export HF_ENDPOINT=https://hf-mirror.com   # 国内加速 BGE 模型下载
export PYTHONPATH=src
```

### 2.2 原始数据

仓库已内置示例数据，位于：

```
data/insurance/raw/
├── policies/          # 5 份保险条款 JSON
│   ├── POL-CRIT-001.json
│   ├── POL-MEDI-001.json
│   └── ...
├── faq/
│   └── faq_v1.json
└── tickets/
    └── tickets_v1.json
```

若使用自有数据，请保持相同目录结构与 JSON 格式（详见 [src/m_data/README.md](../../src/m_data/README.md)）。

---

## 3. 流水线架构

```
[Collector]  采集原始记录（条款 / FAQ / 工单）
     ↓
[Normalizer] 文本规范化（全半角、空白、标点）
     ↓
[PIIScrubber] PII 脱敏（手机号、身份证、姓名等）
     ↓
[Filter]     过滤无效记录
     ├──────────────────┐
     ↓                  ↓
[PairBuilder]      [SFTBuilder]
 DPO 配对            SFT 样本
     ↓                  ↓
[Validator]        [SFT Validator]
 规则校验             格式校验
     ↓                  ↓
[Exporter]         [Exporter]
 dpo_train_v1.2.jsonl   insurance_sft_v1.jsonl
```

---

## 4. 数据源准备

### 4.1 保险条款（policy）

结构化 JSON，包含 `policy_id`、条款章节 `articles` 等字段。用于：

- 规则硬负例的合规/违规答案构造
- PolicyStore 向量检索修复

### 4.2 FAQ（faq）

问答对 JSON，包含 `question`、`answer`、`category` 等。用于：

- SFT 正例样本
- 规则配对的问题来源

### 4.3 工单（ticket）

客服工单 JSON，包含用户问题、专家回复、RAG 回复等。用于：

- LLM-as-Judge 策略（需 expert + RAG 双答案）
- 检索差异策略（需 RAG 服务）

---

## 5. 配置文件详解

主配置文件：[`configs/data/insurance_dpo_gen.yaml`](../../configs/data/insurance_dpo_gen.yaml)

```yaml
project: insurance_dpo
seed: 42
version: dpo_v1.2

# ── 数据源 ──
sources:
  policy:
    enabled: true
    type: policy
    path: data/insurance/raw/policies/
    parser: json
  faq:
    enabled: true
    type: faq
    path: data/insurance/raw/faq/
  ticket:
    enabled: true
    type: ticket
    path: data/insurance/raw/tickets/
    filter_category: compliance_qa

# ── 配对策略 ──
strategies:
  rule_based:
    enabled: true          # 默认开启，无外部依赖
  llm_judge:
    enabled: false         # 需 Judge 模型 endpoint
    judge_model: qwen2.5-7b-instruct
    judge_endpoint: http://127.0.0.1:8001/v1/chat/completions
    api_key: ""
  retrieval_diff:
    enabled: false         # 需司内 RAG 服务
    rag_endpoint: http://rag.internal/v1/rag/query

# ── 质量阈值 ──
quality:
  min_prompt_len: 5
  max_prompt_len: 1024
  min_response_len: 10
  max_response_len: 2048
  max_chosen_rejected_similarity: 0.95

# ── PolicyStore 条款检索修复 ──
policy_store:
  enabled: true
  data_dir: data/insurance/raw/policies/
  embedding_model: BAAI/bge-small-zh-v1.5
  milvus_db_path: ./milvus_data/policy_store.db

# ── 输出路径 ──
output:
  path: data/insurance/dpo_train_v1.2.jsonl
  sft_path: data/insurance/insurance_sft_v1.jsonl
```

**关键开关说明：**

| 配置项 | 开源默认 | 说明 |
|--------|----------|------|
| `strategies.rule_based.enabled` | `true` | 开箱即用，无需外部服务 |
| `strategies.llm_judge.enabled` | `false` | 需本地 vLLM 或 OpenAI 兼容 API |
| `strategies.retrieval_diff.enabled` | `false` | 需 RAG 检索服务 |
| `policy_store.enabled` | `true` | 缺依赖时自动回退模板兜底 |

---

## 6. 三种配对策略

| 维度 | 策略 A: `rule_based` | 策略 B: `llm_judge` | 策略 C: `retrieval_diff` |
|------|----------------------|---------------------|--------------------------|
| **chosen 来源** | 合规答案 / FAQ 专家答案 | LLM 判定胜出者 | 完整索引 RAG 答案 |
| **rejected 来源** | 人工模板错误答案 | LLM 判定落败者 | 截断索引 RAG 答案 |
| **信号强度** | 强（确定性） | 中（主观判断） | 中（检索质量差异） |
| **需要 LLM** | 否 | 是 | 否 |
| **需要外部服务** | 否 | Judge endpoint | RAG endpoint |
| **扩展性** | 低（需编写模板） | 高 | 高 |

### 6.1 策略 A：规则硬负例（推荐入门）

内置 ~155 条保险 QA 模板 + FAQ/工单扩展，构造「合规 chosen vs 违规 rejected」配对。例如：

- **chosen**：「等待期内确诊一般不予赔付，合同另有约定的除外。具体参见条款第5.2条。」
- **rejected**：「等待期内确诊也会赔付，公司都会赔的。」

### 6.2 策略 B：LLM-as-Judge

对同一 prompt 的 expert 答案与 RAG 答案，调用 Judge 模型（如 Qwen2.5-7B-Instruct）做 pairwise 比较，胜出者为 chosen。

启用方式：

```yaml
strategies:
  llm_judge:
    enabled: true
    judge_endpoint: http://127.0.0.1:8001/v1/chat/completions
    api_key: "sk-xxx"   # 第三方 API 时填写；本地模型留空
```

Judge 提示词模板位于 `src/m_data/prompts/judge_pairwise.txt`。

### 6.3 策略 C：检索差异

同一问题分别用「完整知识库索引」和「截断知识库索引」调用 RAG 服务，答案质量差异天然构成偏好对。

---

## 7. 运行命令

### 7.1 干跑（不写文件，验证配置）

```bash
copaw-dpo data --config configs/data/insurance_dpo_gen.yaml --dry-run --verbose
```

或使用模块入口：

```bash
PYTHONPATH=src python -m m_data.cli \
    --config configs/data/insurance_dpo_gen.yaml \
    --dry-run --verbose
```

### 7.2 完整运行

```bash
copaw-dpo data --config configs/data/insurance_dpo_gen.yaml
```

预期输出摘要：

```
============================================================
Pipeline Summary
============================================================
  DPO samples:  7247
  SFT samples:  7048
  Elapsed:      0.7s
  Validator:    7247/7247 passed (100.0%)
============================================================
  DPO written:  7247
  SFT written:  7048
```

> 全量 PolicyStore 索引时首次运行较慢（嵌入模型下载 + Milvus 建索引约 60–70s），后续增量运行会快很多。

### 7.3 增量运行

仅处理指定日期之后的新增数据：

```bash
copaw-dpo data --config configs/data/insurance_dpo_gen.yaml --since 2026-06-01
```

### 7.4 一键集成测试

```bash
bash deploy/run_dpo_data_smoke.sh
# 或
PYTHONPATH=src python tests/m_data/test_dpo_full_pipeline.py
```

---

## 8. 输出格式

### 8.1 DPO 样本（`dpo_train_v1.2.jsonl`）

每行一条 JSON：

```json
{
  "prompt": "重疾险等待期内确诊是否赔付？",
  "chosen": "等待期内确诊一般不予赔付，合同另有约定的除外。具体参见条款第5.2条。",
  "rejected": "等待期内确诊也会赔付，公司都会赔的。",
  "source": "rule_based",
  "policy_id": "POL-CRIT-001",
  "judge_model": null,
  "judge_score_chosen": null,
  "judge_score_rejected": null,
  "pii_scrubbed": true,
  "version": "dpo_v1.2"
}
```

### 8.2 SFT 样本（`insurance_sft_v1.jsonl`）

```json
{
  "instruction": "请回答用户的保险问题。",
  "input": "重疾险的等待期一般是多久？",
  "output": "一般重疾险的等待期为90天...",
  "system": "你是AI财保助理，需严格依据条款作答。",
  "source": "sft_s-a",
  "version": "sft_v1"
}
```

### 8.3 LLaMA-Factory 数据集注册

`data/insurance/dataset_info.json` 已注册两个数据集，训练时直接引用：

```json
{
  "insurance_sft_v1": {
    "file_name": "insurance_sft_v1.jsonl",
    "columns": { "prompt": "instruction", "query": "input", "response": "output", "system": "system" }
  },
  "insurance_dpo_v1.2": {
    "file_name": "dpo_train_v1.2.jsonl",
    "ranking": true,
    "columns": { "prompt": "prompt", "chosen": "chosen", "rejected": "rejected" }
  }
}
```

---

## 9. 校验与条款引用修复

### 9.1 六类校验规则

| 规则 | 阈值 | 不通过处理 |
|------|------|------------|
| prompt 长度 | [5, 1024] | 丢弃 |
| chosen/rejected 长度 | [10, 2048] | 丢弃 |
| PII 检测 | 0 命中 | 回退重脱敏 |
| 条款引用 | 业务关键词场景 chosen 须含条款/保单/合同/法 | 尝试修复或丢弃 |
| chosen ≠ rejected | 相似度 < 0.95 | 丢弃 |
| pii_scrubbed 标记 | 必须为 true | 丢弃 |

### 9.2 PolicyStore 条款引用修复

当 chosen 缺少具体条款引用时，Validator 通过 **PolicyStore** 从原始条款中检索真实条文注入：

```
Milvus 混合检索（BGE 向量 + 关键词重排）
    ↓ 命中
追加: "依据POL-CRIT-001：第2.1条（等待期）：..."
    ↓ 未命中
ID 兜底 → 通用兜底模板
```

技术选型：

| 组件 | 选型 |
|------|------|
| 向量库 | Milvus Lite（嵌入式，无需独立服务） |
| Embedding | BAAI/bge-small-zh-v1.5（512 维） |
| 混合检索 | 65% 向量 + 35% 关键词 |

---

## 10. 自定义与扩展

### 10.1 添加新数据源

1. 在 `src/m_data/sources/` 下继承 `DataSource` 基类
2. 实现 `collect()` 方法，返回 `RawRecord` 列表
3. 在 yaml 的 `sources` 中注册

### 10.2 添加规则模板

编辑 `src/m_data/pair_builder.py` 中的 `HARD_NEGATIVE_TEMPLATES`，或提供外部规则文件：

```yaml
strategies:
  rule_based:
    enabled: true
    rules_file: configs/data/rules/my_hard_negatives.yaml
```

### 10.3 调整质量阈值

修改 `quality` 段即可，例如放宽 prompt 上限：

```yaml
quality:
  max_prompt_len: 2048
```

---

## 11. 质量验收

### 11.1 单元测试

```bash
PYTHONPATH=src python -m pytest tests/m_data/ -v
```

当前覆盖 Validator、PIIScrubber、PairBuilder、Exporter 等 43+ 用例。

### 11.2 产出检查

```bash
# 样本数量
wc -l data/insurance/dpo_train_v1.2.jsonl
wc -l data/insurance/insurance_sft_v1.jsonl

# 随机抽查
shuf -n 3 data/insurance/dpo_train_v1.2.jsonl | python -m json.tool

# 来源分布
python -c "
import json
from collections import Counter
c = Counter()
with open('data/insurance/dpo_train_v1.2.jsonl') as f:
    for line in f:
        c[json.loads(line)['source']] += 1
print(c)
"
```

### 11.3 验收标准

| 指标 | 目标 |
|------|------|
| Validator 通过率 | ≥ 95% |
| chosen/rejected 语义差异 | 人工抽检 ≥ 90% 有效 |
| PII 残留 | 0 命中 |

---

## 12. 常见问题

| 问题 | 解决方案 |
|------|----------|
| PolicyStore 初始化失败 | 安装 `pymilvus[milvus_lite]` 和 `sentence-transformers`；或设 `policy_store.enabled: false` |
| BGE 模型下载超时 | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
| DPO 样本数为 0 | 检查 `sources.*.enabled` 和数据路径是否存在 |
| llm_judge 策略无产出 | 确认 ticket 数据含 expert + RAG 双答案字段 |
| chosen/rejected 过于相似 | 降低 `max_chosen_rejected_similarity` 或检查配对逻辑 |

---

**下一步**：[How-to-run-SFT?](./How-to-run-SFT%3F) — 使用生成的 SFT 数据进行 LoRA 微调。
