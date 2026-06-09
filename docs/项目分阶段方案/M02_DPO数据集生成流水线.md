# M02 DPO 数据集生成流水线

> 阶段编号：M02
> 阶段名称：DPO 数据集生成流水线
> 预估工期：4 人天
> 关联文档：[项目方案.md § 5.1 M-DATA](../项目方案.md) | [项目方案.md § 6 数据设计](../项目方案.md)
> 上游阶段：M01
> 下游阶段：M03、M04
> 对应功能需求：FR-03

---

## 1. 阶段定位

M02 是**训练阶段的输入供给方**，产出两类训练数据集：

1. **SFT 数据集**（`insurance_sft_v1.jsonl`）：用于 LoRA SFT 微调阶段，让基座具备保险业务指令遵循能力。
2. **DPO 数据集**（`dpo_train_v1.2.jsonl`）：在 SFT 模型基础上做对齐训练，要求 chosen/rejected 对。

本阶段不涉及训练，但产出的数据集质量直接决定 DPO 训练收敛性与最终对齐效果。核心难点：

1. **多源数据融合**：条款 PDF、FAQ 库、历史工单三种异构数据源的统一接入。
2. **负例构造多样性**：单一策略（A 或 B 或 C）覆盖率不足，需三种互补策略合并去重。
3. **业务合规口径**：涉及"赔付 / 等待期 / 免赔 / 告知"等强业务关键词时，chosen 必须显式引用条款编号。
4. **PII 脱敏**：客户姓名、手机号、身份证等敏感信息 100% 脱敏后方可进入训练集。

---

## 2. 阶段目标

### 2.1 业务目标

| 维度 | 目标 | 衡量方式 |
|------|------|----------|
| SFT 数据规模 | 产出 ≥ 3000 条 instruction-input-output 样本 | jsonl 行数 |
| DPO 数据规模 | 产出 ≥ 5000 条 chosen/rejected 对 | jsonl 行数 |
| 来源覆盖 | 条款 / FAQ / 工单 / llm_judge / retrieval_diff 五类 source 均有样本 | source 分布表 |
| 合规覆盖 | "赔付 / 等待期 / 免赔 / 告知" 等关键词的 chosen 100% 含条款引用 | 规则校验脚本 |
| PII 安全 | PII 命中率 = 0 | 正则扫描脚本 |

### 2.2 技术目标

| 维度 | 目标 | 衡量方式 |
|------|------|----------|
| 数据质量 | Validator 通过率 ≥ 85% | m_data/validator 报告 |
| 流水线可重跑 | 全量重跑耗时 ≤ 2 h（含 LLM-as-Judge 调用） | CI 计时 |
| 与司内 RAG 端连通 | 调用 `/v1/rag/query` 获取 baseline 答案 ≥ 1000 条 | 接口日志 |

---

## 3. 核心任务

### 3.1 数据源采集器实现

实现三类数据源的统一接入，遵循 `DataSource` 抽象接口：

```python
# m_data/sources/base.py
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterator

class RawRecord:
    """原始记录，字段未定型"""
    def __init__(self, source: str, content: dict, raw_meta: dict):
        self.source = source
        self.content = content
        self.raw_meta = raw_meta

class DataSource(ABC):
    @abstractmethod
    def fetch(self, since: datetime, limit: int) -> Iterator[RawRecord]: ...

class InsurancePolicySource(DataSource):
    """从条款库解析：PDF/HTML → 结构化条款记录"""
    ...

class FAQSource(DataSource):
    """从司内 FAQ 库导出：分类 + 问题 + 答案"""
    ...

class TicketSource(DataSource):
    """从工单系统导出'合规问答'子集"""
    ...
```

| 数据源 | 内容 | 获取方式 | 脱敏要求 |
|--------|------|----------|----------|
| 保险条款 | 各险种条款 PDF/HTML | 业务方提供，解析入库 | 内部编号脱敏 |
| 业务 FAQ | 客服 FAQ 库 | 司内知识库导出 | 客户姓名/手机号脱敏 |
| 历史工单 | 客服与用户问答对 | 司内工单系统（仅"合规问答"子集） | 全量 PII 脱敏 |

### 3.2 Chosen/Rejected 构造策略实现

针对保险业务场景，采用三种互补策略，最终合并去重：

**策略 A：基于业务规则的硬负例**

针对同一问题，依据保险条款强制要求生成"合规答案（chosen）"与"违规答案（rejected）"。

示例：

| Prompt | Chosen | Rejected |
|--------|--------|----------|
| "重疾险等待期内确诊是否赔付？" | "等待期内确诊一般不予赔付，合同另有约定的除外。具体参见条款 X 第 Y 条。" | "等待期内确诊也会赔付，公司都会赔的。" |
| "百万医疗险的免赔额是怎么计算的？" | "百万医疗险通常设置 1 万元年度免赔额，通过社保/公费医疗报销的部分可抵扣免赔额。具体以保单条款为准。" | "百万医疗险没有免赔额，所有费用都能报销。" |
| "投保前未告知高血压，理赔会被拒吗？" | "若投保前未如实告知健康状况，保险公司有权根据《保险法》第十六条解除合同或拒赔；但若超过两年不可抗辩期且非故意，则理赔权利受保护。" | "这种情况保险公司一定会拒赔，没有任何商量余地。" |

**策略 B：基于 LLM-as-Judge 的软负例**

用 Qwen2.5-7B-Instruct（judge 模型）对比"司内普通 RAG 端答案"与"专家人工标注答案"，输出 pairwise preference。

Judge Prompt 模板：

```
你是一名保险业务专家。请比较以下两个答案哪一个更合规、更准确、更符合用户问题。

【问题】{prompt}

【答案 A】{candidate_a}

【答案 B】{candidate_b}

请输出严格 JSON：{"winner": "A" | "B" | "TIE", "score_a": 0~1, "score_b": 0~1, "reason": "..."}
```

**策略 C：基于检索召回差异**

对同一 query，分别用"完整索引"和"截断索引"做 RAG 召回：

- chosen = 完整索引答案（前提是 chosen 通过了规则校验）
- rejected = 截断索引答案

### 3.3 PII 脱敏与规范化

**PII 脱敏规则**（正则 + 词典双重）：

```python
PII_PATTERNS = [
    re.compile(r"\d{17}[\dXx]"),                       # 身份证
    re.compile(r"1[3-9]\d{9}"),                         # 手机号
    re.compile(r"\d{16,19}"),                           # 银行卡
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),            # 邮箱
]
```

**规范化**：去除多余空白、统一标点（半角/全角）、HTML tag 清除、长度截断。

### 3.4 规则校验（Validator）

| 质量维度 | 检查规则 | 不通过处理 |
|----------|----------|------------|
| 长度 | prompt 长度 ∈ [5, 1024]；chosen/rejected ∈ [10, 2048] | 丢弃或人工复核 |
| 敏感词 | 包含身份证/手机号/银行卡/邮箱原始格式则不通过 | 强制 PII 脱敏 |
| 重复 | 同一 prompt 不允许多次出现（按 hash 去重） | 保留最新版本 |
| 必引条款 | 涉及"赔付/等待期/免赔/告知"等关键词的 chosen 必须包含"条款/保单/合同"之一 | 回流到 PairBuilder 重做 |
| chosen≠rejected | 字符级相似度 < 0.95 | 丢弃 |
| PII | 100% 通过正则+词典双重脱敏 | 整条丢弃 |

**校验伪代码**：

```python
# m_data/validator.py
REQUIRED_TERMS_FOR_CLAIMS = ["条款", "保单", "合同", "法"]

def validate(sample: dict) -> tuple[bool, str]:
    if not (5 <= len(sample["prompt"]) <= 1024):
        return False, "prompt length out of range"
    for field in ("prompt", "chosen", "rejected"):
        for pat in PII_PATTERNS:
            if pat.search(sample[field]):
                return False, f"PII detected in {field}"
    if any(kw in sample["chosen"] for kw in ["赔付", "等待期", "免赔", "告知"]):
        if not any(t in sample["chosen"] for t in REQUIRED_TERMS_FOR_CLAIMS):
            return False, "missing required policy reference"
    if similarity(sample["chosen"], sample["rejected"]) > 0.95:
        return False, "chosen and rejected too similar"
    return True, "ok"
```

### 3.5 流水线编排

```
[Collector]  ── 多源并发采集 ──▶ [Normalizer]  ── 文本规范化 ──▶ [PIIScrubber]
   │                                                            │
   ▼                                                            ▼
[Filter]  ── 长度/敏感词过滤 ──▶ [PairBuilder] ── 配对 chosen/rejected
   │                                          │
   ▼                                          ▼
[SFTBuilder] ── 配对 instruction/output   [Validator] ── 规则校验（必引条款）
   │                                          │
   ▼                                          ▼
[sft_train_v1.jsonl]                    [Exporter]  ── jsonl 写出
                                              │
                                              ▼
                                  [dpo_train_v1.2.jsonl]
```

### 3.6 与司内普通 RAG 端对接

- **入参对接**：DPO 数据生成器调用司内普通 RAG 端的 `/v1/rag/query` 接口，获取"baseline 答案"，作为策略 B 的 rejected 候选。
- **出参对接**：生成完成后，将 `dpo_train_v1.2.jsonl` 上传到司内共享存储 `/shared/datasets/insurance_dpo/`，供后续训练流程读取。

### 3.7 SFT 数据集构造

SFT 数据集用于 M03 的 LoRA SFT 微调，目的是让 Qwen2.5-1.5B-Instruct 学会遵循保险业务指令格式，并熟悉条款引用规范。

**数据 Schema**（与 LLaMA-Factory `alpaca` 模板兼容）：

```json
{
  "instruction": "请回答用户的保险问题。",
  "input": "重疾险等待期内确诊是否赔付？",
  "output": "等待期内确诊一般不予赔付，合同另有约定的除外。具体参见条款 X 第 Y 条。",
  "system": "你是蚂蚁保险的智能客服，需严格依据条款作答。"
}
```

**构造策略**：

| 策略 | 做法 | 来源 | 占比目标 |
|------|------|------|----------|
| 策略 S-A | 条款 + FAQ → instruction/input/output | 解析条款 + FAQ 库 | ≥ 50% |
| 策略 S-B | 历史工单"合规问答"子集 → 改写为 instruction 格式 | 工单系统 | ≥ 30% |
| 策略 S-C | 司内专家标注补全 | 业务标注团队 | ≥ 10%（长尾问题） |

**质量门槛**：

- `output` 中业务关键词（赔付/等待期/免赔/告知）必须引用条款编号（与 DPO 校验规则一致）。
- `output` 长度 ∈ [30, 600] 字符，避免过短（无信息）或过长（噪声）。
- 与 DPO 数据集共享同一份 PII 脱敏管线（直接复用 3.3 节的 `PIIScrubber`）。

**产物**：`data/insurance/insurance_sft_v1.jsonl`（≥ 3000 条），同步上传到 `/shared/datasets/insurance_dpo/`。

---

## 4. 交付物清单

| 编号 | 交付物 | 路径 | 说明 |
|------|--------|------|------|
| D-M02-01 | 数据源采集器 | `m_data/sources/{base,policy,faq,ticket}.py` | 3 类源 + 抽象基类 |
| D-M02-02 | chosen/rejected 配对器 | `m_data/pair_builder.py` | 3 种策略实现 |
| D-M02-03 | PII 脱敏器 | `m_data/pii_scrubber.py` | 正则+词典 |
| D-M02-04 | 规范化器 | `m_data/normalizer.py` | 文本清洗 |
| D-M02-05 | 规则校验器 | `m_data/validator.py` | 6 类规则 |
| D-M02-06 | JSONL 导出器 | `m_data/exporter.py` | 写出 |
| D-M02-07 | 流水线编排入口 | `m_data/pipeline.py` | Collector→Normalizer→...→Exporter |
| D-M02-08 | Judge Prompt 模板 | `m_data/prompts/judge_pairwise.txt` | LLM-as-Judge 用 |
| D-M02-09 | 配置文件 | `configs/data/insurance_dpo_gen.yaml` | 数据源 + 策略参数 |
| D-M02-10 | DPO 数据集产物 | `data/insurance/dpo_train_v1.2.jsonl` | ≥ 5000 条 |
| D-M02-11 | SFT 数据集产物 | `data/insurance/insurance_sft_v1.jsonl` | ≥ 3000 条（M03 SFT 阶段输入） |
| D-M02-12 | 评测留出集 | `data/eval/insurance_qa_500.jsonl` | 从司内保险语料留出 500 条（M05 评测用） |
| D-M02-13 | 数据质量报告 | `reports/dpo_data_quality_v1.2.md` | 通过率 / 分布 / PII 命中率 |
| D-M02-14 | 流水线 README | `m_data/README.md` | 运行说明 |
| D-M02-15 | 单元测试 | `tests/m_data/*.py` | 各模块单测 |

---

## 5. 验收标准

| 类别 | 验收项 | 量化阈值 | 检查方式 |
|------|--------|----------|----------|
| SFT 数据规模 | `insurance_sft_v1.jsonl` 行数 | ≥ 3000 | `wc -l` |
| DPO 数据规模 | jsonl 总行数 | ≥ 5000 | `wc -l` |
| Validator 通过率 | 整体通过率 | ≥ 85% | Validator 报告 |
| PII 安全 | PII 命中率 | = 0 | 正则扫描脚本 |
| 来源分布 | 5 类 source 均有样本，单一 source 占比 | < 70% | 分布表 |
| 必引条款 | "赔付 / 等待期 / 免赔 / 告知" 关键词样本的 chosen 引用条款比例 | = 100% | Validator |
| chosen≠rejected | 字符级相似度 > 0.95 的比例 | < 1% | Validator |
| 长度 | 超长 / 过短样本比例 | < 5% | Validator |
| RAG 对接 | baseline 答案调用成功率 | ≥ 99% | 接口日志 |
| 评测留出 | `insurance_qa_500.jsonl` 不进入训练集 | hash 校验 0 重叠 | 对比脚本 |
| 单测 | m_data 模块单测覆盖率 | ≥ 60% | pytest-cov |
| 可重跑 | 全量重跑耗时 | ≤ 2 h | CI 计时 |

---

## 6. 依赖关系

### 6.1 上游依赖

```
┌──────────┐
│   M01    │  基础设施与环境
└────┬─────┘
     ▼
┌──────────┐
│   M02    │  ← 本阶段
└──────────┘
```

依赖 M01 提供的：
- Docker 基础镜像（Python 3.10 环境）
- 与司内存储 / RAG 端的网络联通
- Judge 模型（Qwen2.5-7B-Instruct）部署就绪

### 6.2 下游依赖

```
┌──────────┐
│   M02    │  ← 本阶段
└────┬─────┘
     ├──────────────────┐
     ▼                  ▼
┌──────────┐      ┌──────────┐
│   M03    │      │   M05    │
│ LoRA+DPO │      │ 评测模块 │
│  训练    │      │ (评测集) │
└──────────┘      └──────────┘
```

- M03 直接消费 `dpo_train_v1.2.jsonl` 作为训练输入。
- M05 也可直接对 M-DPO 输出的中间 checkpoint 做离线评测。

### 6.3 跨阶段接口契约

| 接口 | 契约 | 调用方 |
|------|------|--------|
| `insurance_sft_v1.jsonl` | 每行 JSON，必含 `instruction/input/output/system/source/version` | M03 SFT 阶段 |
| `dpo_train_v1.2.jsonl` | 每行 JSON，必含 `prompt/chosen/rejected/source/pii_scrubbed/version` | M03 |
| `insurance_qa_500.jsonl` | 评测留出集，不进入任何训练管线 | M05 评测 |
| 与司内 RAG 端 `/v1/rag/query` | 请求体 `{user_query, context_docs}`，响应 `{answer, policy_refs}` | 本阶段 → RAG 端 |
| Judge 模型 endpoint | OpenAI 兼容 `/v1/chat/completions`，返回 JSON `{"winner": "A\|B\|TIE", ...}` | 本阶段 → Judge 模型 |
| 共享存储 `/shared/datasets/insurance_dpo/` | 可读写 | 本阶段写出 |

---

## 7. 详细技术规范

### 7.1 DPO 数据 Schema

每行一个 JSON 对象：

```json
{
  "prompt": "用户问题原文",
  "chosen": "合规/正确/专家级答案",
  "rejected": "不合规/错误/普通 RAG 端答案",
  "source": "policy_v1 | faq_v2 | ticket_v3 | llm_judge | retrieval_diff",
  "policy_id": "险种条款编号（可为空）",
  "judge_model": "qwen2.5-7b-instruct（仅 source=llm_judge 时填写）",
  "judge_score_chosen": 0.92,
  "judge_score_rejected": 0.31,
  "pii_scrubbed": true,
  "version": "dpo_v1.2"
}
```

### 7.2 数据集示例

```json
{"prompt": "重疾险等待期内确诊是否赔付？", "chosen": "等待期内确诊一般不予赔付，合同另有约定的除外。具体参见条款 X 第 Y 条。", "rejected": "等待期内确诊也会赔付，公司都会赔的。", "source": "policy_v1", "policy_id": "POL-CRIT-001", "judge_model": null, "judge_score_chosen": null, "judge_score_rejected": null, "pii_scrubbed": true, "version": "dpo_v1.2"}
{"prompt": "百万医疗险的免赔额是怎么计算的？", "chosen": "百万医疗险通常设置 1 万元年度免赔额，通过社保/公费医疗报销的部分可抵扣免赔额。具体以保单条款为准。", "rejected": "百万医疗险没有免赔额，所有费用都能报销。", "source": "faq_v2", "policy_id": null, "judge_model": null, "judge_score_chosen": null, "judge_score_rejected": null, "pii_scrubbed": true, "version": "dpo_v1.2"}
{"prompt": "投保前未告知高血压，理赔会被拒吗？", "chosen": "若投保前未如实告知健康状况，保险公司有权根据《保险法》第十六条解除合同或拒赔；但若超过两年不可抗辩期且非故意，则理赔权利受保护。", "rejected": "这种情况保险公司一定会拒赔，没有任何商量余地。", "source": "llm_judge", "policy_id": null, "judge_model": "qwen2.5-7b-instruct", "judge_score_chosen": 0.91, "judge_score_rejected": 0.28, "pii_scrubbed": true, "version": "dpo_v1.2"}
```

### 7.3 数据流转全景图

```
┌─────────────────────────────────────────────────────────────────┐
│                     M02 数据流转全景                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [保险业务语料]                                                 │
│  ├── 条款 PDF/HTML                                             │
│  ├── 业务 FAQ 库                                                │
│  └── 历史工单(合规问答子集)                                       │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────┐                                                   │
│  │  采集器  │ (Collector)                                        │
│  └──────────┘                                                   │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────┐                                                   │
│  │ PII 脱敏 │ (PIIScrubber)                                      │
│  └──────────┘                                                   │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────┐                                                   │
│  │ 规范化   │ (Normalizer)                                       │
│  └──────────┘                                                   │
│      │                                                          │
│      ├─────────────────────────┐                                │
│      ▼                         ▼                                │
│  ┌─────────────┐         ┌──────────────┐                       │
│  │ SFT 数据集  │         │ DPO 数据集   │                       │
│  │ 构造器      │         │ (策略 A/B/C) │                       │
│  │ (策略 S-A/B/C)│        └──────────────┘                       │
│  └─────────────┘              │                                  │
│      │                        ▼                                  │
│      │                 ┌──────────────┐                          │
│      │                 │ 规则校验     │ ◀── 校验规则库             │
│      │                 │ (Validator)  │                          │
│      │                 └──────────────┘                          │
│      │                        │                                  │
│      │                        ▼                                  │
│      │                 ┌──────────────┐                          │
│      │                 │ Exporter     │                          │
│      │                 └──────────────┘                          │
│      ▼                        ▼                                  │
│  insurance_sft_v1.jsonl   dpo_train_v1.2.jsonl                   │
│      │                        │                                  │
│      └────────┬───────────────┘                                  │
│               ▼                                                  │
│        [共享存储]                                                │
│   /shared/datasets/insurance_dpo/                                │
│               │                                                  │
│               ▼                                                  │
│          M03 训练流程 (输入)                                      │
│                                                                 │
│  ─────────── 以下为评测留出，不进入训练集 ───────────             │
│                                                                 │
│  [保险业务语料] → 随机留出 500 条 → insurance_qa_500.jsonl       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

> 注：医疗问答测试集（`medical_qa_1000.jsonl`）与通用对话测试集（`alpaca_zh_200.jsonl`）均为公开数据集，由 M05 直接导入，不属于 M02 范围。

### 7.4 配置文件示例

```yaml
# configs/data/insurance_dpo_gen.yaml
project: insurance_dpo
seed: 42
version: dpo_v1.2

sources:
  policy:
    enabled: true
    path: /shared/raw/insurance/policies/
    parser: pdf_html
    limit: 1000
  faq:
    enabled: true
    path: /shared/raw/insurance/faq/
    limit: 2000
  ticket:
    enabled: true
    path: /shared/raw/insurance/tickets/
    filter_category: compliance_qa
    limit: 2000

strategies:
  rule_based:
    enabled: true
    rules_file: configs/data/rules/hard_negatives.yaml
  llm_judge:
    enabled: true
    judge_model: qwen2.5-7b-instruct
    judge_endpoint: http://127.0.0.1:8001/v1/chat/completions
    rag_endpoint: http://rag.internal/v1/rag/query
  retrieval_diff:
    enabled: true
    full_index: /shared/indexes/insurance_full/
    trunc_index: /shared/indexes/insurance_trunc/

quality:
  min_prompt_len: 5
  max_prompt_len: 1024
  min_response_len: 10
  max_response_len: 2048
  max_chosen_rejected_similarity: 0.95

output:
  path: data/insurance/dpo_train_v1.2.jsonl
  shared_path: /shared/datasets/insurance_dpo/dpo_train_v1.2.jsonl
```

### 7.5 流水线运行命令

```bash
# 单次运行
python -m m_data.pipeline --config configs/data/insurance_dpo_gen.yaml

# 增量运行（since 参数）
python -m m_data.pipeline --config configs/data/insurance_dpo_gen.yaml --since 2026-06-01

# 干跑（只统计，不写出）
python -m m_data.pipeline --config configs/data/insurance_dpo_gen.yaml --dry-run
```

---

## 8. 风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| Judge 模型服务不可用 | 策略 B 数据为零 | 提前部署并 CI 守护；策略 A / C 数据可独立完成 |
| 司内 RAG 端 `/v1/rag/query` 接口变更 | baseline 答案获取失败 | 适配层封装，接口变更只需改 `m_data/rag_client.py` |
| 条款 PDF 解析失败 | policy_v1 source 数据缺失 | 多解析器 fallback（pdfplumber → PyPDF2 → OCR） |
| LLM-as-Judge 输出格式错误 | pair_builder 拒收 | JSON parser 容错 + 二次重试 |
| Validator 通过率 < 85% | 数据规模不足 | 调低相似度阈值 / 放宽长度限制 / 增加人工标注 |
| PII 脱敏不彻底 | 合规风险 | 双层校验（正则+词典），CI 全量扫描 |

---

## 9. 阶段完成 checklist

- [ ] `m_data/` 模块完整代码（Collector / Normalizer / PIIScrubber / PairBuilder / SFTBuilder / Validator / Exporter / Pipeline）
- [ ] 3 类数据源采集器全部跑通
- [ ] 3 种 chosen/rejected 构造策略全部跑通
- [ ] `data/insurance/dpo_train_v1.2.jsonl` 生成 ≥ 5000 条
- [ ] `data/insurance/insurance_sft_v1.jsonl` 生成 ≥ 3000 条
- [ ] `data/eval/insurance_qa_500.jsonl` 留出 ≥ 500 条
- [ ] Validator 通过率 ≥ 85%
- [ ] PII 命中率 = 0
- [ ] 数据质量报告输出
- [ ] 与司内普通 RAG 端连通性测试通过
- [ ] Judge 模型连通性测试通过
- [ ] `m_data/` 单测覆盖率 ≥ 60%
- [ ] `m_data/README.md` 撰写完成
