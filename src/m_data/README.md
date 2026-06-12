# m_data — DPO 数据集生成流水线 (M02)

> 阶段：M02 DPO 数据集生成流水线
> 关联文档：[M02_DPO数据集生成流水线.md](../../docs/项目分阶段方案/M02_DPO数据集生成流水线.md)

---

## 1. 模块概览

```
m_data/
├── __init__.py                  # 模块顶层导出
├── cli.py                       # CLI 入口
├── normalizer.py                # 文本规范化器
├── pii_scrubber.py              # PII 脱敏器（正则 + 词典）
├── pair_builder.py              # Chosen/Rejected 配对构造器（3 策略）
├── sft_builder.py               # SFT 数据集构造器
├── validator.py                 # 规则校验器（6 类规则）
├── exporter.py                  # JSONL 导出器
├── pipeline.py                  # 流水线编排
├── prompts/
│   └── judge_pairwise.txt       # LLM-as-Judge 提示词模板
└── sources/
    ├── __init__.py
    ├── base.py                  # DataSource 抽象基类 + RawRecord
    ├── policy.py                # 保险条款数据源
    ├── faq.py                   # FAQ 数据源
    └── ticket.py                # 工单数据源
```

---

## 2. 快速开始

### 2.1 本地干跑（不写文件）

```bash
cd /root/autodl-tmp/agroup-dpo
PYTHONPATH=src python -m m_data.cli \
    --config configs/data/insurance_dpo_gen.yaml \
    --dry-run --verbose
```

### 2.2 完整运行

```bash
PYTHONPATH=src python -m m_data.cli \
    --config configs/data/insurance_dpo_gen.yaml
```

### 2.3 增量运行

```bash
PYTHONPATH=src python -m m_data.cli \
    --config configs/data/insurance_dpo_gen.yaml \
    --since 2026-06-01
```

---

## 3. 流水线阶段

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
                 dpo_train_v1.2.jsonl    insurance_sft_v1.jsonl
```

---

## 4. 数据 Schema

### 4.1 DPO 样本 (dpo_train_v1.2.jsonl)

```json
{
  "prompt": "用户问题原文",
  "chosen": "合规/正确/专家级答案",
  "rejected": "不合规/错误/普通RAG端答案",
  "source": "rule_based | llm_judge | retrieval_diff",
  "policy_id": "险种条款编号（可为空）",
  "judge_model": "qwen2.5-7b-instruct（仅source=llm_judge）",
  "judge_score_chosen": 0.92,
  "judge_score_rejected": 0.31,
  "pii_scrubbed": true,
  "version": "dpo_v1.2"
}
```

### 4.2 SFT 样本 (insurance_sft_v1.jsonl)

```json
{
  "instruction": "请回答用户的保险问题。",
  "input": "重疾险等待期内确诊是否赔付？",
  "output": "等待期内确诊一般不予赔付...",
  "system": "你是AI财保助理，需严格依据条款作答。"
  "source": "sft_s-a",
  "version": "sft_v1"
}
```

---

## 5. 三种配对策略

| 策略 | 标识 | 说明 | 依赖 |
|------|------|------|------|
| 策略 A: 规则硬负例 | `rule_based` | 内置模板 + FAQ/工单记录构造正例，随机负例 | 无外部依赖 |
| 策略 B: LLM-as-Judge | `llm_judge` | Qwen2.5-7B-Instruct 对比 RAG 与专家答案 | Judge 模型 endpoint |
| 策略 C: 检索差异 | `retrieval_diff` | 完整索引 vs 截断索引 RAG 答案对比 | 司内 RAG 端 endpoint |

策略 B/C 可通过配置文件 `enabled: false` 关闭。

---

## 6. 校验规则

| 规则 | 阈值 | 不通过处理 |
|------|------|------------|
| prompt 长度 | [5, 1024] | 丢弃 |
| chosen/rejected 长度 | [10, 2048] | 丢弃 |
| PII 检测 | 0 命中 | 回退重脱敏 |
| 条款引用 | 业务关键词 chosen 必须含条款/保单/合同/法 | 丢弃 |
| chosen≠rejected | 相似度 < 0.95 | 丢弃 |
| pii_scrubbed 标记 | 必须为 true | 丢弃 |

---

## 7. 测试

```bash
cd /root/autodl-tmp/agroup-dpo
PYTHONPATH=src python -m pytest tests/m_data/ -v
```

当前测试覆盖：Validator、PIIScrubber、PairBuilder、Exporter 共 43 个用例。
