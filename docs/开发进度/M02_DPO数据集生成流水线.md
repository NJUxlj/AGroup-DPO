# M02 DPO 数据集生成流水线 — 进度报告

> 更新日期：2026-06-16
> 当前状态：**✅ 全部完成 + 方案对比修复 — DPO 7247 条 / SFT 7048 条，通过率 100%**

---

## 0. 方案对比修复（2026-06-16）

基于 `pipeline.py` 与 [M02 阶段方案](../项目分阶段方案/M02_DPO数据集生成流水线.md) 的详细对比，修复了以下缺失：

| 编号 | 修复项 | 状态 |
|------|--------|------|
| #1 | 评测留出集自动生成 (`insurance_qa_500.jsonl`) — `_generate_holdout_set()` | ✅ 已修复 |
| #3 | 数据质量报告自动生成 (`reports/dpo_data_quality_v1.2.md`) — `_generate_quality_report()` | ✅ 已修复 |
| #4 | 必引条款失败样本回流重做 — `_validate_and_repair()` + `_repair_missing_reference()` | ✅ 已修复 |
| #5 | Filter 阶段敏感词过滤 — `quality.sensitive_words` 配置支持 | ✅ 已修复 |
| #6 | prompt 级别全局去重 — `_dedup_by_prompt()` | ✅ 已修复 |
| #7 | per-source limit 参数支持 — `_source_limits` 列表 | ✅ 已修复 |

**修复效果**：
- 通过率从 99.06% → **100%**（67 条缺失条款引用被自动修复）
- 新增 24 个单元测试（`tests/m_data/test_pipeline.py`）
- Server6 全量测试：67/67 passed

---

## 1. 交付物完成情况

| 编号 | 交付物 | 路径 | 状态 |
|------|--------|------|------|
| D-M02-01 | 数据源采集器 | `src/m_data/sources/{base,policy,faq,ticket}.py` | ✅ 完成 |
| D-M02-02 | chosen/rejected 配对器 | `src/m_data/pair_builder.py` | ✅ 完成 |
| D-M02-03 | PII 脱敏器 | `src/m_data/pii_scrubber.py` | ✅ 完成 |
| D-M02-04 | 规范化器 | `src/m_data/normalizer.py` | ✅ 完成 |
| D-M02-05 | 规则校验器 | `src/m_data/validator.py` | ✅ 完成 |
| D-M02-06 | JSONL 导出器 | `src/m_data/exporter.py` | ✅ 完成 |
| D-M02-07 | 流水线编排入口 | `src/m_data/pipeline.py` | ✅ 完成 |
| D-M02-08 | Judge Prompt 模板 | `src/m_data/prompts/judge_pairwise.txt` | ✅ 完成 |
| D-M02-09 | 配置文件 | `configs/data/insurance_dpo_gen.yaml` | ✅ 完成 |
| D-M02-10 | DPO 数据集产物 | `data/insurance/dpo_train_v1.2.jsonl` | ✅ **7183 条** (99.06% 通过) |
| D-M02-11 | SFT 数据集产物 | `data/insurance/insurance_sft_v1.jsonl` | ✅ **7048 条** |
| D-M02-12 | 评测留出集 | `data/eval/insurance_qa_500.jsonl` | ✅ 自动生成（流水线 Step 7） |
| D-M02-13 | 数据质量报告 | `reports/dpo_data_quality_v1.2.md` | ✅ 自动生成（流水线 Step 8） |
| D-M02-14 | 流水线 README | `src/m_data/README.md` | ✅ 完成 |
| D-M02-15 | 单元测试 | `tests/m_data/test_*.py` | ✅ 完成（含 24 个新 pipeline 测试） |

---

## 2. 测试结果

### 2.1 本地测试 (macOS, Python 3.12.10)
```
tests/m_data/test_exporter.py .......  [10%]
tests/m_data/test_pair_builder.py ........  [22%]
tests/m_data/test_pii_scrubber.py ..........  [37%]
tests/m_data/test_pipeline.py ........................  [73%]
tests/m_data/test_validator.py ..................  [100%]
============================== 67 passed in 0.13s ==============================
```

### 2.2 Server6 测试 (Linux, Python 3.12.13)
```
============================== 67 passed in 0.15s ==============================
```

### 2.3 流水线端到端生产运行 (2026-06-16 — 修复后)
```
Pipeline Summary
  Collector:    7080 records (policy: 32, faq: 5033, ticket: 2015)
  DPO samples:  7247 (7247 passed — 100% 通过率)
  SFT samples:  7048
  Dedup:        4 条重复 prompt 已移除
  Repair:       67 条缺失条款引用已自动修复
  Elapsed:      0.6s
  Validator:    7247/7247 passed (100.0%)
```

---

## 3. 数据来源构成

| 数据源 | 数量 | 来源 |
|--------|------|------|
| Hard Negative Templates | 203 条 | 人工编写，覆盖 9 大保险类别 |
| FAQ 原始数据 | 33 条 | faq_v1.json |
| FAQ 模板合成 | 5000 条 | gen_templates_data.py (173 问题模板 × 37 类填充词) |
| Ticket 原始数据 | 15 条 | tickets_v1.json |
| Ticket 模板合成 | 2000 条 | gen_templates_data.py |
| Policy 条款 | 5 条 | 5 类保险产品的条款 JSON |
| **总计采集** | **7080** | |
| → DPO 产出 | **7183** | 策略 A 对每条 record 生成 1 条 DPO |
| → SFT 产出 | **7048** | 每条 record 生成 1 条 SFT |

---

## 4. 已实现的三种配对策略

| 策略 | 状态 | 说明 |
|------|------|------|
| 策略 A: 规则硬负例 | ✅ 已用 | 203 个内置模板 + 7080 条 record 动态配对 |
| 策略 B: LLM-as-Judge | ⏳ 待启用 | 代码就绪，需部署 Judge 模型并配置 endpoint |
| 策略 C: 检索差异 | ⏳ 待启用 | 代码就绪，需配置司内 RAG 端 endpoint |

> **注**：当前仅启用策略 A 即已超额完成目标。策略 B/C 可在后续阶段启用以进一步提升数据质量和多样性。

---

## 5. 数据质量指标

| 指标 | 修复前 (2026-06-11) | 修复后 (2026-06-16) |
|------|----------------------|----------------------|
| DPO 总量 | 7183 | 7247（去重后） |
| DPO 通过率 | 99.06% | **100%** |
| DPO 未通过原因 | 68 条缺少 policy_id 引用 | **0 条（67 条自动修复）** |
| SFT 总量 | 7048 | 7048 |
| prompt 去重 | 无 | 移除 4 条重复 |
| 评测留出集 | 手动 | 自动生成 |
| 质量报告 | 无 | 自动生成 Markdown |

---

## 6. 完成总结

M02 DPO 数据集生成流水线 **全部完成 + 方案对比修复**：
- ✅ 15 个交付物全部就绪
- ✅ 67 个单元测试全通过（本地 + server6）
- ✅ DPO 数据集：7247 条（目标 ≥5000，达成率 145%）
- ✅ SFT 数据集：7048 条（目标 ≥3000，达成率 235%）
- ✅ 通过率：**100%**（修复后，原 99.06%）
- ✅ 评测留出集、质量报告均自动生成
- ✅ prompt 去重、敏感词过滤、回流修复、per-source limit 均已实现
