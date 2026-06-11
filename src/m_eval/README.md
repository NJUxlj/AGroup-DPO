# M-EVAL 评测模块

> FR-08 下半：Accuracy / BLEU-4 / ROUGE-L + 推理延迟统计 + 双格式报告

## 架构

```
m_eval/
├── metrics.py     # accuracy_score / bleu_4_score / rouge_l_score
├── latency.py     # LatencyStat + aggregate_latency (p50/p95/p99)
├── reporter.py    # EvalReporter（JSON + Markdown 双格式）
└── cli.py         # 评测流水线 CLI
```

## 快速开始

### 1. 指标计算

```python
from m_eval.metrics import accuracy_score, bleu_4_score, rouge_l_score

preds = ["模型生成的答案1", "模型生成的答案2"]
refs  = ["参考答案1", "参考答案2"]

acc   = accuracy_score(preds, refs)   # 分类题精确匹配 / 开放题归一化比对
bleu  = bleu_4_score(preds, refs)     # sacrebleu BLEU-4 (0-1)
rouge = rouge_l_score(preds, refs)    # ROUGE-L F-measure (0-1)
```

### 2. 延迟统计

```python
from m_eval.latency import aggregate_latency
from m_infer import InferResponse

responses = [
    InferResponse(text="...", latency_ms=120.0, total_latency_ms=450.0),
    InferResponse(text="...", latency_ms=180.0, total_latency_ms=520.0),
]

stats = aggregate_latency(responses)
print(f"p50 total: {stats.p50_total_ms:.0f}ms")
print(f"throughput: {stats.throughput_samples_per_s:.1f} samples/s")
```

### 3. 评测报告

```python
from m_eval.reporter import EvalReporter
from m_eval.latency import aggregate_latency

reporter = EvalReporter(
    model_version="qwen2_5_1_5b_insurance_dpo_v1.2",
    infer_backend="vllm",
)

# 添加数据集评测结果
reporter.add_dataset("medical_qa", accuracy=0.682, bleu_4=0.341, rouge_l=0.482, n_samples=1000)
reporter.add_dataset("insurance_qa", accuracy=0.755, bleu_4=0.389, rouge_l=0.521, n_samples=500)

# 添加延迟统计
reporter.set_latency(latency_stats)

# 添加 baseline 对比
reporter.set_baseline(
    baseline_model="qwen2_5_1_5b_instruct_no_dpo",
    accuracy_gain="+13.4%",
    rouge_l_gain="+9.1%",
)

# 产出报告
json_path, md_path = reporter.write("reports/eval_report")
```

产出文件：
- `reports/eval_report.json` — 结构化 JSON
- `reports/eval_report.md` — Markdown 表格 + 结论

### 4. 命令行评测流水线

```bash
# 单数据集评测
python -m m_eval.cli \
    --model merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \
    --eval-data data/eval/medical_qa_1000.jsonl \
    --output reports/eval_report

# 批量评测（目录下所有 .jsonl）
python -m m_eval.cli \
    --model merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \
    --eval-data data/eval/ \
    --output reports/eval_report
```

## 评测指标说明

| 指标 | 计算方式 | 适用场景 |
|------|----------|----------|
| Accuracy | 分类题：正则抽取选项 + 严格匹配；开放题：归一化文本比对 | 客观题 / 短答案 |
| BLEU-4 | `sacrebleu.corpus_bleu`（tokenized 13a + brevity penalty） | n-gram 重合度 |
| ROUGE-L | `rouge-score` ROUGE-L（LCS based, mean across samples） | 最长公共子序列重合度 |
| Latency | p50/p95/p99 first-token + total + throughput | 性能验收 |

## 已知限制

- **BLEU-4 短文本**：参考答案 < 4 tokens 时返回 0.0（无法计算 4-gram）
- **ROUGE-L 中文**：`rouge-score` 默认 tokenizer 按空格分词，中文需先用 `jieba` 预处理
- **开放式问答 Accuracy**：当前使用归一化文本比对，大评测集建议配合 LLM-as-Judge

## 评测数据集

评测数据集位于 `data/eval/`，JSONL 格式：

```json
{"id": "med_qa_0001", "category": "diagnosis", "question": "糖尿病患者的空腹血糖控制目标是多少？", "reference_answer": "一般成人 2 型糖尿病患者空腹血糖目标为 4.4~7.0 mmol/L", "answer_type": "open"}
{"id": "med_qa_0002", "category": "drug", "question": "青霉素过敏患者可以使用头孢类抗生素吗？A. 可以 B. 不可以 C. 部分可以 D. 不确定", "reference_answer": "C", "answer_type": "choice"}
```

| 数据集 | 样本量 | 用途 |
|--------|--------|------|
| `medical_qa_1000.jsonl` | 1000（当前 15 条样例） | 主评测集 |
| `insurance_qa_500.jsonl` | 500（当前 10 条样例） | 业务回归 |
| `alpaca_zh_200.jsonl` | 200（当前 8 条样例） | 防退化 |
