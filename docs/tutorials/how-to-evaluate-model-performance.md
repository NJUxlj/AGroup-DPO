# 如何评测模型性能

> 面向开源社区的模型评测教程  
> 对应模块：`src/m_eval/` · 阶段 M05（FR-08）

---

## 目录

1. [评测在流水线中的位置](#1-评测在流水线中的位置)
2. [前置条件](#2-前置条件)
3. [评测架构](#3-评测架构)
4. [评测数据集](#4-评测数据集)
5. [快速开始](#5-快速开始)
6. [配置驱动评测](#6-配置驱动评测)
7. [Baseline 对比](#7-baseline-对比)
8. [评测指标说明](#8-评测指标说明)
9. [报告解读](#9-报告解读)
10. [一键评测脚本](#10-一键评测脚本)
11. [高级用法](#11-高级用法)
12. [常见问题](#12-常见问题)

---

## 1. 评测在流水线中的位置

模型评测是 DPO 流水线的**最终验收环节**：

```
DPO 训练 → merge LoRA → merged 模型
                              ↓
                    M-EVAL 自动评测（本文）
                              ↓
                    JSON + Markdown 双格式报告
```

评测覆盖三个维度：

- **效果**：Accuracy / BLEU-4 / ROUGE-L
- **性能**：p50 / p95 / p99 推理延迟 + 吞吐量
- **对比**：与 baseline 模型的增益分析

---

## 2. 前置条件

### 2.1 环境

```bash
conda activate llm   # 或独立的 vllm 环境
pip install -e ".[inference,evaluation]"
export VLLM_USE_FLASHINFER_SAMPLER=0   # vLLM ≥ 0.22.1 时需要
export PYTHONPATH=src
```

### 2.2 待评测模型

需为 **merge 后的完整 HuggingFace 模型**（safetensors 格式）：

```bash
ls merged_models/qwen2_5_1_5b_insurance_dpo_v1.2/model*.safetensors
```

若尚未 merge：

```bash
copaw-dpo merge \
    --base saves/qwen2_5_1_5b/insurance_sft_v1/lora_merged \
    --adapter saves/qwen2_5_1_5b/insurance_dpo_v1.2/lora \
    --output merged_models/qwen2_5_1_5b_insurance_dpo_v1.2
```

### 2.3 评测数据集

仓库内置 3 个评测集，共 1700 条：

```
data/eval/
├── medical_qa_1000.jsonl     # 1000 条医疗选择题 + 开放题
├── insurance_qa_500.jsonl    # 500 条保险业务回归
└── alpaca_zh_200.jsonl       # 200 条中文通用能力防退化
```

---

## 3. 评测架构

```
m_eval/
├── config.py      # configs/eval.yaml 配置加载
├── metrics.py     # accuracy / bleu_4 / rouge_l 计算
├── latency.py     # p50/p95/p99 延迟统计
├── reporter.py    # JSON + Markdown 双格式报告
└── cli.py         # 评测流水线 CLI
```

评测流水线步骤：

```
加载评测数据 → 构建推理后端 (vLLM) → 批量推理 → 计算指标 → 生成报告
```

---

## 4. 评测数据集

### 4.1 数据格式

每行一条 JSON：

```json
{
  "id": "med_qa_0001",
  "category": "diagnosis",
  "question": "糖尿病患者的空腹血糖控制目标是多少？",
  "reference_answer": "一般成人 2 型糖尿病患者空腹血糖目标为 4.4~7.0 mmol/L",
  "answer_type": "open"
}
```

```json
{
  "id": "med_qa_0002",
  "category": "drug",
  "question": "青霉素过敏患者可以使用头孢类抗生素吗？A. 可以 B. 不可以 C. 部分可以 D. 不确定",
  "reference_answer": "C",
  "answer_type": "choice"
}
```

### 4.2 数据集说明

| 数据集 | 样本量 | 用途 | 来源 |
|--------|--------|------|------|
| `medical_qa_1000` | 1000 | 主评测集（选择题 + 开放题） | CMB-Exam holdout |
| `insurance_qa_500` | 500 | 保险业务回归 | FAQ/工单 holdout + 合成 |
| `alpaca_zh_200` | 200 | 防退化（通用中文能力） | ChineseAlpacaEval |

### 4.3 重建评测集

```bash
python scripts/build_eval_datasets.py           # 全部重建
python scripts/build_eval_datasets.py --only medical   # 仅医疗集
```

---

## 5. 快速开始

### 5.1 单数据集评测

```bash
copaw-dpo evaluate \
    --model merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \
    --eval-data data/eval/insurance_qa_500.jsonl \
    --output reports/eval_report_dpo_v1.2
```

### 5.2 多数据集目录评测

对 `data/eval/` 下所有 `.jsonl` 批量评测：

```bash
copaw-dpo evaluate \
    --model merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \
    --eval-data data/eval/ \
    --output reports/eval_report_dpo_v1.2
```

### 5.3 指定推理后端

```bash
copaw-dpo evaluate \
    --model merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \
    --eval-data data/eval/ \
    --backend vllm \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --output reports/eval_report_dpo_v1.2
```

### 5.4 产出文件

```
reports/
├── eval_report_dpo_v1.2.json    # 结构化 JSON
└── eval_report_dpo_v1.2.md      # Markdown 表格 + 结论
```

---

## 6. 配置驱动评测

推荐使用 [`configs/eval.yaml`](../../configs/eval.yaml) 统一管理评测参数：

```yaml
eval:
  output_dir: reports/
  max_new_tokens: 256
  temperature: 0.3

  datasets:
    - name: medical_qa_1000
      path: data/eval/medical_qa_1000.jsonl
    - name: insurance_qa_500
      path: data/eval/insurance_qa_500.jsonl
    - name: alpaca_zh_200
      path: data/eval/alpaca_zh_200.jsonl

  thresholds:
    accuracy: 0.60
    bleu_4: 0.30
    rouge_l: 0.45
    p50_latency_ms: 1200
    p95_latency_ms: 2500
```

运行：

```bash
copaw-dpo evaluate \
    --config configs/eval.yaml \
    --model merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \
    --output reports/eval_report_dpo_v1.2
```

CLI 参数可覆盖 yaml 中的字段，例如 `--max-new-tokens 512`。

---

## 7. Baseline 对比

对比 DPO 模型与未对齐 baseline 的效果增益：

```bash
# 1. 先评测 baseline（原始 Instruct 模型或 SFT-only 模型）
copaw-dpo evaluate \
    --config configs/eval.yaml \
    --model /path/to/models/Qwen2.5-1.5B-Instruct \
    --output reports/eval_report_baseline_v0

# 2. 评测 DPO 模型并对比
copaw-dpo evaluate \
    --config configs/eval.yaml \
    --model merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \
    --output reports/eval_report_dpo_v1.2 \
    --baseline-report reports/eval_report_baseline_v0.json
```

报告中会自动计算 Accuracy / ROUGE-L 等指标的提升百分比。

---

## 8. 评测指标说明

| 指标 | 计算方式 | 适用场景 | 取值范围 |
|------|----------|----------|----------|
| **Accuracy** | 选择题：正则抽取选项 + 严格匹配；开放题：归一化文本比对或 LLM-as-Judge | 客观题 / 开放题 | 0–1 |
| **BLEU-4** | `sacrebleu.corpus_bleu`（tokenized 13a + brevity penalty） | n-gram 重合度 | 0–1 |
| **ROUGE-L** | `rouge-score` ROUGE-L F-measure（LCS based） | 最长公共子序列重合度 | 0–1 |
| **Latency p50/p95/p99** | 首 token + 总延迟分位数 | 性能验收 | ms |
| **Throughput** | samples/s | 吞吐能力 | samples/s |

### 8.1 Accuracy 分支逻辑

```
answer_type == "choice"
    → 从模型输出中正则抽取 A/B/C/D → 与 reference 严格匹配

answer_type == "open" && judge_required == true
    → 调用推理后端做 LLM-as-Judge 判定

answer_type == "open"（默认）
    → 文本归一化（去空白/标点/大小写）后精确比对
```

### 8.2 验收阈值（configs/eval.yaml）

| 指标 | 阈值 |
|------|------|
| Accuracy | ≥ 0.60 |
| BLEU-4 | ≥ 0.30 |
| ROUGE-L | ≥ 0.45 |
| p50 延迟 | ≤ 1200 ms |
| p95 延迟 | ≤ 2500 ms |

---

## 9. 报告解读

### 9.1 JSON 报告结构

```json
{
  "model_version": "qwen2_5_1_5b_insurance_dpo_v1.2",
  "infer_backend": "vllm",
  "timestamp": "2026-06-23T10:00:00",
  "datasets": {
    "medical_qa_1000": {
      "accuracy": 0.682,
      "bleu_4": 0.341,
      "rouge_l": 0.482,
      "n_samples": 1000
    },
    "insurance_qa_500": {
      "accuracy": 0.755,
      "bleu_4": 0.389,
      "rouge_l": 0.521,
      "n_samples": 500
    }
  },
  "latency": {
    "p50_total_ms": 920.1,
    "p95_total_ms": 1850.3,
    "throughput_samples_per_s": 1.2
  },
  "baseline_comparison": {
    "baseline_model": "qwen2_5_1_5b_instruct_no_dpo",
    "accuracy_gain": "+13.4%",
    "rouge_l_gain": "+9.1%"
  }
}
```

### 9.2 Markdown 报告

Markdown 报告包含：

- 各数据集指标汇总表
- 与 baseline 的对比表
- 延迟统计
- 阈值达标/未达标标记
- 简要结论

### 9.3 如何解读结果

| 观察 | 含义 | 建议 |
|------|------|------|
| insurance_qa ↑, alpaca_zh ↓ | DPO 过拟合业务 | 增大 β 或加入更多通用数据 |
| 全部指标 ↑ | DPO 有效 | 可进入部署 |
| BLEU/ROUGE 低但 Accuracy 高 | 模型回答语义正确但表述不同 | 正常，开放题 BLEU 本身偏低 |
| p95 延迟超标 | 推理性能不足 | 检查 vLLM 配置或使用 tensor parallel |

---

## 10. 一键评测脚本

### 10.1 merge + 全量评测

```bash
bash deploy/run_merge_and_eval.sh
```

自动执行：DPO LoRA merge → 1700 条全量评测 → 产出 JSON + MD 报告。

### 10.2 评测模块全面测试

```bash
bash deploy/run_m_eval_test.sh
```

覆盖：pytest 单测 → 指标计算冒烟 → 全量 vLLM 评测流水线。

### 10.3 评测依赖冒烟

```bash
python scripts/smoke_eval.py
```

---

## 11. 高级用法

### 11.1 Python API

```python
from m_eval.metrics import accuracy_score, bleu_4_score, rouge_l_score
from m_eval.latency import aggregate_latency
from m_eval.reporter import EvalReporter

preds = ["模型生成的答案1", "模型生成的答案2"]
refs  = ["参考答案1", "参考答案2"]

acc   = accuracy_score(preds, refs)
bleu  = bleu_4_score(preds, refs)
rouge = rouge_l_score(preds, refs)

reporter = EvalReporter(
    model_version="qwen2_5_1_5b_insurance_dpo_v1.2",
    infer_backend="vllm",
)
reporter.add_dataset("insurance_qa", accuracy=acc, bleu_4=bleu, rouge_l=rouge, n_samples=2)
json_path, md_path = reporter.write("reports/eval_report_custom")
```

### 11.2 使用 xinference 后端评测

```bash
copaw-dpo evaluate \
    --model merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \
    --eval-data data/eval/insurance_qa_500.jsonl \
    --backend xinference \
    --output reports/eval_xinfer
```

需先启动 xinference 服务并注册模型。

### 11.3 自定义评测集

创建 JSONL 文件并直接传入：

```bash
copaw-dpo evaluate \
    --model merged_models/my_model \
    --eval-data my_eval_set.jsonl \
    --output reports/my_eval
```

每行格式：

```json
{"question": "...", "reference_answer": "...", "answer_type": "open"}
```

---

## 12. 常见问题

| 问题 | 解决方案 |
|------|----------|
| vLLM 启动 OOM | 降低 `--gpu-memory-utilization`；使用更小模型 smoke 测试 |
| BLEU-4 全为 0 | 参考答案 < 4 tokens 时 BLEU 无法计算 4-gram，属正常 |
| ROUGE-L 中文偏低 | `rouge-score` 默认按空格分词；短文本 ROUGE 普遍偏低 |
| 评测耗时过长 | 1700 条全量约 30–60 min；可先用单数据集 smoke |
| `judge_required` 样本慢 | LLM-as-Judge 需额外推理调用，大评测集显著增加耗时 |
| baseline 对比无数据 | 确认 `--baseline-report` 指向有效的 JSON 报告 |
| ModuleNotFoundError: vllm | 安装 `pip install "vllm>=0.8.5,<0.9.0"` |

---

## 附录：完整流水线回顾

```bash
# 0. 部署
pip install -e ".[all]"

# 1. 数据
copaw-dpo data --config configs/data/insurance_dpo_gen.yaml

# 2. SFT
copaw-dpo train --config configs/train_lora_qwen2_5_1_5b_insurance.yaml
copaw-dpo merge --base <base> --adapter saves/.../sft/lora --output saves/.../lora_merged

# 3. DPO
copaw-dpo train --config configs/train_dpo_qwen2_5_1_5b_insurance.yaml
copaw-dpo merge --base saves/.../lora_merged --adapter saves/.../dpo/lora \
    --output merged_models/qwen2_5_1_5b_insurance_dpo_v1.2

# 4. 评测（本文）
copaw-dpo evaluate --config configs/eval.yaml \
    --model merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \
    --output reports/eval_report_dpo_v1.2 \
    --baseline-report reports/eval_report_baseline_v0.json

# 5. 部署推理服务
copaw-dpo infer --config configs/infer.yaml --host 0.0.0.0 --port 8080
```
