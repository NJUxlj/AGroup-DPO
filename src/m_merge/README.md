# m_merge — LoRA 与基座合并导出 (FR-06)

将 PEFT LoRA adapter 合并为完整 HuggingFace safetensors 模型，供 vLLM / xinference 直接加载。

## 合并公式

\[
W_{\mathrm{merged}} = W_{\mathrm{base}} + \frac{\alpha}{r} \cdot B \cdot A
\]

| 符号 | 含义 |
|------|------|
| \(W_{\mathrm{base}}\) | 基座模型原权重 |
| \(A \in \mathbb{R}^{r \times d_{\mathrm{in}}}\) | LoRA 低秩矩阵 A |
| \(B \in \mathbb{R}^{d_{\mathrm{out}} \times r}\) | LoRA 低秩矩阵 B |
| \(\alpha\) | 缩放系数（lora_alpha） |
| \(r\) | 秩（lora_rank） |

## 快速开始

### Python API

```python
from m_merge import merge_and_export

merge_and_export(
    base_model_path="Qwen/Qwen2.5-1.5B-Instruct",
    adapter_path="saves/qwen2_5_1_5b/insurance_dpo_v1.2/lora",
    export_dir="merged_models/qwen2_5_1_5b_insurance_dpo_v1.2",
)
```

### CLI

```bash
# 基本用法（CPU）
python -m m_merge.cli \
    --base Qwen/Qwen2.5-1.5B-Instruct \
    --adapter saves/.../lora \
    --output merged_models/my_model

# GPU 加速 + bfloat16 + 保留原始 adapter
python -m m_merge.cli \
    --base /path/to/base \
    --adapter /path/to/lora \
    --output ./merged \
    --device cuda \
    --dtype bfloat16 \
    --keep-adapter \
    --offload-folder /tmp/offload
```

## 产物结构

```
merged_models/<model_name>/
├── config.json
├── generation_config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── merges.txt / vocab.json
└── lora_adapter_backup/          # --keep-adapter 时
```

## API 参考

### `merge_and_export()`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `base_model_path` | `str` | 必填 | 基座模型路径 |
| `adapter_path` | `str` | 必填 | PEFT adapter 路径 |
| `export_dir` | `str` | 必填 | 导出目录 |
| `export_size` | `int` | `5` | 单文件最大 GB |
| `export_device` | `str` | `"cpu"` | `"cpu"` / `"cuda"` |
| `torch_dtype` | `Optional[torch.dtype]` | `None` | 默认自动推断 |
| `offload_folder` | `Optional[str]` | `None` | 大模型磁盘卸载 |

### `_validate_adapter_is_lora()`

校验 `adapter_config.json` 中 `peft_type` 是否为 `"LORA"`。

## 依赖

- `peft >= 0.11.1`
- `transformers >= 4.43.4`
- `torch >= 2.7.1`

## 日志

本模块使用项目统一日志管理器 `CustomLogger`，日志输出到 `logs/m_merge_exporter.log` 和 `logs/m_merge_cli.log`。
