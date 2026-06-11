# M-INFER 推理加速后端抽象层

> FR-08 上半：vLLM / xinference 双后端可切换封装

## 架构

```
m_infer/
├── base.py            # InferBackend ABC + InferRequest/InferResponse
├── registry.py        # INFER_REGISTRY 注册表
├── factory.py         # build_infer_backend() 工厂函数
├── vllm_backend.py    # VLLMBackend（PagedAttention + continuous batching）
├── xinference_backend.py  # XinferenceBackend（HTTP /v1/completions）
├── server.py          # FastAPI 推理服务（/v1/insurance/qa + /health）
├── rag_handler.py     # RAG 对接路由 + Pydantic 模型
└── cli.py             # 命令行入口
```

## 快速开始

### 1. Python API 调用

```python
from m_infer import build_infer_backend, InferRequest

# 加载模型
backend = build_infer_backend("vllm", "merged_models/qwen2_5_1_5b_insurance_dpo_v1.2")

# 单条推理
resp = backend.infer(InferRequest(prompt="保险等待期是什么？", max_new_tokens=128))
print(resp.text)

# 批量推理
reqs = [InferRequest(prompt=q) for q in ["问题1", "问题2", "问题3"]]
responses = backend.batch_infer(reqs)

# 释放资源
backend.shutdown()
```

### 2. 命令行推理

```bash
# vLLM 模式
python -m m_infer.cli --backend vllm --model-path merged_models/qwen2_5_1_5b_insurance_dpo_v1.2

# 切换到 xinference（仅修改 --backend 参数）
python -m m_infer.cli --backend xinference --model-path merged_models/...
```

### 3. 启动 HTTP 推理服务（与司内 RAG 端对接）

```bash
python -m m_infer.server \
    --backend vllm \
    --model-path merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \
    --host 0.0.0.0 \
    --port 8080
```

服务启动后：

```bash
# 健康检查
curl http://127.0.0.1:8080/health

# 保险问答
curl -X POST http://127.0.0.1:8080/v1/insurance/qa \
  -H "Content-Type: application/json" \
  -d '{
    "user_query": "保险等待期内确诊是否赔付？",
    "context_docs": [
      {"id": "policy_001", "text": "等待期内确诊不予赔付"}
    ],
    "max_new_tokens": 256,
    "temperature": 0.3
  }'
```

### 4. 注册自定义后端

```python
from m_infer.registry import register_backend
from m_infer.base import InferBackend

class MyBackend(InferBackend):
    def load(self, model_path, **kwargs): ...
    def infer(self, req): ...
    def shutdown(self): ...

register_backend("my_backend", "my_module.my_backend:MyBackend")
backend = build_infer_backend("my_backend", "model_path")
```

## 配置

```yaml
# configs/infer.yaml
infer:
  backend: vllm          # vllm | xinference
  model_path: merged_models/qwen2_5_1_5b_insurance_dpo_v1.2
  vllm:
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.85
    max_model_len: 2048
  xinference:
    server_endpoint: http://127.0.0.1:9997
    model_uid: qwen2_5_insurance
```

## 环境要求

- **vLLM**: vLLM ≥ 0.8.5, CUDA ≥ 12.4
- **xinference**: xinference ≥ 0.15.4, 需先启动 xinference 服务
- **vLLM 0.22.1+**: 需设置 `VLLM_USE_FLASHINFER_SAMPLER=0`

## RAG 接口契约（与司内 RAG 端）

### 请求 `POST /v1/insurance/qa`

```json
{
  "user_query": "重疾险等待期内确诊是否赔付？",
  "context_docs": [
    {"id": "policy_xxx", "text": "等待期内确诊不予赔付..."}
  ],
  "max_new_tokens": 512,
  "temperature": 0.3
}
```

### 响应

```json
{
  "answer": "等待期内确诊一般不予赔付...",
  "policy_refs": ["policy_xxx § 等待期内确诊不予赔付..."],
  "first_token_latency_ms": 180.5,
  "total_latency_ms": 920.1,
  "model_version": "qwen2_5_1_5b_insurance_dpo_v1.2",
  "request_id": "uuid"
}
```
