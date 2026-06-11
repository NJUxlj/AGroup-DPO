# 部署与运行环境说明 (M01 阶段交付物 D-M01-09)

> 蚂蚁集团保险 DPO 项目 · 基础设施与环境准备阶段 (M01)
> 关联文档: [项目分阶段方案/M01_基础设施与环境准备.md](../docs/项目分阶段方案/M01_基础设施与环境准备.md)

---

## 1. 目录结构

```
deploy/
├── Dockerfile.base            # 基础训练镜像 (D-M01-01)
├── Dockerfile.deepspeed       # DeepSpeed ZeRO3 layer (D-M01-02)
├── Dockerfile.fsdp            # FSDP layer (D-M01-02)
├── Dockerfile.megatron        # Megatron-LM layer (D-M01-02)
├── Dockerfile.accelerate      # accelerate layer (D-M01-02)
├── Dockerfile.infer-vllm      # vLLM 推理 layer (D-M01-03)
├── Dockerfile.infer-xinfer    # xinference 推理 layer (D-M01-03)
├── Dockerfile.eval            # 评测 layer (D-M01-04)
├── docker-compose.yml         # 训练节点多后端编排 (D-M01-05a)
├── xinference-compose.yml     # xinference 服务编排 (D-M01-05b)
├── vllm-compose.yml           # vLLM 服务编排 (D-M01-05c)
└── README.md                  # 本文件
```

---

## 2. 镜像分层设计

```
┌────────────────────────────────────────────────────────────┐
│ copaw-dpo-base:latest                                      │
│  python3.10 + cuda12.4 + cudnn8 + pytorch2.4 + LLaMA-Fac  │
└──────────────┬─────────────────────┬──────────────────────┘
               │                     │
   ┌───────────┼──────────┐   ┌─────┴───────┐    ┌──────────┐
   ▼           ▼          ▼   ▼             ▼    ▼          ▼
deepspeed   fsdp      megatron accelerate  vllm  xinfer   eval
   │           │          │   │             │    │          │
   └───── 训练后端 4 选 1 ─┘   └─── 推理 2 选 1 ─┘  └─ 评测 ─┘
```

**好处**: 拉取时按需取 layer, 基础镜像 ≈ 12 GB, 单 layer ≤ 2 GB, CI 推送快。

---

## 3. 构建命令

```bash
# 0. 准备: 登录私有 registry (如需)
# docker login registry.antgroup.com

# 1. 构建基础镜像
docker build -f deploy/Dockerfile.base -t copaw-dpo-base:latest .

# 2. 构建 4 个分布式后端 layer
for backend in deepspeed fsdp megatron accelerate; do
    docker build -f deploy/Dockerfile.$backend -t copaw-dpo-$backend:latest .
done

# 3. 构建 2 个推理 layer
docker build -f deploy/Dockerfile.infer-vllm    -t copaw-dpo-infer-vllm:latest   .
docker build -f deploy/Dockerfile.infer-xinfer  -t copaw-dpo-infer-xinfer:latest .

# 4. 构建评测 layer
docker build -f deploy/Dockerfile.eval -t copaw-dpo-eval:latest .

# 5. 验证: 应有 7 个 copaw-dpo-* 镜像
docker images | grep copaw-dpo
```

**预估时间**:
- 基础镜像: 25 min (含 torch 编译)
- 后端 layer: 5 min/个
- 推理 layer: 8 min/个
- 评测 layer: 4 min
- **总计 ≈ 60 min**

---

## 4. docker-compose 编排

### 4.1 训练节点 (4 个后端)

```bash
# DeepSpeed ZeRO3 训练 (主路径)
docker compose -f deploy/docker-compose.yml up train-deepspeed

# FSDP 训练
docker compose -f deploy/docker-compose.yml up train-fsdp

# accelerate (单机多卡调试)
docker compose -f deploy/docker-compose.yml up train-accelerate

# Megatron (1.5B 暂不启用, 仅留接口)
docker compose -f deploy/docker-compose.yml up train-megatron
```

### 4.2 vLLM 推理服务

```bash
docker compose -f deploy/vllm-compose.yml up -d
# 验证: curl http://localhost:8080/v1/models
```

### 4.3 xinference 推理服务

```bash
docker compose -f deploy/xinference-compose.yml up -d
# 验证: curl http://localhost:9997/v1/models
```

---

## 5. 容器内卷挂载约定

| 容器路径 | 物理路径 | 用途 |
|----------|----------|------|
| `/shared` | `/shared` | 共享存储（数据集 / checkpoint） |
| `/workspace/src` | `../src` | 项目源码 |
| `/workspace/configs` | `../configs` | 训练 / 推理配置 |
| `/workspace/scripts` | `../scripts` | 烟雾测试 / 工具脚本 |
| `/workspace/data` | `../data` | 训练数据（jsonl） |
| `/workspace/saves` | `../saves` | LoRA / DPO 训练产物 |

> 跨节点同步通过 `rsync` 推到 server2, 严格遵循"本地编辑 → 上传"流程。

---

## 6. 烟雾测试执行 (M01 阶段核心)

按 M01 § 3.6 的结果记录模板, 在 server2 上一键跑全套:

```bash
# 1. NCCL 通信检查 (2 卡 all-reduce)
docker run --gpus all --rm -it \
    -e RANK=0 -e LOCAL_RANK=0 -e WORLD_SIZE=2 \
    -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=29500 \
    copaw-dpo-deepspeed:latest \
    torchrun --nproc_per_node=2 --nnodes=1 \
        --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 \
        scripts/check_nccl.py --size_mb 256

# 2. LLaMA-Factory SFT 烟雾
docker run --gpus all --rm -it \
    -v $(pwd)/configs:/workspace/configs \
    -v $(pwd)/data:/workspace/data \
    -v $(pwd)/saves:/workspace/saves \
    copaw-dpo-base:latest \
    llamafactory-cli train configs/smoke_lora_qwen2_5_1_5b.yaml

# 3. vLLM 推理烟雾
docker run --gpus all --rm -it \
    -v $(pwd)/scripts:/workspace/scripts \
    copaw-dpo-infer-vllm:latest \
    python scripts/smoke_vllm.py

# 4. xinference 推理烟雾
docker run --gpus all --rm -it \
    -v $(pwd)/scripts:/workspace/scripts \
    copaw-dpo-infer-xinfer:latest \
    bash scripts/smoke_xinfer.sh

# 5. 评测依赖单元测试
docker run --gpus all --rm -it \
    copaw-dpo-eval:latest \
    python scripts/smoke_eval.py
```

**判定标准**: 5 项全部 PASS → M01 阶段验收通过。

---

## 7. 与 server2 的协同流程

> 严格遵循项目部署规则: **本地编辑 → rsync 推 server2 → 远端只跑不编辑**

```bash
# 本地: 修改 deploy/ 或 scripts/ 后
rsync -avz --delete \
    -e "ssh -p 16531" \
    --exclude='saves/' --exclude='.git/' --exclude='__pycache__/' \
    /Users/xiniuyiliao/Desktop/application_code/Ant-Group-DPO/ \
    root@connect.bjb2.seetacloud.com:/workspace/ant-group-dpo/

# 远端: 进入工作目录, 执行上面的烟雾测试
ssh -p 16531 root@connect.bjb2.seetacloud.com
cd /workspace/ant-group-dpo
bash deploy/run_m01_smoke.sh   # 一键跑 5 项
```

> 若远端发现 bug, 改本地 → 推 → 再跑, 禁止 `vim` 改远端文件。

---

## 8. 常见问题

| 现象 | 排查 | 解决 |
|------|------|------|
| `docker build` 报 nvcc 找不到 | base 镜像 tag 错误 | 严格用 `nvidia/cuda:12.4.1-cudnn8-devel-ubuntu22.04` |
| torchrun 跨卡通信 OOM | `--shm-size` 太小 | 加 `--shm-size=16g` |
| vLLM 启动报 `Illegal memory access` | GPU 驱动 < 535 | 升级驱动 |
| xinference 注册模型超时 | 镜像源未生效 | 配 `XINFERENCE_MODEL_SRC=modelscope` |

---

## 9. 升级策略

| 升级项 | 周期 | 风险 | 升级流程 |
|--------|------|------|----------|
| PyTorch minor | 季度 | 中 | CI 跑 5 项烟雾 + 1 次 100 条 DPO 训练 |
| CUDA minor | 半年 | 高 | 完整重跑 M01 + M03 + M05 验收 |
| LLaMA-Factory | 双月 | 中 | 锁版本升级, smoke + 完整 SFT 各 1 次 |
| DeepSpeed | 季度 | 中 | 锁版本升级, ZeRO3 训练冒烟 |

---

## 10. 阶段交付物对齐

| 编号 | 路径 | 状态 |
|------|------|------|
| D-M01-01 | Dockerfile.base | ✓ |
| D-M01-02 | Dockerfile.{deepspeed,fsdp,megatron,accelerate} | ✓ |
| D-M01-03 | Dockerfile.infer-{vllm,xinfer} | ✓ |
| D-M01-04 | Dockerfile.eval | ✓ |
| D-M01-05a | docker-compose.yml | ✓ |
| D-M01-05b | xinference-compose.yml | ✓ |
| D-M01-05c | vllm-compose.yml | ✓ |
| D-M01-06 | configs/smoke_lora_qwen2_5_1_5b.yaml | ✓ |
| D-M01-07 | scripts/{smoke_vllm.py,smoke_xinfer.sh,smoke_eval.py} | ✓ |
| D-M01-08 | scripts/check_nccl.py | ✓ |
| D-M01-09 | deploy/README.md (本文) | ✓ |
| D-M01-10 | requirements.txt / pyproject.toml | ✓ |
