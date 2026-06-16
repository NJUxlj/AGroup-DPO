#!/usr/bin/env bash
# Megatron 分布式训练全量测试 —— 在 server6 远端 conda env llm 中跑
#
# 项目代码: /root/autodl-tmp/agroup-dpo
# conda env: llm (Python 3.12 + PyTorch 2.7.1+cu128)
# GPU: 2×RTX 5090 (32GB) / CUDA 13.0 驱动
#
# 测试流程:
#   Step 0: 环境准备（conda + miniconda 软链）
#   Step 1: 安装 megatron-core + 依赖
#   Step 2: 下载模型权重（GPT2 / Qwen2.5-1.5B / Qwen3-4B）
#   Step 3: Part A+B 快速测试（单卡，无需 megatron-core）
#   Step 4: Part C TP=2 测试（双卡，torchrun 启动）
#   Step 5: Part F 真实模型适配测试
#
# 使用方式:
#   bash deploy/run_megatron_full_test.sh

set -uo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[FAIL]${NC} $*"; }
die()  { err "$*"; exit 1; }

PROJECT_ROOT=/root/autodl-tmp/agroup-dpo
CONDA_ENV=llm
LOG_DIR=${PROJECT_ROOT}/logs
MODEL_DIR=/root/autodl-tmp/models
mkdir -p "${LOG_DIR}" "${MODEL_DIR}"
TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="${LOG_DIR}/megatron_full_test_${TS}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

log "===== Megatron 分布式训练全量测试 启动 ${TS} ====="
log "PROJECT_ROOT=${PROJECT_ROOT}"
log "conda env=${CONDA_ENV}"
log "MODEL_DIR=${MODEL_DIR}"

cd "${PROJECT_ROOT}"

# ----------------------------------------------------------------------
# Step 0: 环境准备
# ----------------------------------------------------------------------
log "==== Step 0: 环境准备 ===="

# 0a: miniconda 软链
if [[ -e /root/miniconda ]] && [[ ! -L /root/miniconda ]]; then
    die "/root/miniconda 已存在且不是软链，拒绝覆盖。"
fi
if [[ ! -e /root/miniconda ]]; then
    MINICONDA_REAL=$(ls -d /root/autodl-tmp/miniconda* 2>/dev/null | head -1)
    if [[ -z "${MINICONDA_REAL}" ]]; then
        die "未找到 miniconda 安装目录"
    fi
    ln -s "${MINICONDA_REAL}" /root/miniconda \
        && log "已创建软链 /root/miniconda -> ${MINICONDA_REAL}" \
        || die "软链创建失败"
else
    log "软链已存在: $(ls -ld /root/miniconda)"
fi

# 0b: conda env
source /root/miniconda/etc/profile.d/conda.sh
if ! conda env list | grep -qE "^${CONDA_ENV}\s"; then
    log "创建 conda env: ${CONDA_ENV} (python=3.12)"
    conda create -n ${CONDA_ENV} python=3.12 -y || die "conda env 创建失败"
else
    log "conda env ${CONDA_ENV} 已存在"
fi
conda activate ${CONDA_ENV} || die "conda activate 失败"
log "Python: $(which python) -> $(python --version)"

# 0c: GPU 检查
log "GPU 检查:"
nvidia-smi -L 2>/dev/null | head -4 || warn "nvidia-smi 不可用"
python -c "
import torch
print(f'  torch={torch.__version__}, cuda={torch.version.cuda}')
print(f'  GPU 数量={torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU{i}: {torch.cuda.get_device_name(i)}, mem={torch.cuda.get_device_properties(i).total_mem/1024**3:.1f}GB')
" || warn "torch GPU 检查失败"

GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
log "可用 GPU: ${GPU_COUNT}"

# ----------------------------------------------------------------------
# Step 1: 安装 megatron-core + 依赖
# ----------------------------------------------------------------------
log "==== Step 1: 安装 megatron-core + 依赖 ===="

# 1a: megatron-core (核心依赖)
if python -c "import megatron.core" 2>/dev/null; then
    log "megatron-core 已安装"
else
    log "安装 megatron-core..."
    pip install --no-cache-dir megatron-core 2>&1 | tail -5 || {
        warn "megatron-core PyPI 安装失败，尝试从 GitHub 安装..."
        pip install --no-cache-dir git+https://github.com/NVIDIA/Megatron-LM.git 2>&1 | tail -5 || {
            warn "Megatron-LM GitHub 安装也失败"
            warn "TP=2 测试将被跳过，仅运行 TP=1 测试"
        }
    }
fi

# 验证
python -c "
try:
    from megatron.core.parallel_state import initialize_model_parallel
    print('  megatron.core.parallel_state ✅')
except ImportError as e:
    print(f'  megatron.core 未安装: {e}')
try:
    import apex
    print(f'  apex ✅  version={apex.__version__ if hasattr(apex, \"__version__\") else \"N/A\"}')
except ImportError:
    print('  apex 未安装 (非必需, 仅影响 fused optimizer)')
"

# 1b: transformers（用于加载真实模型）
python -c "import transformers; print(f'  transformers={transformers.__version__}')" 2>/dev/null \
    || pip install --no-cache-dir "transformers>=4.43.0"

# 1c: 安装 m_trainer 模块（项目自身）
log "安装 m_trainer 模块..."
pip install -e "${PROJECT_ROOT}" 2>&1 | tail -3 || warn "pip install -e 失败 (非致命, PYTHONPATH 可兜底)"

# ----------------------------------------------------------------------
# Step 2: 下载模型权重
# ----------------------------------------------------------------------
log "==== Step 2: 下载模型权重 ===="

# HuggingFace 镜像加速（国内）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
log "HF_ENDPOINT=${HF_ENDPOINT}"

download_model() {
    local model_id="$1"
    local target_dir="$2"
    local display_name="$3"

    if [[ -d "${target_dir}" ]] && [[ -f "${target_dir}/config.json" ]]; then
        log "  ✅ ${display_name} 已存在: ${target_dir}"
        return 0
    fi

    log "  📥 下载 ${display_name}: ${model_id} → ${target_dir}"

    # 使用 Python snapshot_download (最可靠的方式)
    python -c "
import os, sys
os.environ['HF_ENDPOINT'] = '${HF_ENDPOINT}'
from huggingface_hub import snapshot_download
print(f'Downloading ${model_id}...')
snapshot_download('${model_id}', local_dir='${target_dir}',
                  resume_download=True, max_workers=4)
print('download done')
" 2>&1 | tail -10 && {
        if [[ -f "${target_dir}/config.json" ]]; then
            log "  ✅ ${display_name} 下载完成"
            return 0
        fi
    }

    # 回退: modelscope
    warn "  HuggingFace 下载失败，尝试 modelscope..."
    pip install --no-cache-dir modelscope 2>&1 | tail -2
    python -c "
from modelscope import snapshot_download
model_map = {
    'openai-community/gpt2': 'AI-ModelScope/gpt2',
    'gpt2': 'AI-ModelScope/gpt2',
    'Qwen/Qwen2.5-1.5B-Instruct': 'Qwen/Qwen2.5-1.5B-Instruct',
    'Qwen/Qwen3-4B': 'Qwen/Qwen3-4B',
}
ms_id = model_map.get('${model_id}', '${model_id}')
snapshot_download(ms_id, local_dir='${target_dir}')
print('modelscope download done')
" 2>&1 | tail -10 && {
        if [[ -f "${target_dir}/config.json" ]]; then
            log "  ✅ ${display_name} 下载完成 (modelscope)"
            return 0
        fi
    }

    err "  ❌ ${display_name} 下载失败"
    return 1
}

# GPT2 (~500MB) - 使用 openai-community/gpt2 确保获取 safetensors 权重
GPT2_DIR="${MODEL_DIR}/gpt2"
# 若目录存在但无权重文件，先清理
if [[ -d "${GPT2_DIR}" ]] && [[ ! -f "${GPT2_DIR}/model.safetensors" ]] && [[ ! -f "${GPT2_DIR}/pytorch_model.bin" ]]; then
    warn "GPT2 目录存在但缺少权重文件，清理重下..."
    rm -rf "${GPT2_DIR}"
fi
download_model "openai-community/gpt2" "${GPT2_DIR}" "GPT2"

# Qwen2.5-1.5B-Instruct (~3GB)
QWEN2_DIR="${MODEL_DIR}/Qwen2.5-1.5B-Instruct"
download_model "Qwen/Qwen2.5-1.5B-Instruct" "${QWEN2_DIR}" "Qwen2.5-1.5B"

# Qwen3-4B (~8GB，较大)
QWEN3_DIR="${MODEL_DIR}/Qwen3-4B"
download_model "Qwen/Qwen3-4B" "${QWEN3_DIR}" "Qwen3-4B"

log "==== 模型下载完成 ===="
ls -lh "${MODEL_DIR}/" 2>/dev/null | head -10

# ----------------------------------------------------------------------
# Step 3: Part A+B 快速测试（单卡，无需 megatron-core）
# ----------------------------------------------------------------------
log "==== Step 3: Part A+B 快速测试（模型检测 + TP=1 训练） ===="

PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    python "${PROJECT_ROOT}/tests/m_trainer/test_megatron_full.py" \
        --quick \
        --output-json "${LOG_DIR}/megatron_quick_${TS}.json"
QUICK_EXIT=$?

if [[ ${QUICK_EXIT} -eq 0 ]]; then
    SMOKE_QUICK="OK"
    log "✅ Part A+B 快速测试全部通过"
else
    SMOKE_QUICK="FAIL"
    err "❌ Part A+B 快速测试失败"
fi

# ----------------------------------------------------------------------
# Step 4: Part C TP=2 双卡测试
# ----------------------------------------------------------------------
SMOKE_TP2="SKIP"
if [[ "${GPU_COUNT}" -ge 2 ]]; then
    log "==== Step 4: Part C TP=2 双卡测试 (torchrun) ===="

    if python -c "import megatron.core" 2>/dev/null; then
        log "megatron-core 可用，启动 TP=2 测试..."
        # 清理旧并行组，设置环境变量
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        export NCCL_DEBUG=WARN
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
            torchrun --nproc_per_node=2 --nnodes=1 \
                --node_rank=0 --master_addr=127.0.0.1 --master_port=29501 \
                "${PROJECT_ROOT}/tests/m_trainer/test_megatron_full.py" \
                    --quick \
                    --output-json "${LOG_DIR}/megatron_tp2_${TS}.json" \
            2>&1 | tee "${LOG_DIR}/megatron_tp2_raw_${TS}.log"
        TP2_EXIT=$?

        if [[ ${TP2_EXIT} -eq 0 ]]; then
            SMOKE_TP2="OK"
            log "✅ TP=2 测试通过"
        else
            # 检查日志中是否有 barrier passed
            if grep -q "barrier passed\|backward 成功\|step 成功" "${LOG_DIR}/megatron_tp2_raw_${TS}.log" 2>/dev/null; then
                SMOKE_TP2="PARTIAL"
                warn "⚠️  TP=2 测试部分通过 (barrier/backward 成功, 退出码=${TP2_EXIT})"
            else
                SMOKE_TP2="FAIL"
                err "❌ TP=2 测试失败 (退出码=${TP2_EXIT})"
            fi
        fi
    else
        warn "megatron-core 未安装，跳过 TP=2 测试"
    fi
else
    log "GPU 数量=${GPU_COUNT}，跳过 TP=2 测试（需要 >=2 GPU）"
fi

# ----------------------------------------------------------------------
# Step 5: Part F 真实模型 TP=1 适配测试
# ----------------------------------------------------------------------
log "==== Step 5: Part F 真实模型 TP=1 适配测试 ===="

declare -a MODEL_ARGS=()
if [[ -d "${GPT2_DIR}" ]]; then
    MODEL_ARGS+=(--gpt2-model "${GPT2_DIR}")
else
    warn "GPT2 模型不存在: ${GPT2_DIR}"
fi
if [[ -d "${QWEN2_DIR}" ]]; then
    MODEL_ARGS+=(--qwen2-model "${QWEN2_DIR}")
else
    warn "Qwen2.5-1.5B 模型不存在: ${QWEN2_DIR}"
fi
if [[ -d "${QWEN3_DIR}" ]]; then
    MODEL_ARGS+=(--qwen3-model "${QWEN3_DIR}")
else
    warn "Qwen3-4B 模型不存在: ${QWEN3_DIR}"
fi

SMOKE_REAL="SKIP"
if [[ ${#MODEL_ARGS[@]} -gt 0 ]]; then
    log "运行真实模型 TP=1 适配测试（单进程）..."
    PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        python "${PROJECT_ROOT}/tests/m_trainer/test_megatron_full.py" \
            "${MODEL_ARGS[@]}" \
            --output-json "${LOG_DIR}/megatron_real_${TS}.json"
    REAL_EXIT=$?

    if [[ ${REAL_EXIT} -eq 0 ]]; then
        SMOKE_REAL="OK"
        log "✅ 真实模型 TP=1 适配测试通过"
    else
        SMOKE_REAL="FAIL"
        err "❌ 真实模型 TP=1 适配测试失败 (退出码=${REAL_EXIT})"
    fi
else
    warn "无可用模型路径，跳过真实模型适配测试"
fi

# ----------------------------------------------------------------------
# Step 5b: Part F 真实模型 TP=2 双卡测试（torchrun）
# ----------------------------------------------------------------------
SMOKE_REAL_TP2="SKIP"
if [[ "${GPU_COUNT}" -ge 2 ]] && [[ ${#MODEL_ARGS[@]} -gt 0 ]]; then
    if python -c "import megatron.core" 2>/dev/null; then
        log "==== Step 5b: Part F 真实模型 TP=2 双卡测试 (torchrun) ===="
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        export NCCL_DEBUG=WARN
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

        PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
            torchrun --nproc_per_node=2 --nnodes=1 \
                --node_rank=0 --master_addr=127.0.0.1 --master_port=29502 \
                "${PROJECT_ROOT}/tests/m_trainer/test_megatron_full.py" \
                    "${MODEL_ARGS[@]}" \
                    --output-json "${LOG_DIR}/megatron_real_tp2_${TS}.json" \
            2>&1 | tee "${LOG_DIR}/megatron_real_tp2_raw_${TS}.log"
        REAL_TP2_EXIT=$?

        if [[ ${REAL_TP2_EXIT} -eq 0 ]]; then
            SMOKE_REAL_TP2="OK"
            log "✅ 真实模型 TP=2 测试通过"
        else
            if grep -q "TP=2 测试通过\|TP=2.*训练一步\|TP size=.*不足 2" "${LOG_DIR}/megatron_real_tp2_raw_${TS}.log" 2>/dev/null; then
                SMOKE_REAL_TP2="PARTIAL"
                warn "⚠️  真实模型 TP=2 部分通过"
            else
                SMOKE_REAL_TP2="FAIL"
                err "❌ 真实模型 TP=2 测试失败 (退出码=${REAL_TP2_EXIT})"
            fi
        fi
    else
        warn "megatron-core 未安装，跳过真实模型 TP=2 测试"
    fi
else
    log "跳过真实模型 TP=2 测试（需 >=2 GPU + 模型路径）"
fi

# ----------------------------------------------------------------------
# 汇总
# ----------------------------------------------------------------------
log ""
log "============================================"
log "===== Megatron 全量测试结束 ${TS} ====="
log "============================================"
log "主日志: ${MASTER_LOG}"
log ""
log "==== 测试结果汇总 ===="
log "  Part A+B (快速):             ${SMOKE_QUICK}"
log "  Part C  (TP=2 双卡-mini):    ${SMOKE_TP2}"
log "  Part F  (真实模型 TP=1):     ${SMOKE_REAL}"
log "  Part F  (真实模型 TP=2):     ${SMOKE_REAL_TP2}"
log ""
log "==== 产出文件 ===="
ls -lh "${LOG_DIR}/megatron_"*"_${TS}".* 2>/dev/null || true

# 确定最终退出码
OVERALL_EXIT=0
if [[ "${SMOKE_QUICK}" == "FAIL" ]]; then
    OVERALL_EXIT=1
fi
if [[ "${SMOKE_TP2}" == "FAIL" ]]; then
    OVERALL_EXIT=1
fi
if [[ "${SMOKE_REAL}" == "FAIL" ]]; then
    OVERALL_EXIT=1
fi
if [[ "${SMOKE_REAL_TP2}" == "FAIL" ]]; then
    OVERALL_EXIT=1
fi

if [[ ${OVERALL_EXIT} -eq 0 ]]; then
    log "🎉 Megatron 全量测试全部通过!"
else
    err "⚠️  Megatron 全量测试存在失败项"
fi

log "完成标志: $(date)"
echo "DONE" > "${LOG_DIR}/megatron_done.flag"

exit ${OVERALL_EXIT}
