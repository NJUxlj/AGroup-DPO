#!/usr/bin/env bash
# server2 双卡测试: DeepSpeed ZeRO3 (SFT+DPO) → Megatron TP=2 (SFT+DPO)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/m_trainer_dual_gpu_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}" "${PROJECT_ROOT}/saves/smoke"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================"
echo " m_trainer 双卡测试 (DeepSpeed + Megatron)"
echo " 时间: $(date)"
echo "========================================"

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
elif [ -f /root/autodl-tmp/miniconda/etc/profile.d/conda.sh ]; then
    source /root/autodl-tmp/miniconda/etc/profile.d/conda.sh
fi
conda activate llm

python -c "import loguru" 2>/dev/null || pip install -q loguru==0.7.2

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=120
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

clear_gpu() {
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 5
}

GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
echo "GPU 数量: ${GPU_COUNT}"
if [ "${GPU_COUNT}" -lt 2 ]; then
    echo "ERROR: 需要至少 2 张 GPU"
    exit 1
fi

MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct"
[ -d "${MODEL_PATH}" ] || { echo "模型不存在: ${MODEL_PATH}"; exit 1; }

PASS=0
FAIL=0

run_step() {
    local name="$1"
    shift
    echo ""
    echo "========================================"
    echo ">>> ${name}"
    echo "========================================"
    if "$@"; then
        echo ">>> ${name}: PASS ✓"
        PASS=$((PASS + 1))
    else
        echo ">>> ${name}: FAIL ✗"
        FAIL=$((FAIL + 1))
    fi
}

verify_ckpt() {
    local ckpt="$1"
    if [ -d "${ckpt}" ]; then
        if [ -f "${ckpt}/adapter_model.safetensors" ] || [ -f "${ckpt}/adapter_config.json" ] || [ -f "${ckpt}/model.safetensors" ] || [ -f "${ckpt}/config.json" ]; then
            echo "Checkpoint OK: ${ckpt}"
            ls -la "${ckpt}" | head -8
            return 0
        fi
    fi
    echo "ERROR: checkpoint 未生成: ${ckpt}"
    return 1
}

# ---- Phase 1: DeepSpeed 双卡 ----
clear_gpu
run_step "DeepSpeed 双卡 SFT (20 steps)" \
    deepspeed --num_gpus=2 --master_port=29500 \
        --module m_trainer.cli -- \
        --config configs/smoke_custom_sft_insurance_deepspeed_2gpu.yaml \
        --backend deepspeed

verify_ckpt "${PROJECT_ROOT}/saves/smoke/custom_sft_insurance_deepspeed_2gpu/checkpoint-final" \
    || { FAIL=$((FAIL + 1)); }

clear_gpu
run_step "DeepSpeed 双卡 DPO (20 steps)" \
    deepspeed --num_gpus=2 --master_port=29501 \
        --module m_trainer.cli -- \
        --config configs/smoke_custom_dpo_insurance_deepspeed_2gpu.yaml \
        --backend deepspeed

verify_ckpt "${PROJECT_ROOT}/saves/smoke/custom_dpo_insurance_deepspeed_2gpu/checkpoint-final" \
    || { FAIL=$((FAIL + 1)); }

# ---- Phase 2: Megatron TP=2 ----
clear_gpu
if ! python -c "import megatron.core" 2>/dev/null; then
    echo "安装 megatron-core..."
    pip install -q megatron-core 2>&1 | tail -3 || \
        pip install -q "git+https://github.com/NVIDIA/Megatron-LM.git" 2>&1 | tail -3 || true
fi

if python -c "import megatron.core" 2>/dev/null; then
    run_step "Megatron TP=2 SFT (10 steps)" \
        torchrun --nproc_per_node=2 --nnodes=1 \
            --node_rank=0 --master_addr=127.0.0.1 --master_port=29502 \
            -m m_trainer.cli \
            --config configs/smoke_custom_sft_insurance_megatron_2gpu.yaml \
            --backend megatron

    verify_ckpt "${PROJECT_ROOT}/saves/smoke/custom_sft_insurance_megatron_2gpu/checkpoint-final" \
        || { FAIL=$((FAIL + 1)); }

    clear_gpu
    run_step "Megatron TP=2 DPO (10 steps)" \
        torchrun --nproc_per_node=2 --nnodes=1 \
            --node_rank=0 --master_addr=127.0.0.1 --master_port=29503 \
            -m m_trainer.cli \
            --config configs/smoke_custom_dpo_insurance_megatron_2gpu.yaml \
            --backend megatron

    verify_ckpt "${PROJECT_ROOT}/saves/smoke/custom_dpo_insurance_megatron_2gpu/checkpoint-final" \
        || { FAIL=$((FAIL + 1)); }
else
    echo "ERROR: megatron-core 未安装，跳过 Megatron 双卡测试"
    FAIL=$((FAIL + 2))
fi

echo ""
echo "========================================"
echo " 双卡测试汇总"
echo "========================================"
echo "  PASS: ${PASS}"
echo "  FAIL: ${FAIL}"
echo "  日志: ${LOG_FILE}"
echo "========================================"

[ "${FAIL}" -eq 0 ] || exit 1
echo "全部通过 🎉"
