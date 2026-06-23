#!/usr/bin/env bash
# 在 server2 conda env llm 中执行 m_trainer 全量测试 + CustomTrainer SFT/DPO 真实微调

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/m_trainer_test_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}" "${PROJECT_ROOT}/saves/smoke"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================"
echo " m_trainer 全量测试 + SFT/DPO 微调"
echo " 时间: $(date)"
echo " 项目: ${PROJECT_ROOT}"
echo "========================================"

# 激活 conda
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
elif [ -f /root/autodl-tmp/miniconda/etc/profile.d/conda.sh ]; then
    source /root/autodl-tmp/miniconda/etc/profile.d/conda.sh
fi
conda activate llm

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

# 确保关键依赖已安装
python -c "import loguru" 2>/dev/null || pip install -q loguru==0.7.2
python -c "import pytest" 2>/dev/null || pip install -q pytest

echo ""
echo ">>> 环境信息"
python --version
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
fi

MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct"
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: 模型不存在 ${MODEL_PATH}"
    exit 1
fi
echo "模型路径: ${MODEL_PATH} ✓"

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

# ---- 1. pytest 单元测试 ----
run_step "pytest tests/m_trainer/" \
    python -m pytest tests/m_trainer/ -v --tb=short

# ---- 2. M04 后端冒烟 ----
run_step "smoke_m04.py 后端冒烟" \
    python scripts/smoke_m04.py

# ---- 3. CustomTrainer SFT 真实微调 ----
run_step "CustomTrainer SFT 微调 (20 steps)" \
    python -m m_trainer.cli \
        --config configs/smoke_custom_sft_insurance.yaml \
        --backend accelerate

# 验证 SFT checkpoint
SFT_CKPT="${PROJECT_ROOT}/saves/smoke/custom_sft_insurance/checkpoint-final"
if [ -d "${SFT_CKPT}" ]; then
    echo "SFT checkpoint 存在: ${SFT_CKPT}"
    ls -la "${SFT_CKPT}" | head -10
else
    echo "ERROR: SFT checkpoint 未生成"
    exit 1
fi

# ---- 4. CustomTrainer DPO 真实微调 ----
run_step "CustomTrainer DPO 微调 (20 steps)" \
    python -m m_trainer.cli \
        --config configs/smoke_custom_dpo_insurance.yaml \
        --backend accelerate

# 验证 DPO checkpoint
DPO_CKPT="${PROJECT_ROOT}/saves/smoke/custom_dpo_insurance/checkpoint-final"
if [ -d "${DPO_CKPT}" ]; then
    echo "DPO checkpoint 存在: ${DPO_CKPT}"
    ls -la "${DPO_CKPT}" | head -10
else
    echo "ERROR: DPO checkpoint 未生成"
    exit 1
fi

# ---- 汇总 ----
echo ""
echo "========================================"
echo " 测试汇总"
echo "========================================"
echo "  PASS: ${PASS}"
echo "  FAIL: ${FAIL}"
echo "  日志: ${LOG_FILE}"
echo "========================================"

if [ "${FAIL}" -gt 0 ]; then
    echo "有步骤失败, 请检查日志"
    exit 1
fi

echo "全部通过 🎉"
