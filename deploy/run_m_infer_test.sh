#!/usr/bin/env bash
# server2 m_infer 全面测试：单测 + vLLM 冒烟 + CLI + RAG HTTP 服务

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/m_infer_test_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}" "${PROJECT_ROOT}/reports"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================"
echo " m_infer 全面测试 (server2)"
echo " 时间: $(date)"
echo " 日志: ${LOG_FILE}"
echo "========================================"

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
elif [ -f /root/autodl-tmp/miniconda/etc/profile.d/conda.sh ]; then
    source /root/autodl-tmp/miniconda/etc/profile.d/conda.sh
fi

CONDA_SH=""
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    CONDA_SH="source /root/miniconda3/etc/profile.d/conda.sh"
elif [ -f /root/autodl-tmp/miniconda/etc/profile.d/conda.sh ]; then
    CONDA_SH="source /root/autodl-tmp/miniconda/etc/profile.d/conda.sh"
fi

VLLM_ENV="llm"
if [ -d /root/autodl-tmp/envs/vllm ]; then
    VLLM_ENV="vllm"
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/agroup-dpo/merged_models/qwen2_5_1_5b_insurance_dpo_v1.2}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct}"

PASS=0
FAIL=0
SKIP=0

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

skip_step() {
    local name="$1"
    local reason="$2"
    echo ""
    echo ">>> ${name}: SKIP (${reason})"
    SKIP=$((SKIP + 1))
}

clear_gpu() {
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 3
}

# ---- Phase 1: pytest 单测 (llm env) ----
run_step "pytest tests/m_infer/" bash -c "
    ${CONDA_SH} && conda activate llm
    cd '${PROJECT_ROOT}'
    export PYTHONPATH='${PROJECT_ROOT}/src':\${PYTHONPATH:-}
    python -m pytest tests/m_infer/ -v --tb=short
"

# ---- Phase 2: 模型路径检查 ----
echo ""
echo "=== 模型路径检查 ==="
if [ -d "${MODEL_PATH}" ]; then
    echo "DPO merge 模型: ${MODEL_PATH} ✓"
    ls -lh "${MODEL_PATH}"/*.safetensors 2>/dev/null | head -3 || ls -lh "${MODEL_PATH}" | head -5
    USE_MODEL="${MODEL_PATH}"
elif [ -d "${BASE_MODEL}" ]; then
    echo "WARN: merge 模型不存在，回退 base 模型: ${BASE_MODEL}"
    USE_MODEL="${BASE_MODEL}"
else
    echo "ERROR: 无可用模型 (${MODEL_PATH} / ${BASE_MODEL})"
    USE_MODEL=""
fi

if [ -z "${USE_MODEL}" ]; then
    skip_step "vLLM smoke_m05.py" "模型不存在"
    skip_step "m_infer CLI 批量推理" "模型不存在"
    skip_step "RAG HTTP smoke_m05_rag.py" "模型不存在"
else
    export MODEL_PATH="${USE_MODEL}"

    # ---- Phase 3: vLLM 冒烟 (vllm env) ----
    clear_gpu
    run_step "vLLM smoke_m05.py (加载+单条+batch+指标)" bash -c "
        ${CONDA_SH} && conda activate ${VLLM_ENV}
        cd '${PROJECT_ROOT}'
        export PYTHONPATH='${PROJECT_ROOT}/src':\${PYTHONPATH:-}
        export VLLM_USE_FLASHINFER_SAMPLER=0
        export CUDA_VISIBLE_DEVICES=0
        export MODEL_PATH='${USE_MODEL}'
        python scripts/smoke_m05.py
    "

    clear_gpu

    # ---- Phase 4: CLI 批量推理 ----
    run_step "m_infer CLI 批量推理 (2 prompts)" bash -c "
        ${CONDA_SH} && conda activate ${VLLM_ENV}
        cd '${PROJECT_ROOT}'
        export PYTHONPATH='${PROJECT_ROOT}/src':\${PYTHONPATH:-}
        export VLLM_USE_FLASHINFER_SAMPLER=0
        export CUDA_VISIBLE_DEVICES=0
        python -m m_infer.cli \
            --backend vllm \
            --model '${USE_MODEL}' \
            --tensor-parallel-size 1 \
            --max-model-len 2048 \
            --gpu-memory-utilization 0.85 \
            --max-new-tokens 64 \
            --temperature 0.3 \
            --prompts '保险等待期是什么？' '重疾险保障范围有哪些？' \
            --output '${PROJECT_ROOT}/reports/m_infer_cli_smoke.json'
        test -s '${PROJECT_ROOT}/reports/m_infer_cli_smoke.json'
    "

    clear_gpu

    # ---- Phase 5: RAG HTTP 端到端 ----
    run_step "RAG HTTP smoke_m05_rag.py (10/10)" bash -c "
        ${CONDA_SH} && conda activate ${VLLM_ENV}
        cd '${PROJECT_ROOT}'
        export PYTHONPATH='${PROJECT_ROOT}/src':\${PYTHONPATH:-}
        export VLLM_USE_FLASHINFER_SAMPLER=0
        export CUDA_VISIBLE_DEVICES=0
        export MODEL_PATH='${USE_MODEL}'
        export BOOT_WAIT_SEC=120
        python scripts/smoke_m05_rag.py
    "
fi

# ---- Phase 6: xinference 后端（可选） ----
echo ""
echo "=== xinference 后端检测 ==="
if bash -c "${CONDA_SH} && conda activate llm && python -c 'import xinference'" 2>/dev/null; then
    clear_gpu
    run_step "xinference 冒烟 (Python Client 5/5)" bash -c "
        ${CONDA_SH} && conda activate llm
        cd '${PROJECT_ROOT}'
        export MODEL_PATH='${BASE_MODEL}'
        python scripts/smoke_xinfer.py
    " || true
else
    skip_step "xinference 冒烟" "xinference 未安装在 llm 环境"
fi

echo ""
echo "========================================"
echo " m_infer 测试汇总"
echo "========================================"
echo "  PASS: ${PASS}"
echo "  FAIL: ${FAIL}"
echo "  SKIP: ${SKIP}"
echo "  日志: ${LOG_FILE}"
echo "========================================"

[ "${FAIL}" -eq 0 ] || exit 1
echo "全部通过 🎉"
