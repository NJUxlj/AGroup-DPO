#!/usr/bin/env bash
# server2: LoRA merge → 全量 1700 条 m_eval 评测

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/merge_and_eval_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}" "${PROJECT_ROOT}/merged_models" "${PROJECT_ROOT}/reports"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================"
echo " Merge + 全量评测 (server2)"
echo " 时间: $(date)"
echo " 日志: ${LOG_FILE}"
echo "========================================"

CONDA_SH=""
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    CONDA_SH="source /root/miniconda3/etc/profile.d/conda.sh"
elif [ -f /root/autodl-tmp/miniconda/etc/profile.d/conda.sh ]; then
    CONDA_SH="source /root/autodl-tmp/miniconda/etc/profile.d/conda.sh"
fi

VLLM_ENV="llm"
[ -d /root/autodl-tmp/envs/vllm ] && VLLM_ENV="vllm"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

BASE_MODEL="/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct"
DPO_ADAPTER="${PROJECT_ROOT}/saves/smoke/custom_dpo_insurance/checkpoint-final"
MERGE_OUT="${PROJECT_ROOT}/merged_models/qwen2_5_1_5b_insurance_dpo_v1.2"
EVAL_OUT="${PROJECT_ROOT}/reports/eval_report_dpo_v1_2_full"

# ---- Step 0: 检查 DPO adapter ----
if [ ! -f "${DPO_ADAPTER}/adapter_model.safetensors" ]; then
    echo "ERROR: DPO adapter 不存在: ${DPO_ADAPTER}"
    echo "请先运行 CustomTrainer DPO 训练"
    exit 1
fi
echo "DPO adapter: ${DPO_ADAPTER} ✓"
ls -lh "${DPO_ADAPTER}/adapter_model.safetensors"

# ---- Step 1: Merge LoRA → safetensors ----
echo ""
echo ">>> [1/2] m_merge: base + DPO LoRA → ${MERGE_OUT}"
bash -c "
    ${CONDA_SH} && conda activate llm
    cd '${PROJECT_ROOT}'
    export PYTHONPATH='${PROJECT_ROOT}/src':\${PYTHONPATH:-}
    export CUDA_VISIBLE_DEVICES=0
    rm -rf '${MERGE_OUT}'
    python -m m_merge.cli \
        --base '${BASE_MODEL}' \
        --adapter '${DPO_ADAPTER}' \
        --output '${MERGE_OUT}' \
        --device cuda \
        --dtype bfloat16 \
        --size 5
"
if [ ! -f "${MERGE_OUT}/model.safetensors" ] && [ ! -f "${MERGE_OUT}/model-00001-of-00001.safetensors" ]; then
    # 可能是分片命名
    if ! ls "${MERGE_OUT}"/*.safetensors 1>/dev/null 2>&1; then
        echo "ERROR: merge 产物未生成"
        exit 1
    fi
fi
echo "Merge OK:"
ls -lh "${MERGE_OUT}"/*.safetensors 2>/dev/null | head -5

# ---- Step 2: 全量 1700 条评测 ----
echo ""
echo ">>> [2/2] m_eval: 全量评测 data/eval/ (1700 条)"
bash -c "
    ${CONDA_SH} && conda activate ${VLLM_ENV}
    cd '${PROJECT_ROOT}'
    export PYTHONPATH='${PROJECT_ROOT}/src':\${PYTHONPATH:-}
    export VLLM_USE_FLASHINFER_SAMPLER=0
    export CUDA_VISIBLE_DEVICES=0
    python -m m_eval.cli \
        --config configs/eval.yaml \
        --model '${MERGE_OUT}' \
        --eval-data data/eval/ \
        --backend vllm \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.85 \
        --output '${EVAL_OUT}'
    test -f reports/eval_report_dpo_v1_2_full.json
    test -f reports/eval_report_dpo_v1_2_full.md
"

echo ""
echo "========================================"
echo " 完成"
echo "  merge: ${MERGE_OUT}"
echo "  报告:  ${EVAL_OUT}.json / .md"
echo "  日志:  ${LOG_FILE}"
echo "========================================"
