#!/usr/bin/env bash
# server2 m_eval 全面测试：单测 + 指标冒烟 + 全量评测集 vLLM 流水线

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/m_eval_test_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}" "${PROJECT_ROOT}/reports"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================"
echo " m_eval 全面测试 (server2)"
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
    bash -c "${CONDA_SH} && conda activate ${VLLM_ENV} && python -c 'import torch; torch.cuda.empty_cache()'" 2>/dev/null || true
    sleep 3
}

count_eval_samples() {
    local total=0
    for f in "${PROJECT_ROOT}"/data/eval/*.jsonl; do
        [ -f "$f" ] || continue
        local n
        n=$(wc -l < "$f" | tr -d ' ')
        echo "  $(basename "$f"): ${n} 条"
        total=$((total + n))
    done
    echo "  合计: ${total} 条"
}

# ---- Phase 1: pytest 单测 ----
run_step "pytest tests/m_eval/" bash -c "
    ${CONDA_SH} && conda activate llm
    cd '${PROJECT_ROOT}'
    export PYTHONPATH='${PROJECT_ROOT}/src':\${PYTHONPATH:-}
    python -m pytest tests/m_eval/ -v --tb=short
"

# ---- Phase 2: 评测依赖冒烟（无 GPU） ----
run_step "scripts/smoke_eval.py (sacrebleu/rouge)" bash -c "
    ${CONDA_SH} && conda activate llm
    cd '${PROJECT_ROOT}'
    python scripts/smoke_eval.py
"

# ---- Phase 3: 评测集统计 ----
echo ""
echo "=== data/eval 评测集 ==="
count_eval_samples

USE_MODEL="${MODEL_PATH}"
if [ ! -d "${USE_MODEL}" ]; then
    echo "WARN: merge 模型不存在，回退 base: ${BASE_MODEL}"
    USE_MODEL="${BASE_MODEL}"
fi

if [ ! -d "${USE_MODEL}" ]; then
    skip_step "m_eval CLI 全量评测 (data/eval/)" "模型不存在"
    skip_step "逐数据集独立评测" "模型不存在"
else
    export USE_MODEL

    clear_gpu

    # ---- Phase 4: 配置驱动全量评测（3 个 jsonl，1700 条） ----
    run_step "m_eval CLI 全量评测 (configs/eval.yaml + data/eval/)" bash -c "
        ${CONDA_SH} && conda activate ${VLLM_ENV}
        cd '${PROJECT_ROOT}'
        export PYTHONPATH='${PROJECT_ROOT}/src':\${PYTHONPATH:-}
        export VLLM_USE_FLASHINFER_SAMPLER=0
        export CUDA_VISIBLE_DEVICES=0
        python -m m_eval.cli \
            --config configs/eval.yaml \
            --model '${USE_MODEL}' \
            --eval-data data/eval/ \
            --backend vllm \
            --tensor-parallel-size 1 \
            --gpu-memory-utilization 0.85 \
            --output reports/eval_report_server2_full
        test -f reports/eval_report_server2_full.json
        test -f reports/eval_report_server2_full.md
        python -c \"
import json
with open('reports/eval_report_server2_full.json') as f:
    r = json.load(f)
ds = r.get('datasets', {})
names = sorted(ds.keys())
print('datasets in report:', names)
assert len(ds) == 3, f'expected 3 datasets, got {len(ds)}'
for name in ['medical_qa_1000', 'insurance_qa_500', 'alpaca_zh_200']:
    assert name in ds, f'missing dataset {name}'
    assert ds[name]['n_samples'] > 0, f'empty dataset {name}'
print('report validation OK')
\"
    "

    clear_gpu

    # ---- Phase 5: 单文件模式抽检（最小集 alpaca 200 条，验证 CLI 单 dataset 路径） ----
    run_step "单数据集评测: alpaca_zh_200 (200条)" bash -c "
        ${CONDA_SH} && conda activate ${VLLM_ENV}
        cd '${PROJECT_ROOT}'
        export PYTHONPATH='${PROJECT_ROOT}/src':\${PYTHONPATH:-}
        export VLLM_USE_FLASHINFER_SAMPLER=0
        export CUDA_VISIBLE_DEVICES=0
        python -m m_eval.cli \
            --model '${USE_MODEL}' \
            --eval-data data/eval/alpaca_zh_200.jsonl \
            --backend vllm \
            --max-new-tokens 256 \
            --temperature 0.3 \
            --output reports/eval_report_server2_alpaca_zh_200
        test -f reports/eval_report_server2_alpaca_zh_200.json
    "
fi

echo ""
echo "========================================"
echo " m_eval 测试汇总"
echo "========================================"
echo "  PASS: ${PASS}"
echo "  FAIL: ${FAIL}"
echo "  SKIP: ${SKIP}"
echo "  日志: ${LOG_FILE}"
echo "  报告: ${PROJECT_ROOT}/reports/eval_report_server2_*"
echo "========================================"

[ "${FAIL}" -eq 0 ] || exit 1
echo "全部通过 🎉"
