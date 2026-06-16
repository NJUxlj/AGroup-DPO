#!/usr/bin/env bash
# DPO 数据合成全量测试 —— 在 server6 远端 conda env llm 中跑
#
# 项目代码: /root/autodl-tmp/agroup-dpo
# conda env: llm (Python 3.12)
# 测试脚本: tests/m_data/test_dpo_full_pipeline.py
#
# 测试覆盖:
#   1. PolicyStore Milvus 索引（嵌入模型: BAAI/bge-small-zh-v1.5）
#   2. PolicyStore 混合检索验证
#   3. 全量 DPO Pipeline 运行（采集 → 规范化 → 脱敏 → 过滤 → 配对 → 校验+修复 → 写出）
#   4. 输出质量检查（条款引用是否真实）
#   5. Validator 校验统计

set -uo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[FAIL]${NC} $*"; }
die()  { err "$*"; exit 1; }

PROJECT_ROOT=/root/autodl-tmp/agroup-dpo
CONDA_ENV=llm
LOG_DIR=${PROJECT_ROOT}/logs
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="${LOG_DIR}/dpo_data_smoke_${TS}.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

log "===== DPO 数据合成全量测试启动 ${TS} ====="
log "PROJECT_ROOT=${PROJECT_ROOT}"
log "conda env=${CONDA_ENV}"

cd "${PROJECT_ROOT}"

# ----------------------------------------------------------------------
# step 0a: 软链自愈 - miniconda
# ----------------------------------------------------------------------
log "==== 步骤 0a: 软链自愈 /root/miniconda ===="
if [[ -e /root/miniconda ]] && [[ ! -L /root/miniconda ]]; then
    die "/root/miniconda 已存在且不是软链, 拒绝覆盖."
fi
if [[ ! -e /root/miniconda ]]; then
    MINICONDA_REAL=$(ls -d /root/autodl-tmp/miniconda* 2>/dev/null | head -1)
    if [[ -z "${MINICONDA_REAL}" ]]; then
        die "未找到 miniconda 安装目录 (/root/autodl-tmp/miniconda*). 请先安装 miniconda."
    fi
    ln -s "${MINICONDA_REAL}" /root/miniconda \
        && log "已创建软链 /root/miniconda -> ${MINICONDA_REAL}" \
        || die "软链创建失败"
else
    log "软链已存在: $(ls -ld /root/miniconda)"
fi

# ----------------------------------------------------------------------
# step 0b: conda env llm
# ----------------------------------------------------------------------
log "==== 步骤 0b: 准备 conda env: ${CONDA_ENV} ===="
# shellcheck disable=SC1091
source /root/miniconda/etc/profile.d/conda.sh
if ! conda env list | grep -qE "^${CONDA_ENV}\s"; then
    log "创建 conda env: ${CONDA_ENV} (python=3.12)"
    conda create -n ${CONDA_ENV} python=3.12 -y \
        || die "conda env 创建失败"
else
    log "conda env ${CONDA_ENV} 已存在, 跳过创建"
fi

conda activate ${CONDA_ENV} || die "conda activate 失败"
log "当前 Python: $(which python) -> $(python --version)"

# ----------------------------------------------------------------------
# step 1: 基础依赖（数据流水线核心）—— 已安装则跳过
# ----------------------------------------------------------------------
log "==== 步骤 1: 检查并安装数据流水线基础依赖 ===="
MISSING_BASE=false
python -c "import yaml, numpy, pandas, tqdm, loguru, pydantic, httpx, openai" 2>/dev/null \
    && log "基础依赖已就绪, 跳过安装" \
    || MISSING_BASE=true

if [[ "${MISSING_BASE}" == "true" ]]; then
    log "安装基础依赖 (无缓存, 可见进度)..."
    pip install --no-cache-dir \
        numpy==1.26.4 \
        pandas==2.2.2 \
        pyyaml==6.0.1 \
        tqdm==4.66.4 \
        loguru==0.7.2 \
        pydantic==2.8.2 \
        httpx==0.27.0 \
        openai==1.35.0 \
        || die "基础依赖安装失败"
fi

# ----------------------------------------------------------------------
# step 2: PolicyStore 依赖（Milvus + sentence-transformers）—— 已安装则跳过
# ----------------------------------------------------------------------
log "==== 步骤 2: 检查并安装 PolicyStore 依赖 ===="
MISSING_PS=false
python -c "import pymilvus; import sentence_transformers" 2>/dev/null \
    && log "PolicyStore 依赖已就绪, 跳过安装" \
    || MISSING_PS=true

if [[ "${MISSING_PS}" == "true" ]]; then
    log "安装 pymilvus (无缓存, 可见进度)..."
    pip install --no-cache-dir "pymilvus>=2.4.0" || die "pymilvus 安装失败"
    log "安装 sentence-transformers (包较大, 约 2GB, 耐心等待)..."
    pip install --no-cache-dir "sentence-transformers>=2.7.0" || die "sentence-transformers 安装失败"
fi

log "pymilvus: $(python -c 'import pymilvus; print(pymilvus.__version__)' 2>/dev/null || echo 'N/A')"
log "sentence-transformers: $(python -c 'import sentence_transformers; print(sentence_transformers.__version__)' 2>/dev/null || echo 'N/A')"

# ----------------------------------------------------------------------
# step 3: HuggingFace 镜像配置（国内加速下载 BGE 模型）
# ----------------------------------------------------------------------
log "==== 步骤 3: HuggingFace 镜像配置 ===="
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
log "HF_ENDPOINT=${HF_ENDPOINT}"

# ----------------------------------------------------------------------
# step 4: 运行 DPO 数据合成全量集成测试
# ----------------------------------------------------------------------
log "==== 步骤 4: 运行 DPO 数据合成全量集成测试 ===="
log "测试脚本: tests/m_data/test_dpo_full_pipeline.py"
log "============================================"

PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}" \
    python "${PROJECT_ROOT}/tests/m_data/test_dpo_full_pipeline.py"
EXIT_CODE=$?

log ""
log "===== DPO 数据合成全量测试结束 ${TS} ====="
log "主日志: ${MASTER_LOG}"
log "退出码: ${EXIT_CODE}"

if [[ ${EXIT_CODE} -eq 0 ]]; then
    log "🎉 DPO 数据合成全量测试全部通过!"
else
    err "⚠️  DPO 数据合成全量测试存在失败项 (退出码=${EXIT_CODE})"
    err "请检查远端日志: ${MASTER_LOG}"
fi

log ""
log "==== 产出文件检查 ===="
log "DPO JSONL: $(ls -lh ${PROJECT_ROOT}/data/insurance/dpo_train_v1.2.jsonl 2>/dev/null || echo 'NOT FOUND')"
log "SFT JSONL: $(ls -lh ${PROJECT_ROOT}/data/insurance/insurance_sft_v1.jsonl 2>/dev/null || echo 'NOT FOUND')"
log "Milvus DB: $(ls -lhd ${PROJECT_ROOT}/milvus_data/ 2>/dev/null || echo 'NOT FOUND')"
log "质量报告: $(ls -lh ${PROJECT_ROOT}/reports/dpo_data_quality_v1.2.md 2>/dev/null || echo 'NOT FOUND')"

exit ${EXIT_CODE}
