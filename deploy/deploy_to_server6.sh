#!/usr/bin/env bash
# 本地一键推送 + 远端执行 DPO 数据合成全量测试
# 流程: rsync 推送 → ssh 启 screen → 在 screen 中跑 run_dpo_data_smoke.sh
#
# Server6: connect.westd.seetacloud.com:41038

set -euo pipefail

LOCAL_ROOT=/Users/xiniuyiliao/Desktop/application_code/AGroup-DPO
REMOTE_HOST="root@connect.westd.seetacloud.com"
REMOTE_PORT=41038
REMOTE_PASS='cyi8gWNJki3l'
REMOTE_ROOT=/root/autodl-tmp/agroup-dpo
SCREEN_NAME=dpo_data_smoke

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[FAIL]${NC} $*"; }

# ------------------------------------------------------------------
# 1. rsync 推送
# ------------------------------------------------------------------
log "[1/4] rsync 推送本地代码到 server6: ${REMOTE_ROOT}"
SSHPASS="${REMOTE_PASS}" sshpass -e rsync -avz --delete \
    -e "ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no" \
    --exclude='saves/' --exclude='.git/' --exclude='__pycache__/' \
    --exclude='.pytest_cache/' --exclude='*.pyc' \
    --exclude='logs/' --exclude='reports/' \
    --exclude='milvus_data/' --exclude='wandb/' \
    --exclude='outputs/' --exclude='.venv/' \
    --exclude='merged_models/' --exclude='rag_storage/' \
    "${LOCAL_ROOT}/" "${REMOTE_HOST}:${REMOTE_ROOT}/"
log "rsync 完成"

# ------------------------------------------------------------------
# 2. 远端建目录 + 确认环境
# ------------------------------------------------------------------
log "[2/4] 远端准备工作目录"
SSHPASS="${REMOTE_PASS}" sshpass -e ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no \
    "${REMOTE_HOST}" \
    "mkdir -p ${REMOTE_ROOT}/logs ${REMOTE_ROOT}/reports ${REMOTE_ROOT}/milvus_data ${REMOTE_ROOT}/data/insurance \
     && echo '远端目录就绪' \
     && echo '--- conda env 检查 ---' \
     && ls -d /root/miniconda 2>/dev/null || echo '(miniconda 软链不存在, 将在脚本中创建)' \
     && echo '--- GPU 检查 ---' \
     && nvidia-smi -L 2>/dev/null | head -3 || echo '(无 GPU, 数据合成无需 GPU)' \
     && echo '--- 磁盘 ---' \
     && df -h /root/autodl-tmp | tail -1"

# ------------------------------------------------------------------
# 3. 拷贝运行脚本到远端
# ------------------------------------------------------------------
log "[3/4] 推送运行脚本到远端"
# run_dpo_data_smoke.sh 已在 deploy/ 目录下，rsync 已包含

# ------------------------------------------------------------------
# 4. screen 后台执行全量测试
# ------------------------------------------------------------------
log "[4/4] 远端用 screen 后台执行 DPO 数据合成全量测试"
SSHPASS="${REMOTE_PASS}" sshpass -e ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no \
    "${REMOTE_HOST}" \
    "if screen -ls | grep -q ${SCREEN_NAME}; then
        screen -X -S ${SCREEN_NAME} quit || true
        sleep 2
     fi
     cd ${REMOTE_ROOT} && \
     screen -dmS ${SCREEN_NAME} bash -c 'bash deploy/run_dpo_data_smoke.sh 2>&1 | tee logs/dpo_data_smoke_screen.log; echo DONE > logs/dpo_data_done.flag'
     sleep 2
     screen -ls
     echo ''
     echo 'screen 会话: ${SCREEN_NAME} 已启动'
     echo '主日志: ${REMOTE_ROOT}/logs/dpo_data_smoke_screen.log'
     echo '完成标志: ${REMOTE_ROOT}/logs/dpo_data_done.flag'
     echo ''
     echo '查看实时进度:'
     echo '  ssh -p ${REMOTE_PORT} ${REMOTE_HOST}'
     echo '  screen -r ${SCREEN_NAME}'
     echo '  tail -f ${REMOTE_ROOT}/logs/dpo_data_smoke_*.log'"

log ""
log "============================================"
log "部署完成! DPO 数据合成全量测试在 server6 后台运行"
log "预计耗时: 10-20 min（含 embedding 模型下载 + 索引构建 + Pipeline 全量运行）"
log ""
log "检查状态:"
log "  ssh -p ${REMOTE_PORT} ${REMOTE_HOST} 'cat ${REMOTE_ROOT}/logs/dpo_data_done.flag'"
log ""
log "拉取日志到本地:"
log "  bash deploy/fetch_dpo_data_logs.sh  (待创建)"
log "============================================"
