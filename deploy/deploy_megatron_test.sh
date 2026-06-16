#!/usr/bin/env bash
# 本地一键推送 + server6 远程执行 Megatron 分布式训练全量测试
#
# 流程:
#   1. rsync 推送本地代码到 server6
#   2. 远端准备目录
#   3. screen 后台执行全量测试
#
# Server6: connect.westd.seetacloud.com:41038
# GPU: 2×RTX 5090 (32GB) / CUDA 13.0 驱动
# 远端路径: /root/autodl-tmp/agroup-dpo

set -euo pipefail

LOCAL_ROOT=/Users/xiniuyiliao/Desktop/application_code/AGroup-DPO
REMOTE_HOST="root@connect.westd.seetacloud.com"
REMOTE_PORT=41038
REMOTE_PASS='cyi8gWNJki3l'
REMOTE_ROOT=/root/autodl-tmp/agroup-dpo
SCREEN_NAME=megatron_test

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[FAIL]${NC} $*"; }

# ------------------------------------------------------------------
# 1. rsync 推送
# ------------------------------------------------------------------
log "[1/5] rsync 推送本地代码到 server6: ${REMOTE_ROOT}"
SSHPASS="${REMOTE_PASS}" sshpass -e rsync -avz --delete \
    -e "ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no" \
    --exclude='saves/' --exclude='.git/' --exclude='__pycache__/' \
    --exclude='.pytest_cache/' --exclude='*.pyc' \
    --exclude='logs/' --exclude='reports/' \
    --exclude='milvus_data/' --exclude='wandb/' \
    --exclude='outputs/' --exclude='.venv/' \
    --exclude='merged_models/' --exclude='rag_storage/' \
    --exclude='LightRAG/' --exclude='AcademicPaw/' \
    --exclude='GRPO-Factory/' --exclude='Med-Data-Factory/' \
    --exclude='Med-Qwen/' --exclude='Reward-Factory/' \
    --exclude='nanobot/' --exclude='LeetCode/' \
    "${LOCAL_ROOT}/" "${REMOTE_HOST}:${REMOTE_ROOT}/"
log "rsync 完成"

# ------------------------------------------------------------------
# 2. 远端建目录 + 确认环境
# ------------------------------------------------------------------
log "[2/5] 远端准备工作目录 + 环境检查"
SSHPASS="${REMOTE_PASS}" sshpass -e ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no \
    "${REMOTE_HOST}" \
    "mkdir -p ${REMOTE_ROOT}/logs ${REMOTE_ROOT}/reports /root/autodl-tmp/models \
     && echo '远端目录就绪' \
     && echo '--- conda 检查 ---' \
     && ls -d /root/miniconda 2>/dev/null || echo '(软链不存在, 脚本中创建)' \
     && echo '--- GPU 检查 ---' \
     && nvidia-smi -L 2>/dev/null | head -4 || echo '(无 GPU)' \
     && echo '--- 磁盘 ---' \
     && df -h /root/autodl-tmp | tail -1 \
     && echo '--- 模型缓存 ---' \
     && ls -lh /root/autodl-tmp/models/ 2>/dev/null | head -10 || echo '(空)'"

# ------------------------------------------------------------------
# 3. 确保运行脚本可执行
# ------------------------------------------------------------------
log "[3/5] 确保远端脚本可执行"
SSHPASS="${REMOTE_PASS}" sshpass -e ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no \
    "${REMOTE_HOST}" \
    "chmod +x ${REMOTE_ROOT}/deploy/run_megatron_full_test.sh"

# ------------------------------------------------------------------
# 4. screen 后台执行
# ------------------------------------------------------------------
log "[4/5] 远端用 screen 后台执行 Megatron 全量测试"
SSHPASS="${REMOTE_PASS}" sshpass -e ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no \
    "${REMOTE_HOST}" \
    "if screen -ls 2>/dev/null | grep -q ${SCREEN_NAME}; then
        screen -X -S ${SCREEN_NAME} quit || true
        sleep 2
     fi
     cd ${REMOTE_ROOT} && \
     screen -dmS ${SCREEN_NAME} bash -c '
        bash deploy/run_megatron_full_test.sh 2>&1
        echo DONE > logs/megatron_done.flag
     '
     sleep 3
     screen -ls 2>/dev/null | grep ${SCREEN_NAME} || echo '(screen 启动检查)'
     echo ''
     echo 'screen 会话: ${SCREEN_NAME} 已启动'
     echo '主日志: ${REMOTE_ROOT}/logs/megatron_full_test_*.log'
     echo '完成标志: ${REMOTE_ROOT}/logs/megatron_done.flag'
     echo ''
     echo '查看进度:'
     echo '  ssh -p ${REMOTE_PORT} ${REMOTE_HOST}'
     echo '  screen -r ${SCREEN_NAME}'
     echo '  tail -f ${REMOTE_ROOT}/logs/megatron_full_test_*.log'"

# ------------------------------------------------------------------
# 5. 指令提示
# ------------------------------------------------------------------
log ""
log "============================================"
log "部署完成! Megatron 全量测试在 server6 后台运行"
log "预计耗时: 30-60 min（含模型下载 + TP=1/2 测试 + 真实模型适配）"
log ""
log "📊 检查状态:"
log "   ssh -p ${REMOTE_PORT} ${REMOTE_HOST} 'cat ${REMOTE_ROOT}/logs/megatron_done.flag'"
log ""
log "📋 实时日志:"
log "   ssh -p ${REMOTE_PORT} ${REMOTE_HOST} 'tail -f ${REMOTE_ROOT}/logs/megatron_full_test_*.log'"
log ""
log "📥 拉取日志到本地:"
log "   mkdir -p logs/server6_megatron"
log "   SSHPASS='${REMOTE_PASS}' sshpass -e rsync -avz \\"
log "       -e 'ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no' \\"
log "       '${REMOTE_HOST}:${REMOTE_ROOT}/logs/megatron_*' logs/server6_megatron/"
log "============================================"
