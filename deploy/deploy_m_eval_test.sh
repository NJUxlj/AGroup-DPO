#!/usr/bin/env bash
# 本地推送 + 远端 screen 执行 m_eval 全面测试

set -euo pipefail

LOCAL_ROOT=/Users/xiniuyiliao/Desktop/application_code/AGroup-DPO
REMOTE_HOST="root@connect.bjb2.seetacloud.com"
REMOTE_PORT=16531
REMOTE_PASS='9EyLfeGrqOxs'
REMOTE_ROOT=/root/autodl-tmp/agroup-dpo
SCREEN_NAME=m_eval_test

GREEN='\033[0;32m'; NC='\033[0m'
log() { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }

log "[1/3] rsync 推送到 ${REMOTE_HOST}:${REMOTE_ROOT}"
SSHPASS="${REMOTE_PASS}" sshpass -e rsync -avz --delete \
    -e "ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no" \
    --exclude='saves/' --exclude='.git/' --exclude='__pycache__/' \
    --exclude='.venv/' --exclude='.venv-eval/' --exclude='venv/' \
    --exclude='.pytest_cache/' --exclude='*.egg-info/' \
    "${LOCAL_ROOT}/" "${REMOTE_HOST}:${REMOTE_ROOT}/"

log "[2/3] 远端准备"
SSHPASS="${REMOTE_PASS}" sshpass -e ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no \
    "${REMOTE_HOST}" "mkdir -p ${REMOTE_ROOT}/logs ${REMOTE_ROOT}/reports"

log "[3/3] screen 后台执行 m_eval 测试"
SSHPASS="${REMOTE_PASS}" sshpass -e ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no \
    "${REMOTE_HOST}" \
    "if screen -ls | grep -q ${SCREEN_NAME}; then screen -X -S ${SCREEN_NAME} quit || true; sleep 2; fi
     cd ${REMOTE_ROOT} && chmod +x deploy/run_m_eval_test.sh && \
     screen -dmS ${SCREEN_NAME} bash -c 'bash deploy/run_m_eval_test.sh 2>&1 | tee logs/m_eval_test_screen.log; echo DONE > logs/m_eval_test_done.flag'
     sleep 2 && screen -ls"

log "部署完成，日志: ${REMOTE_ROOT}/logs/m_eval_test_screen.log"
log "注意: 全量 1700 条 + alpaca 单集 200 条，预计耗时 20-60 分钟"
