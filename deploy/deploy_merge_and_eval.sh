#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT=/Users/xiniuyiliao/Desktop/application_code/AGroup-DPO
REMOTE_HOST="root@connect.bjb2.seetacloud.com"
REMOTE_PORT=16531
REMOTE_PASS='9EyLfeGrqOxs'
REMOTE_ROOT=/root/autodl-tmp/agroup-dpo
SCREEN_NAME=merge_and_eval

GREEN='\033[0;32m'; NC='\033[0m'
log() { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }

log "[1/3] rsync → ${REMOTE_HOST}:${REMOTE_ROOT}"
SSHPASS="${REMOTE_PASS}" sshpass -e rsync -avz \
    -e "ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no" \
    --exclude='saves/' --exclude='.git/' --exclude='__pycache__/' \
    --exclude='.venv/' --exclude='merged_models/' \
    "${LOCAL_ROOT}/" "${REMOTE_HOST}:${REMOTE_ROOT}/"

log "[2/3] 远端准备"
SSHPASS="${REMOTE_PASS}" sshpass -e ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no \
    "${REMOTE_HOST}" "mkdir -p ${REMOTE_ROOT}/logs ${REMOTE_ROOT}/merged_models ${REMOTE_ROOT}/reports"

log "[3/3] screen 后台执行 merge + eval"
SSHPASS="${REMOTE_PASS}" sshpass -e ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no \
    "${REMOTE_HOST}" \
    "if screen -ls | grep -q ${SCREEN_NAME}; then screen -X -S ${SCREEN_NAME} quit || true; sleep 2; fi
     cd ${REMOTE_ROOT} && chmod +x deploy/run_merge_and_eval.sh && \
     screen -dmS ${SCREEN_NAME} bash -c 'bash deploy/run_merge_and_eval.sh 2>&1 | tee logs/merge_and_eval_screen.log; echo DONE > logs/merge_and_eval_done.flag'
     sleep 2 && screen -ls"

log "完成。日志: ${REMOTE_ROOT}/logs/merge_and_eval_screen.log"
