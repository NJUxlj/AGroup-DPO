#!/usr/bin/env bash
# 拉取 server2 上的 M01 烟雾测试日志到本地
# 用法: bash deploy/fetch_m01_logs.sh

set -euo pipefail

REMOTE_HOST="root@connect.bjb2.seetacloud.com"
REMOTE_PORT=16531
REMOTE_PASS='9EyLfeGrqOxs'
REMOTE_ROOT=/root/autodl-tmp/agroup-dpo
LOCAL_ROOT=/Users/xiniuyiliao/Desktop/application_code/AGroup-DPO

GREEN='\033[0;32m'
log() { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }

log "[1/3] 检查远端执行状态"
SSHPASS="${REMOTE_PASS}" sshpass -e ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no \
    "${REMOTE_HOST}" \
    "if [[ -f ${REMOTE_ROOT}/logs/m01_done.flag ]]; then
        echo 'STATUS: DONE'
     else
        echo 'STATUS: RUNNING'
     fi
     echo '--- 远端日志列表 (最新 10 个) ---'
     ls -lt ${REMOTE_ROOT}/logs/ 2>/dev/null | head -15"

log "[2/3] 拉取日志到本地 logs/ 目录"
mkdir -p "${LOCAL_ROOT}/logs/server2_m01"
SSHPASS="${REMOTE_PASS}" sshpass -e rsync -avz \
    -e "ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no" \
    "${REMOTE_HOST}:${REMOTE_ROOT}/logs/" "${LOCAL_ROOT}/logs/server2_m01/"

log "[3/3] 拉取完成"
log "本地日志目录: ${LOCAL_ROOT}/logs/server2_m01/"
ls -lt "${LOCAL_ROOT}/logs/server2_m01/" | head -15
