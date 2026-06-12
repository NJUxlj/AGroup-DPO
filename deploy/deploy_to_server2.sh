#!/usr/bin/env bash
# 本地一键推送 + 远端执行 M01 烟雾测试
# 流程: rsync 推送 → ssh 启 screen → 在 screen 中跑 run_m01_smoke.sh

set -euo pipefail

LOCAL_ROOT=/Users/xiniuyiliao/Desktop/application_code/AGroup-DPO
REMOTE_HOST="root@connect.bjb2.seetacloud.com"
REMOTE_PORT=16531
REMOTE_PASS='9EyLfeGrqOxs'
REMOTE_ROOT=/root/autodl-tmp/agroup-dpo
SCREEN_NAME=m01_smoke

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

# 1. rsync 推送
log "[1/3] rsync 推送本地代码到远端 ${REMOTE_HOST}:${REMOTE_ROOT}"
SSHPASS="${REMOTE_PASS}" sshpass -e rsync -avz --delete \
    -e "ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no" \
    --exclude='saves/' --exclude='.git/' --exclude='__pycache__/' \
    --exclude='logs/m01_smoke_*.log' --exclude='logs/build_*.log' \
    "${LOCAL_ROOT}/" "${REMOTE_HOST}:${REMOTE_ROOT}/"
log "rsync 完成"

# 2. 在远端建工作目录
log "[2/3] 远端准备工作目录"
SSHPASS="${REMOTE_PASS}" sshpass -e ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no \
    "${REMOTE_HOST}" \
    "mkdir -p ${REMOTE_ROOT}/logs ${REMOTE_ROOT}/saves && echo '远端目录就绪'"

# 3. screen 后台执行烟雾测试
log "[3/3] 远端用 screen 后台执行 M01 烟雾测试"
SSHPASS="${REMOTE_PASS}" sshpass -e ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no \
    "${REMOTE_HOST}" \
    "if screen -ls | grep -q ${SCREEN_NAME}; then
        screen -X -S ${SCREEN_NAME} quit || true
        sleep 2
     fi
     cd ${REMOTE_ROOT} && \
     screen -dmS ${SCREEN_NAME} bash -c 'bash deploy/run_m01_smoke.sh 2>&1 | tee logs/m01_smoke_screen.log; echo DONE > logs/m01_done.flag'
     sleep 2
     screen -ls
     echo 'screen 会话已启动, 主日志: ${REMOTE_ROOT}/logs/m01_smoke_*.log'
     echo '查看实时进度: ssh 进入后, screen -r ${SCREEN_NAME}'
     echo '查看完成标志: cat ${REMOTE_ROOT}/logs/m01_done.flag'"

log ""
log "部署命令已完成, M01 烟雾测试在远端后台跑"
log "本地只需等待 (约 30-60 min, 主要是 PyTorch 2.7 安装 + 镜像构建)"
log "完成后用 deploy/fetch_m01_logs.sh 拉取日志"
