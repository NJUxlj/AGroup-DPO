#!/usr/bin/env bash
# 本地一键推送 + 远端执行 m_trainer 全量测试 + CustomTrainer SFT/DPO 真实微调
# 流程: rsync 推送 → screen 后台跑 run_m_trainer_test.sh

set -euo pipefail

LOCAL_ROOT=/Users/xiniuyiliao/Desktop/application_code/AGroup-DPO
REMOTE_HOST="root@connect.bjb2.seetacloud.com"
REMOTE_PORT=16531
REMOTE_PASS='9EyLfeGrqOxs'
REMOTE_ROOT=/root/autodl-tmp/agroup-dpo
SCREEN_NAME=m_trainer_test

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

log "[1/3] rsync 推送本地代码到远端 ${REMOTE_HOST}:${REMOTE_ROOT}"
SSHPASS="${REMOTE_PASS}" sshpass -e rsync -avz --delete \
    -e "ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no" \
    --exclude='saves/' --exclude='.git/' --exclude='__pycache__/' \
    --exclude='.venv/' --exclude='.venv-eval/' --exclude='venv/' \
    --exclude='.pytest_cache/' --exclude='*.egg-info/' \
    --exclude='logs/m_trainer_*.log' \
    "${LOCAL_ROOT}/" "${REMOTE_HOST}:${REMOTE_ROOT}/"
log "rsync 完成"

log "[2/3] 远端准备工作目录"
SSHPASS="${REMOTE_PASS}" sshpass -e ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no \
    "${REMOTE_HOST}" \
    "mkdir -p ${REMOTE_ROOT}/logs ${REMOTE_ROOT}/saves/smoke && echo '远端目录就绪'"

log "[3/3] 远端用 screen 后台执行 m_trainer 测试 + SFT/DPO 微调"
SSHPASS="${REMOTE_PASS}" sshpass -e ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no \
    "${REMOTE_HOST}" \
    "if screen -ls | grep -q ${SCREEN_NAME}; then
        screen -X -S ${SCREEN_NAME} quit || true
        sleep 2
     fi
     cd ${REMOTE_ROOT} && \
     screen -dmS ${SCREEN_NAME} bash -c 'bash deploy/run_m_trainer_test.sh 2>&1 | tee logs/m_trainer_test_screen.log; echo DONE > logs/m_trainer_done.flag'
     sleep 2
     screen -ls
     echo 'screen 会话已启动, 日志: ${REMOTE_ROOT}/logs/m_trainer_test_screen.log'
     echo '查看进度: ssh 进入后 screen -r ${SCREEN_NAME}'"

log ""
log "部署完成, m_trainer 测试 + SFT/DPO 微调在远端后台运行"
log "完成后查看: ssh 远端 cat ${REMOTE_ROOT}/logs/m_trainer_done.flag"
