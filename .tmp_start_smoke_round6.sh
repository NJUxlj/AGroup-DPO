#!/bin/bash
# 启动 smoke 第六轮
cd /root/autodl-tmp/ant-group-dpo
rm -f logs/m01_done.flag
screen -dmS m01_smoke bash -c 'source /root/miniconda/bin/activate llm && bash deploy/run_m01_smoke.sh 2>&1 | tee logs/m01_smoke_screen.log; echo DONE > logs/m01_done.flag'
sleep 2
screen -list | grep m01_smoke
