#!/usr/bin/env bash
# xinference 单卡推理冒烟测试 - M01 阶段交付物 D-M01-07
# scripts/smoke_xinfer.sh
# 流程: 启动 xinference -> 注册模型 -> 推理 5 条 -> 关闭服务

set -euo pipefail

LOG_DIR=${LOG_DIR:-logs}
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/m01_smoke_xinfer_${TS}.log"

echo "[smoke-xinfer] ${TS} starting" | tee "${LOG_FILE}"

# 1. 启动 xinference 服务 (后台)
echo "[smoke-xinfer] launching xinference ..." | tee -a "${LOG_FILE}"
xinference-local -H 0.0.0.0 -p 9997 > "${LOG_DIR}/xinference_${TS}.out" 2>&1 &
XINFER_PID=$!
echo "[smoke-xinfer] xinference pid=${XINFER_PID}" | tee -a "${LOG_FILE}"

# 等待服务就绪
echo "[smoke-xinfer] waiting for xinference ready ..." | tee -a "${LOG_FILE}"
for i in {1..60}; do
    if curl -fs http://127.0.0.1:9997/v1/models > /dev/null 2>&1; then
        echo "[smoke-xinfer] xinference ready (after ${i}s)" | tee -a "${LOG_FILE}"
        break
    fi
    sleep 1
done

# 2. 注册模型
echo "[smoke-xinfer] registering model ..." | tee -a "${LOG_FILE}"
REGISTER_RESP=$(curl -fsS -X POST http://127.0.0.1:9997/v1/models \
    -H "Content-Type: application/json" \
    -d '{
        "model_name": "qwen_smoke",
        "model_path": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_type": "LLM"
    }' || echo "register_failed")
echo "[smoke-xinfer] register response: ${REGISTER_RESP}" | tee -a "${LOG_FILE}"

# 3. 推理 5 条
QUERIES=(
    "保险等待期内确诊是否赔付？"
    "百万医疗险的免赔额是怎么计算的？"
    "投保前未告知高血压, 理赔会被拒吗？"
    "什么是重大疾病保险？"
    "保单现金价值是什么？"
)

PASS=0
for q in "${QUERIES[@]}"; do
    RESP=$(curl -fsS -X POST http://127.0.0.1:9997/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"qwen_smoke\",\"prompt\":\"${q}\",\"max_tokens\":64}" 2>/dev/null || echo "")
    if [[ -n "${RESP}" ]] && echo "${RESP}" | grep -q "text"; then
        echo "[smoke-xinfer] [OK] ${q} -> $(echo "${RESP}" | python -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["text"][:60])')" | tee -a "${LOG_FILE}"
        PASS=$((PASS+1))
    else
        echo "[smoke-xinfer] [FAIL] ${q} -> ${RESP}" | tee -a "${LOG_FILE}"
    fi
done

echo "[smoke-xinfer] pass=${PASS}/5" | tee -a "${LOG_FILE}"

# 4. 关闭服务
echo "[smoke-xinfer] stopping xinference ..." | tee -a "${LOG_FILE}"
kill ${XINFER_PID} 2>/dev/null || true
wait ${XINFER_PID} 2>/dev/null || true

if [[ ${PASS} -eq 5 ]]; then
    echo "[smoke-xinfer] status: PASS" | tee -a "${LOG_FILE}"
    exit 0
else
    echo "[smoke-xinfer] status: FAIL" | tee -a "${LOG_FILE}"
    exit 1
fi
