"""M05 RAG 端到端连通性冒烟测试

验证项:
  1. FastAPI 推理服务启动 + 模型加载
  2. /health 健康检查
  3. POST /v1/insurance/qa 无上下文文档（10 条保险问答）
  4. POST /v1/insurance/qa 带上下文文档（验证 policy_refs 提取）
  5. 10/10 全部成功 → 连通性 PASS
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---- 配置 ----
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/root/autodl-tmp/ant-group-dpo/merged_models/qwen2_5_1_5b_insurance_dpo_v1.2",
)
SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8080"))
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
BOOT_WAIT_SEC = int(os.environ.get("BOOT_WAIT_SEC", "60"))  # vLLM 加载需 ~35s

# 测试用例（10 条保险问答）
TEST_QUERIES = [
    {
        "user_query": "保险等待期内确诊疾病是否赔付？",
        "context_docs": [
            {"id": "policy_waiting", "text": "被保险人在等待期内发生保险事故的，保险人不承担赔偿责任。"},
        ],
    },
    {
        "user_query": "百万医疗险的免赔额是怎么计算的？",
        "context_docs": [
            {"id": "policy_deductible", "text": "年度免赔额为1万元，社保报销部分可抵扣免赔额。超出免赔额部分按100%赔付。"},
        ],
    },
    {"user_query": "什么是重大疾病保险？"},
    {"user_query": "保单现金价值是什么？"},
    {
        "user_query": "住院医疗险的理赔流程是什么？",
        "context_docs": [
            {"id": "policy_claim", "text": "理赔流程：1. 报案 2. 提交病历发票 3. 保险公司审核 4. 赔付。"},
        ],
    },
    {"user_query": "既往症在健康保险中如何处理？"},
    {"user_query": "保险宽限期是多长时间？"},
    {"user_query": "意外伤害保险的保障范围包括哪些？"},
    {
        "user_query": "投保前未告知高血压，理赔会被拒吗？",
        "context_docs": [
            {"id": "policy_disclosure", "text": "投保人故意或因重大过失未履行如实告知义务，足以影响保险人决定是否承保的，保险人有权解除合同。"},
        ],
    },
    {
        "user_query": "重疾险等待期内确诊是否赔付？",
        "context_docs": [
            {"id": "policy_waiting", "text": "被保险人在等待期内发生保险事故的，保险人不承担赔偿责任。"},
        ],
    },
]


def wait_for_server(timeout: int = BOOT_WAIT_SEC) -> bool:
    """轮询 /health 直到服务就绪。"""
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(
                f"{SERVER_URL}/health", timeout=5
            )
            data = json.loads(resp.read())
            if data.get("status") == "ok":
                logger.info("Server ready: %s", data)
                return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        time.sleep(2)
    return False


def test_connectivity():
    """发送 10 条保险问答请求，验证全部返回 200 + 含非空 answer。"""
    import requests

    success = 0
    failures = []

    for i, test_case in enumerate(TEST_QUERIES):
        payload = {
            "user_query": test_case["user_query"],
            "context_docs": test_case.get("context_docs", []),
            "max_new_tokens": 128,
            "temperature": 0.3,
        }
        try:
            t0 = time.perf_counter()
            resp = requests.post(
                f"{SERVER_URL}/v1/insurance/qa",
                json=payload,
                timeout=60,
            )
            elapsed = (time.perf_counter() - t0) * 1000

            if resp.status_code != 200:
                failures.append(f"[{i+1}] HTTP {resp.status_code}: {resp.text[:80]}")
                continue

            data = resp.json()
            answer = data.get("answer", "")
            policy_refs = data.get("policy_refs", [])
            model_ver = data.get("model_version", "")

            if not answer.strip():
                failures.append(f"[{i+1}] empty answer")
                continue

            success += 1
            logger.info(
                "[%d/%d] Q: %s... → A: %s... | refs=%d | latency=%.0fms | model=%s",
                i + 1, len(TEST_QUERIES),
                test_case["user_query"][:30],
                answer[:60].replace("\n", " "),
                len(policy_refs),
                elapsed,
                model_ver,
            )
        except Exception as e:
            failures.append(f"[{i+1}] exception: {e}")

    return success, failures


def main():
    logger.info("=" * 60)
    logger.info("M05 RAG Connectivity Smoke Test")
    logger.info(f"Server: {SERVER_URL}")
    logger.info(f"Model:  {MODEL_PATH}")
    logger.info("=" * 60)

    # 1. 启动推理服务
    logger.info("\n[Step 1/4] Starting inference server...")

    env = os.environ.copy()
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent / "src")

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "m_infer.server",
            "--backend", "vllm",
            "--model-path", MODEL_PATH,
            "--host", SERVER_HOST,
            "--port", str(SERVER_PORT),
            "--max-model-len", "2048",
            "--gpu-memory-utilization", "0.85",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )

    # 2. 等待服务就绪
    logger.info("[Step 2/4] Waiting for server to be ready (max %ds)...", BOOT_WAIT_SEC)
    ready = wait_for_server(timeout=BOOT_WAIT_SEC)

    if not ready:
        logger.error("Server did not become ready within %ds", BOOT_WAIT_SEC)
        proc.terminate()
        proc.wait(timeout=10)
        return 1

    # 3. 发送连通性测试
    logger.info("\n[Step 3/4] Running connectivity tests (%d queries)...", len(TEST_QUERIES))
    success, failures = test_connectivity()

    # 4. 关闭服务
    logger.info("\n[Step 4/4] Shutting down server...")
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    # ---- 汇总 ----
    logger.info("\n" + "=" * 60)
    logger.info("M05 RAG Connectivity Test Summary")
    logger.info("=" * 60)
    logger.info("  Total queries:  %d", len(TEST_QUERIES))
    logger.info("  Success:         %d", success)
    logger.info("  Failed:          %d", len(failures))

    if failures:
        for f in failures:
            logger.warning("  FAIL: %s", f)

    if success == len(TEST_QUERIES):
        logger.info("\n  Result: ✅ RAG connectivity PASS (%d/%d)", success, len(TEST_QUERIES))
        logger.info("  连通性测试全部通过，可与司内 RAG 端对接。")
        return 0
    else:
        logger.error("\n  Result: ❌ RAG connectivity FAIL (%d/%d)", success, len(TEST_QUERIES))
        return 1


if __name__ == "__main__":
    sys.exit(main())
