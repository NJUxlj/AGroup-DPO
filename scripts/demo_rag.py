"""M05 RAG 对接 Demo (D-M05-19)

展示完整的 RAG 推理链路：
  1. FastAPI 推理服务启动
  2. API 合约展示（RAGRequest / RAGResponse / ContextDoc）
  3. 无上下文直接推理对比
  4. 带保险条款上下文推理（RAG 模式）
  5. policy_refs 策略引用自动提取
  6. 延迟统计与总结
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---- 配置 ----
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "Qwen/Qwen2.5-1.5B-Instruct",
)
DPO_MODEL_PATH_HINT = os.environ.get(
    "DPO_MODEL_PATH_HINT",
    "/root/autodl-tmp/ant-group-dpo/merged_models/qwen2_5_1_5b_insurance_dpo_v1.2",
)
SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8080"))
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
BOOT_WAIT_SEC = int(os.environ.get("BOOT_WAIT_SEC", "90"))

# ---- Demo 查询用例 ----
DEMO_QUERIES = [
    # 场景1：无上下文 — 测试模型通用保险知识
    {
        "title": "场景1: 无上下文 — 通用保险知识",
        "user_query": "什么是保单的现金价值？",
        "context_docs": [],
    },
    # 场景2：单文档上下文 — RAG 核心用法
    {
        "title": "场景2: 单文档 — 百万医疗险免赔额",
        "user_query": "百万医疗险的免赔额怎么计算？社保报销能不能抵扣？",
        "context_docs": [
            {
                "id": "policy_deductible_2024",
                "text": (
                    "百万医疗险条款第三章第十二条：年度免赔额为人民币1万元整。"
                    "被保险人在社保或其他商业保险已获得的医疗费用补偿，可用于抵扣年度免赔额。"
                    "超出免赔额部分的合理医疗费用，保险人按照100%的比例给付保险金。"
                ),
            },
        ],
    },
    # 场景3：多文档上下文 — 多条条款交叉引用
    {
        "title": "场景3: 多文档 — 等待期 + 如实告知",
        "user_query": "我投保时忘了告知有高血压，现在等待期内查出心脏病，能赔吗？",
        "context_docs": [
            {
                "id": "policy_waiting_period",
                "text": (
                    "被保险人在等待期（自合同生效日起90日）内发生保险事故的，"
                    "保险人不承担给付保险金责任，但应退还已收取的保险费。"
                ),
            },
            {
                "id": "policy_disclosure",
                "text": (
                    "投保人故意或因重大过失未履行如实告知义务，足以影响保险人决定是否同意承保"
                    "或者提高保险费率的，保险人有权解除合同。前款规定的合同解除权，自保险人知道"
                    "有解除事由之日起，超过三十日不行使而消灭。自合同成立之日起超过二年的，"
                    "保险人不得解除合同；发生保险事故的，保险人应当承担赔偿或者给付保险金的责任。"
                ),
            },
        ],
    },
    # 场景4：理赔流程查询
    {
        "title": "场景4: 单文档 — 住院理赔流程",
        "user_query": "住院之后理赔需要准备哪些材料，流程是什么？",
        "context_docs": [
            {
                "id": "policy_claim_process",
                "text": (
                    "理赔申请流程：\n"
                    "1. 出险后48小时内拨打客服热线95519报案；\n"
                    "2. 准备理赔材料：身份证、保单、诊断证明、住院病历、医疗费用发票原件、费用明细清单；\n"
                    "3. 将材料提交至保险公司理赔部或通过APP上传；\n"
                    "4. 保险公司在收到完整材料后30日内作出核定；\n"
                    "5. 核定通过后10日内支付保险金。"
                ),
            },
        ],
    },
    # 场景5：既往症
    {
        "title": "场景5: 无上下文 — 既往症常识",
        "user_query": "投保前有糖尿病，买了健康险后因糖尿病住院能赔吗？",
        "context_docs": [],
    },
]


def print_banner(text: str, char: str = "=") -> None:
    width = 70
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}\n")


def print_section(text: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  >>> {text}")
    print(f"{'─' * 60}")


def print_api_contract() -> None:
    """展示 RAG API 合约。"""
    print_banner("📋 RAG API 合约", "=")

    print("### 请求 POST /v1/insurance/qa")
    print("```json")
    example_request = {
        "user_query": "百万医疗险的免赔额怎么计算？",
        "context_docs": [
            {
                "id": "policy_deductible_2024",
                "text": "年度免赔额为人民币1万元整...",
            }
        ],
        "max_new_tokens": 256,
        "temperature": 0.3,
    }
    print(json.dumps(example_request, indent=2, ensure_ascii=False))
    print("```")

    print("\n### 响应")
    print("```json")
    example_response = {
        "answer": "百万医疗险的年度免赔额为1万元...",
        "policy_refs": ["policy_deductible_2024"],
        "first_token_latency_ms": 45.2,
        "total_latency_ms": 520.3,
        "model_version": "qwen2_5_1_5b_insurance_dpo_v1.2",
        "request_id": "550e8400-e29b-41d4-a716-446655440000",
    }
    print(json.dumps(example_response, indent=2, ensure_ascii=False))
    print("```")

    print("\n### Pydantic 模型")
    print("  ContextDoc   { id: str, text: str }")
    print("  RAGRequest   { user_query, context_docs[], max_new_tokens, temperature }")
    print("  RAGResponse  { answer, policy_refs[], first_token_latency_ms, total_latency_ms, model_version, request_id }")


def wait_for_server(timeout: int = BOOT_WAIT_SEC) -> bool:
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5)
            data = json.loads(resp.read())
            if data.get("status") == "ok":
                logger.info("✅ Server ready: backend=%s, version=%s",
                            data.get("backend", "?"), data.get("model_version", "?"))
                return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        time.sleep(2)
    return False


def run_demo_queries() -> dict:
    """依次执行 Demo 查询并返回结果。"""
    import requests

    results = []
    total_time = 0.0
    success_count = 0

    for i, test_case in enumerate(DEMO_QUERIES):
        print_section(f"Demo {i + 1}/{len(DEMO_QUERIES)}: {test_case['title']}")

        payload = {
            "user_query": test_case["user_query"],
            "context_docs": test_case.get("context_docs", []),
            "max_new_tokens": 256,
            "temperature": 0.3,
        }

        print(f"  📝 Query: {test_case['user_query']}")
        if test_case["context_docs"]:
            for doc in test_case["context_docs"]:
                print(f"  📄 Context [{doc['id']}]: {doc['text'][:80]}...")
        else:
            print(f"  📄 Context: (none — direct inference)")

        try:
            t0 = time.perf_counter()
            resp = requests.post(
                f"{SERVER_URL}/v1/insurance/qa",
                json=payload,
                timeout=120,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            total_time += elapsed

            if resp.status_code != 200:
                print(f"  ❌ HTTP {resp.status_code}: {resp.text[:100]}")
                results.append({"title": test_case["title"], "success": False, "error": f"HTTP {resp.status_code}"})
                continue

            data = resp.json()
            answer = data.get("answer", "")
            policy_refs = data.get("policy_refs", [])
            model_ver = data.get("model_version", "")
            ttft = data.get("first_token_latency_ms", 0)

            success_count += 1
            print(f"  🤖 Answer: {answer[:200]}")
            if len(answer) > 200:
                print(f"           ... ({len(answer)} chars total)")
            if policy_refs:
                print(f"  🔗 Policy Refs: {policy_refs}")
            else:
                print(f"  🔗 Policy Refs: (none extracted — no context or no match)")
            print(f"  ⏱️  Latency: first_token={ttft:.0f}ms, total={elapsed:.0f}ms")
            print(f"  🏷️  Model: {model_ver}")

            results.append({
                "title": test_case["title"],
                "success": True,
                "answer": answer,
                "policy_refs": policy_refs,
                "first_token_ms": ttft,
                "total_ms": elapsed,
                "model_version": model_ver,
            })

        except Exception as e:
            print(f"  ❌ Exception: {e}")
            results.append({"title": test_case["title"], "success": False, "error": str(e)})

    return {
        "results": results,
        "success_count": success_count,
        "total_count": len(DEMO_QUERIES),
        "total_time_ms": total_time,
        "avg_latency_ms": total_time / max(success_count, 1),
    }


def print_summary(demo_result: dict) -> None:
    print_banner("📊 Demo 总结", "=")

    s = demo_result
    print(f"  总查询数:       {s['total_count']}")
    print(f"  成功:           {s['success_count']}")
    print(f"  失败:           {s['total_count'] - s['success_count']}")
    print(f"  总耗时:         {s['total_time_ms']:.0f}ms")
    print(f"  平均延迟:       {s['avg_latency_ms']:.0f}ms")

    print(f"\n  {'─' * 55}")
    print(f"  {'场景':<30s} {'Policy Refs':<15s} {'延迟':>10s}")
    print(f"  {'─' * 55}")
    for r in s["results"]:
        if r["success"]:
            refs_str = ", ".join(r["policy_refs"]) if r["policy_refs"] else "(none)"
            print(f"  {r['title'][:28]:<30s} {refs_str[:13]:<15s} {r['total_ms']:>7.0f}ms")
        else:
            print(f"  {r['title'][:28]:<30s} {'❌ FAILED':<15s} {'—':>10s}")

    print(f"\n{'─' * 70}")
    if s["success_count"] == s["total_count"]:
        print("  ✅ Demo 全部通过！RAG 推理链路工作正常。")
    else:
        print(f"  ⚠️  {s['total_count'] - s['success_count']} 条失败，请检查服务日志。")

    print(f"\n  💡 对接提示：司内 RAG 端只需发送 POST {SERVER_URL}/v1/insurance/qa")
    print(f"     Content-Type: application/json，请求体格式见上方 API 合约。")
    print(f"{'─' * 70}\n")


def main():
    print_banner("M05 RAG 对接 Demo (D-M05-19)", "★")

    # ---- Step 0: API 合约 ----
    print_api_contract()

    # ---- Step 1: 确定模型路径 ----
    actual_model = MODEL_PATH
    if os.path.isdir(DPO_MODEL_PATH_HINT):
        actual_model = DPO_MODEL_PATH_HINT
        logger.info("Found DPO merged model: %s", DPO_MODEL_PATH_HINT)
    else:
        logger.info("DPO merged model not found, using baseline: %s", MODEL_PATH)
        logger.info("(Set MODEL_PATH env to use a different model)")

    print_banner("🚀 启动推理服务", "=")
    print(f"  Model:  {actual_model}")
    print(f"  Server: {SERVER_URL}")
    print(f"  Backend: vLLM (V1 engine)\n")

    # ---- Step 2: 启动服务 ----
    env = os.environ.copy()
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent / "src")

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "m_infer.server",
            "--backend", "vllm",
            "--model-path", actual_model,
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

    # ---- Step 3: 等待就绪 ----
    print("  ⏳ 等待服务就绪（vLLM 加载/编译需要 ~30-60s）...\n")
    ready = wait_for_server(timeout=BOOT_WAIT_SEC)

    if not ready:
        logger.error("服务启动超时")
        print("\n  ❌ 服务未能在 %ds 内就绪，请检查:" % BOOT_WAIT_SEC)
        print("     1. GPU 是否可用 (nvidia-smi)")
        print("     2. 模型路径是否正确")
        print("     3. vLLM 环境是否正常")
        proc.terminate()
        proc.wait(timeout=10)
        return 1

    # ---- Step 4: 执行 Demo 查询 ----
    print_banner("🔄 执行 Demo 查询", "=")
    demo_result = run_demo_queries()

    # ---- Step 5: 关闭服务 ----
    logger.info("Shutting down server...")
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    # ---- Step 6: 总结 ----
    print_summary(demo_result)

    return 0 if demo_result["success_count"] == demo_result["total_count"] else 1


if __name__ == "__main__":
    sys.exit(main())
