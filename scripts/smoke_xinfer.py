#!/usr/bin/env python3
"""
xinference 单卡推理冒烟测试 - M01 阶段交付物 D-M01-07
scripts/smoke_xinfer.py
适配 xinference >= 2.0 API（使用 Python Client）

流程: 启动 xinference → 注册模型 → 推理 5 条 → 报告结果
"""

import sys
import time
import subprocess
import signal
import os


def main():
    print("[smoke-xinfer] starting ...")

    # 1. 启动 xinference 服务（后台）
    print("[smoke-xinfer] [step1] launching xinference service ...")
    log_dir = os.environ.get("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)

    xinfer_proc = subprocess.Popen(
        ["xinference-local", "-H", "0.0.0.0", "-p", "9997"],
        stdout=open(f"{log_dir}/xinference_server.log", "w"),
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    print(f"[smoke-xinfer] xinference pid={xinfer_proc.pid}")

    # 等待服务就绪
    print("[smoke-xinfer] waiting for xinference ready ...")
    from xinference.client import Client

    for i in range(60):
        try:
            client = Client("http://127.0.0.1:9997")
            client.list_models()
            print(f"[smoke-xinfer] xinference ready (after {i+1}s)")
            break
        except Exception:
            time.sleep(1)
    else:
        print("[smoke-xinfer] xinference failed to start")
        os.killpg(os.getpgid(xinfer_proc.pid), signal.SIGTERM)
        return 1

    try:
        # 2. 注册模型
        print("[smoke-xinfer] [step2] launching model ...")
        model_path = os.environ.get(
            "MODEL_PATH", "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct"
        )
        model_uid = client.launch_model(
            model_name="qwen2.5-instruct",
            model_engine="Transformers",
            model_path=model_path,
            enable_virtual_env=False,
        )
        print(f"[smoke-xinfer] model uid: {model_uid}")
        time.sleep(5)

        # 3. 推理 5 条
        queries = [
            "保险等待期内确诊是否赔付？",
            "百万医疗险的免赔额是怎么计算的？",
            "投保前未告知高血压, 理赔会被拒吗？",
            "什么是重大疾病保险？",
            "保单现金价值是什么？",
        ]

        print("[smoke-xinfer] [step3] running inference ...")
        model = client.get_model(model_uid)
        passed = 0
        for q in queries:
            resp = model.chat([{"role": "user", "content": q}])
            text = resp["choices"][0]["message"]["content"]
            if text.strip():
                passed += 1
                print(f"[smoke-xinfer] [OK] {q}")
                print(f"           {text[:80]}...")
            else:
                print(f"[smoke-xinfer] [FAIL] {q} - empty response")

        print(f"\n[smoke-xinfer] pass={passed}/5")
        if passed == 5:
            print("[smoke-xinfer] status: PASS (5/5)")
            return 0
        else:
            print("[smoke-xinfer] status: FAIL")
            return 1

    finally:
        # 4. 关闭服务
        print("[smoke-xinfer] stopping xinference ...")
        os.killpg(os.getpgid(xinfer_proc.pid), signal.SIGTERM)
        xinfer_proc.wait()


if __name__ == "__main__":
    sys.exit(main())
