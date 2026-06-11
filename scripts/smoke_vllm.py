"""
vLLM 单卡推理冒烟测试 - M01 阶段交付物 D-M01-07
scripts/smoke_vllm.py
"""
import os
import sys
import time
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    # 延迟导入, 避免在非 vLLM 镜像中报错
    from vllm import LLM, SamplingParams

    print(f"[smoke-vllm] model={args.model}, tp={args.tensor_parallel_size}")
    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        enforce_eager=False,
        dtype="bfloat16",
    )
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"[smoke-vllm] model loaded in {load_ms:.0f}ms")

    # 5 条保险业务问题
    prompts = [
        "保险等待期内确诊是否赔付？",
        "百万医疗险的免赔额是怎么计算的？",
        "投保前未告知高血压, 理赔会被拒吗？",
        "什么是重大疾病保险？",
        "保单现金价值是什么？",
    ]

    params = SamplingParams(temperature=0.7, max_tokens=args.max_tokens, top_p=0.9)
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params)
    infer_ms = (time.perf_counter() - t0) * 1000

    # 输出
    print(f"\n[smoke-vllm] inference done in {infer_ms:.0f}ms, samples={len(outputs)}")
    results = []
    for prompt, out in zip(prompts, outputs):
        text = out.outputs[0].text
        results.append({"prompt": prompt, "output": text})
        print(f"\nQ: {prompt}")
        print(f"A: {text}")

    # 统计
    non_empty = sum(1 for r in results if r["output"].strip())
    print(f"\n[smoke-vllm] non-empty outputs: {non_empty}/{len(results)}")

    if non_empty == len(results):
        print("[smoke-vllm] status: PASS (5/5)")
        return 0
    else:
        print("[smoke-vllm] status: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
