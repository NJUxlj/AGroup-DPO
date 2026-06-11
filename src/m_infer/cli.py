"""推理 CLI 入口 (FR-08)

支持 vLLM / xinference 双后端切换，通过 --backend 或 yaml 配置驱动。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .base import InferRequest
from .factory import build_infer_backend

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="M-INFER: 保险问答推理服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # vLLM 模式
  python -m m_infer.cli --backend vllm --model merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \\
      --prompts "重疾险等待期内确诊是否赔付？" "什么是重大疾病保险？"

  # xinference 模式（需先启动 xinference-local 服务）
  python -m m_infer.cli --backend xinference --model merged_models/... \\
      --xinference-endpoint http://127.0.0.1:9997 --prompts "你好"
        """,
    )
    parser.add_argument("--backend", choices=["vllm", "xinference"], default="vllm",
                        help="推理后端 (default: vllm)")
    parser.add_argument("--model", required=True,
                        help="模型路径（HF 格式目录）")
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="推理 prompt 列表（若未提供则从 stdin 读取 JSON Lines）")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="最大生成 token 数 (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output", default=None,
                        help="结果输出 JSON 文件路径")

    # vLLM 参数
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="vLLM 张量并行数 (default: 1)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="GPU 显存利用率 (default: 0.85)")
    parser.add_argument("--max-model-len", type=int, default=2048)

    # xinference 参数
    parser.add_argument("--xinference-endpoint", default="http://127.0.0.1:9997")
    parser.add_argument("--xinference-model-uid", default=None)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 构建后端
    kwargs = {}
    if args.backend == "vllm":
        kwargs["tensor_parallel_size"] = args.tensor_parallel_size
        kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
        kwargs["max_model_len"] = args.max_model_len
    elif args.backend == "xinference":
        kwargs["server_endpoint"] = args.xinference_endpoint
        if args.xinference_model_uid:
            kwargs["model_uid"] = args.xinference_model_uid

    logger.info("Building infer backend: %s, model=%s", args.backend, args.model)
    backend = build_infer_backend(args.backend, args.model, **kwargs)

    # 获取 prompts
    if args.prompts:
        prompts = args.prompts
    else:
        prompts = []
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                prompts.append(data.get("prompt", data.get("question", line)))
            except json.JSONDecodeError:
                prompts.append(line)

    if not prompts:
        logger.error("No prompts provided. Use --prompts or pipe JSON Lines via stdin.")
        sys.exit(1)

    # 推理
    requests = [
        InferRequest(
            prompt=p,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        for p in prompts
    ]

    logger.info("Running inference on %d prompts...", len(requests))
    t0 = time.perf_counter()

    if len(requests) == 1:
        responses = [backend.infer(requests[0])]
    else:
        responses = backend.batch_infer(requests)

    total_ms = (time.perf_counter() - t0) * 1000
    logger.info("Inference done in %.0fms (%.1f ms/sample)", total_ms, total_ms / len(responses))

    # 输出结果
    results = []
    for req, resp in zip(requests, responses):
        item = {
            "prompt": req.prompt[:100],
            "output": resp.text[:200],
            "prompt_tokens": resp.prompt_tokens,
            "generated_tokens": resp.generated_tokens,
            "latency_ms": resp.latency_ms,
            "total_latency_ms": resp.total_latency_ms,
        }
        results.append(item)
        print(f"\nQ: {req.prompt[:80]}{'...' if len(req.prompt) > 80 else ''}")
        print(f"A: {resp.text[:150]}{'...' if len(resp.text) > 150 else ''}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Results saved to %s", output_path)

    backend.shutdown()


if __name__ == "__main__":
    main()
