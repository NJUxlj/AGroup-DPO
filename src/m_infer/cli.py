"""推理 CLI 入口 (FR-08)

支持 vLLM / xinference 双后端切换，通过 --config 或 yaml 配置驱动。
带 --host 时启动 HTTP 推理服务（M05 § 7.4）；否则执行批量 prompt 推理。
"""

from __future__ import annotations

import argparse
import json
from utils.logger import CustomLogger
import sys
import time
from pathlib import Path

from .base import InferRequest
from .config import load_infer_config, resolve_infer_settings
from .factory import build_infer_backend

log = CustomLogger.get_logger(__name__)


def _build_backend_kwargs(args: argparse.Namespace) -> tuple[str, str, dict]:
    """合并 config 与 CLI 参数，返回 (backend, model_path, kwargs)。"""
    if args.config:
        cfg = load_infer_config(args.config)
        backend, model_path, kwargs = resolve_infer_settings(
            cfg,
            backend=args.backend,
            model_path=args.model,
        )
    else:
        if not args.model:
            raise ValueError("--model is required when --config is not provided")
        backend = args.backend
        model_path = args.model
        kwargs = {}

    if backend == "vllm":
        kwargs.setdefault("tensor_parallel_size", args.tensor_parallel_size)
        kwargs.setdefault("gpu_memory_utilization", args.gpu_memory_utilization)
        kwargs.setdefault("max_model_len", args.max_model_len)
    elif backend == "xinference":
        kwargs.setdefault("server_endpoint", args.xinference_endpoint)
        if args.xinference_model_uid:
            kwargs["model_uid"] = args.xinference_model_uid

    return backend, model_path, kwargs


def _run_batch_inference(args: argparse.Namespace) -> None:
    """批量 prompt 推理模式。"""
    backend_name, model_path, backend_kwargs = _build_backend_kwargs(args)

    log.info("Building infer backend: %s, model=%s", backend_name, model_path)
    backend = build_infer_backend(backend_name, model_path, **backend_kwargs)

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
        log.error("No prompts provided. Use --prompts or pipe JSON Lines via stdin.")
        sys.exit(1)

    requests = [
        InferRequest(
            prompt=p,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        for p in prompts
    ]

    log.info("Running inference on %d prompts...", len(requests))
    t0 = time.perf_counter()

    if len(requests) == 1:
        responses = [backend.infer(requests[0])]
    else:
        responses = backend.batch_infer(requests)

    total_ms = (time.perf_counter() - t0) * 1000
    log.info("Inference done in %.0fms (%.1f ms/sample)", total_ms, total_ms / len(responses))

    results = []
    for req, resp in zip(requests, responses):
        item = {
            "prompt": req.prompt[:200],
            "output": resp.text[:500],
            "prompt_tokens": resp.prompt_tokens,
            "generated_tokens": resp.generated_tokens,
            "latency_ms": resp.latency_ms,
            "total_latency_ms": resp.total_latency_ms,
        }
        results.append(item)
        print(f"\nQ: {req.prompt[:200]}{'...' if len(req.prompt) > 200 else ''}")
        print(f"A: {resp.text[:500]}{'...' if len(resp.text) > 500 else ''}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        log.info("Results saved to %s", output_path)

    backend.shutdown()


def _run_server(args: argparse.Namespace) -> None:
    """HTTP 推理服务模式（M05 § 7.4）。"""
    from .server import run_server

    backend_name, model_path, backend_kwargs = _build_backend_kwargs(args)
    run_server(
        backend_name=backend_name,
        model_path=model_path,
        model_version=args.model_version,
        host=args.host,
        port=args.port,
        backend_kwargs=backend_kwargs,
    )


def main():
    parser = argparse.ArgumentParser(
        description="M-INFER: 保险问答推理服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 配置驱动 + HTTP 服务（M05 § 7.4）
  copaw-dpo infer --config configs/infer.yaml --host 0.0.0.0 --port 8080

  # vLLM 批量推理
  copaw-dpo infer --backend vllm --model merged_models/qwen2_5_1_5b_insurance_dpo_v1.2 \\
      --prompts "重疾险等待期内确诊是否赔付？"

  # xinference 模式（需先启动 xinference-local 服务）
  copaw-dpo infer --config configs/infer.yaml --prompts "你好"
        """,
    )
    parser.add_argument("--config", "-c", default=None,
                        help="推理配置文件（configs/infer.yaml）")
    parser.add_argument("--backend", choices=["vllm", "xinference"], default=None,
                        help="推理后端（覆盖 config）")
    parser.add_argument("--model", default=None,
                        help="模型路径（HF 格式目录；config 模式下可覆盖 infer.model_path）")
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="推理 prompt 列表（若未提供则从 stdin 读取 JSON Lines）")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="最大生成 token 数 (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output", default=None,
                        help="结果输出 JSON 文件路径")

    # HTTP 服务参数
    parser.add_argument("--host", default=None,
                        help="监听地址；指定时启动 HTTP 服务（默认不启动服务）")
    parser.add_argument("--port", type=int, default=8080, help="监听端口 (default: 8080)")
    parser.add_argument("--model-version", default="",
                        help="模型版本标识（写入 HTTP 响应）")

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

    CustomLogger.configure(level="INFO")

    try:
        if args.host is not None:
            _run_server(args)
        else:
            _run_batch_inference(args)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
