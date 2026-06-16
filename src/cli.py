"""
copaw-dpo — AGroup DPO 统一 CLI 入口。

所有 m_* 模块的命令行功能通过统一入口暴露：

    copaw-dpo train    → 训练（LLaMA-Factory / 自定义后端）
    copaw-dpo data     → DPO/SFT 数据集生成
    copaw-dpo infer    → 推理
    copaw-dpo evaluate → 评测
    copaw-dpo merge    → LoRA 合并导出

Usage:
    copaw-dpo train --config configs/train_dpo_qwen2_5_1_5b_insurance.yaml --backend llamafactory
    copaw-dpo data --config configs/data/insurance_dpo_gen.yaml --dry-run
    copaw-dpo infer --model merged_models/... --prompts "你好"
    copaw-dpo evaluate --model merged_models/... --eval-data data/eval/insurance_qa_500.jsonl
    copaw-dpo merge --base Qwen/Qwen2.5-1.5B-Instruct --adapter saves/.../lora --output merged_models/...
"""

from __future__ import annotations

import sys

COMMANDS: dict[str, str] = {
    "train": "m_trainer.cli",
    "data": "m_data.cli",
    "infer": "m_infer.cli",
    "evaluate": "m_eval.cli",
    "merge": "m_merge.cli",
}

HELP_TEXT = """\
copaw-dpo — AGroup DPO 统一命令行工具

用法: copaw-dpo <command> [args...]

可用命令:
  train      训练（LLaMA-Factory 或自定义后端: deepspeed/fsdp/accelerate/megatron）
  data       生成 DPO/SFT 训练数据
  infer      推理（vLLM / xinference）
  evaluate   评测（Accuracy / BLEU-4 / ROUGE-L）
  merge      LoRA 权重合并导出

示例:
  copaw-dpo train --config configs/train_dpo_qwen2_5_1_5b_insurance.yaml
  copaw-dpo train --config configs/my_train.yaml --backend deepspeed
  copaw-dpo data --config configs/data/insurance_dpo_gen.yaml --dry-run
  copaw-dpo infer --model merged_models/... --prompts "保险等待期是什么？"
  copaw-dpo evaluate --model merged_models/... --eval-data data/eval/insurance_qa_500.jsonl
  copaw-dpo merge --base Qwen/Qwen2.5-1.5B-Instruct --adapter saves/.../lora --output merged_models/...

运行 copaw-dpo <command> --help 查看各命令的详细参数。
"""


def main() -> None:
    # 过滤分布式启动器注入的参数（--local_rank, --master_addr 等）
    _filtered = [sys.argv[0]]
    _skip = False
    for a in sys.argv[1:]:
        if _skip:
            _skip = False
            continue
        if a.startswith("--local_rank") or a.startswith("--master_addr") or a.startswith("--master_port") or a.startswith("--node_rank") or a.startswith("--nproc_per_node") or a.startswith("--nnodes"):
            if "=" not in a:
                _skip = True
            continue
        _filtered.append(a)
    sys.argv = _filtered

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(HELP_TEXT)
        sys.exit(0)

    command = sys.argv[1]

    if command not in COMMANDS:
        print(f"未知命令: {command}")
        print(f"可用命令: {', '.join(COMMANDS.keys())}")
        print(f"运行 copaw-dpo --help 查看帮助。")
        sys.exit(1)

    remaining = sys.argv[2:]

    module_name = COMMANDS[command]

    if command == "train":
        # m_trainer.cli.main() 原生支持 args 参数
        from m_trainer.cli import main as train_main
        sys.exit(train_main(remaining))

    elif command == "data":
        # m_data.cli 使用 argparse 直接解析 sys.argv
        sys.argv = ["copaw-dpo"] + remaining
        import m_data.cli as mod
        mod.main()

    elif command == "infer":
        sys.argv = ["copaw-dpo"] + remaining
        import m_infer.cli as mod
        mod.main()

    elif command == "evaluate":
        sys.argv = ["copaw-dpo"] + remaining
        import m_eval.cli as mod
        mod.main()

    elif command == "merge":
        sys.argv = ["copaw-dpo"] + remaining
        import m_merge.cli as mod
        mod.main()

    else:
        # 不应到达此处
        print(f"内部错误: 未知命令 {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
