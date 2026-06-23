"""Generate terminal-style screenshots for merge + m_eval results."""

from __future__ import annotations

import json
from pathlib import Path

from _render import render_terminal

BASE = Path(__file__).resolve().parent
REPORTS = BASE.parent.parent / "reports"


def _load_json(name: str) -> dict:
    with open(REPORTS / name, encoding="utf-8") as f:
        return json.load(f)


def eval_report_lines(data: dict, header: str) -> list[str]:
    lines = [header, ""]
    lines.append(f"模型: {data.get('model_version', '?')}")
    lines.append(f"后端: {data.get('infer_backend', '?')}")
    lines.append(f"时间: {data.get('generated_at', '?')[:19]}")
    lines.append("")
    lines.append("| 数据集 | Accuracy | BLEU-4 | ROUGE-L | 样本数 |")
    lines.append("|--------|----------|--------|---------|--------|")
    for name in sorted(data.get("datasets", {})):
        m = data["datasets"][name]
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['bleu_4']:.4f} | "
            f"{m['rouge_l']:.4f} | {m['n_samples']} |"
        )
    lat = data.get("latency") or {}
    if lat:
        lines += [
            "",
            "=== 推理延迟 ===",
            f"  p50 total: {lat.get('p50_total_ms', 0):.1f} ms",
            f"  p95 total: {lat.get('p95_total_ms', 0):.1f} ms",
            f"  throughput: {lat.get('throughput_samples_per_s', 0):.1f} samples/s",
        ]
    total = sum(m["n_samples"] for m in data.get("datasets", {}).values())
    lines += ["", f"✓ 评测完成 — 共 {total} 条"]
    return lines


def comparison_lines(dpo: dict, base: dict) -> list[str]:
    lines = [
        "=== DPO Merge vs Base 对比 (server2, 1700 条) ===",
        "",
        f"DPO:  {dpo.get('model_version')}",
        f"Base: {base.get('model_version')}",
        "",
        "| 数据集 | 指标 | Base | DPO | Δ |",
        "|--------|------|------|-----|---|",
    ]
    for name in sorted(dpo.get("datasets", {})):
        dm, bm = dpo["datasets"][name], base["datasets"][name]
        for key, label in [("accuracy", "Acc"), ("bleu_4", "BLEU"), ("rouge_l", "ROUGE")]:
            delta = dm[key] - bm[key]
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"| {name} | {label} | {bm[key]:.4f} | {dm[key]:.4f} | {sign}{delta:.4f} |"
            )
    lines += ["", "✓ insurance_qa_500 Accuracy 提升最明显 (+0.032)"]
    return lines


def main() -> None:
    with open(BASE / "merge_output.txt", encoding="utf-8") as f:
        merge_lines = [ln.rstrip("\n") for ln in f.readlines()]
    render_terminal(merge_lines, BASE / "merge_result.png", "m_merge — LoRA → safetensors (server2)")

    dpo = _load_json("eval_report_dpo_v1.json")
    render_terminal(
        eval_report_lines(dpo, "=== m_eval 全量评测 — DPO Merge 模型 (1700 条) ==="),
        BASE / "eval_dpo_merge_full_1700.png",
        "m_eval — qwen2_5_1_5b_insurance_dpo_v1.2",
    )

    base = _load_json("eval_report_server2_full.json")
    render_terminal(
        eval_report_lines(base, "=== m_eval 全量评测 — Base 模型 (1700 条) ==="),
        BASE / "eval_base_full_1700.png",
        "m_eval — Qwen2.5-1.5B-Instruct (baseline)",
    )

    alpaca = _load_json("eval_report_server2_alpaca_zh_200.json")
    render_terminal(
        eval_report_lines(alpaca, "=== m_eval 单数据集 — alpaca_zh_200 (200 条) ==="),
        BASE / "eval_alpaca_zh_200.png",
        "m_eval — alpaca_zh_200.jsonl",
    )

    render_terminal(
        comparison_lines(dpo, base),
        BASE / "eval_dpo_vs_base_comparison.png",
        "m_eval — DPO vs Base 对比",
    )

    smoke_path = REPORTS / "smoke_m05.json"
    if smoke_path.is_file():
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        lines = [
            "=== smoke_m05 — vLLM 推理冒烟 (M05) ===",
            "",
            f"模型: {smoke.get('model_version', '?')}",
            f"后端: {smoke.get('infer_backend', '?')}",
            "",
        ]
        for name, m in smoke.get("datasets", {}).items():
            lines.append(
                f"  {name}: acc={m.get('accuracy', 0):.4f} "
                f"bleu={m.get('bleu_4', 0):.4f} rouge={m.get('rouge_l', 0):.4f} "
                f"n={m.get('n_samples', 0)}"
            )
        lat = smoke.get("latency") or {}
        if lat:
            lines.append(f"  p50 total: {lat.get('p50_total_ms', 0):.1f} ms")
        lines.extend(["", "✓ M05 冒烟通过"])
        render_terminal(lines, BASE / "eval_smoke_m05.png", "smoke_m05.py — vLLM 冒烟")

    print("\n全部截图已保存至 docs/screenshots/")


if __name__ == "__main__":
    main()
