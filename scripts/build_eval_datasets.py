#!/usr/bin/env python3
"""构建 M05 三类评测数据集 (D-M05-13)。

数据来源（推荐）：
  medical_qa_1000  → CMB-Exam (FreedomIntelligence/CMB) + CMB-Clin 开放题
  insurance_qa_500 → 司内 FAQ/工单 holdout + 合成数据尾部 holdout
  alpaca_zh_200    → ChineseAlpacaEval + GPT-4 参考答案（防退化）

用法:
    PYTHONPATH=src python scripts/build_eval_datasets.py
    PYTHONPATH=src python scripts/build_eval_datasets.py --only medical insurance
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
SOURCES_DIR = EVAL_DIR / "sources"
INSURANCE_RAW = ROOT / "data" / "insurance" / "raw"

MEDICAL_TARGET = 1000
INSURANCE_TARGET = 500
ALPACA_TARGET = 200

SEED = 42


def _download(url: str, dest: Path, timeout: int = 300) -> bool:
    """下载文件到 dest，成功返回 True（优先 curl，兼容 SSL 问题）。"""
    import subprocess

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        try:
            json.load(open(dest, encoding="utf-8"))
            print(f"  [cache hit] {dest.name} ({dest.stat().st_size // 1024} KB)")
            return True
        except (json.JSONDecodeError, OSError):
            print(f"  [warn] corrupt cache, re-download: {dest.name}")

    print(f"  [download] {url}")
    try:
        subprocess.run(
            ["curl", "-kfsSL", "--connect-timeout", "30", "--max-time", str(timeout), "-o", str(dest), url],
            check=True,
            capture_output=True,
        )
        print(f"  [saved] {dest.name} ({dest.stat().st_size // 1024} KB)")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  [warn] curl download failed: {exc}")
        return False


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  [write] {path} ({len(rows)} samples)")


def _parse_options(raw) -> dict[str, str]:
    """解析 CMB option 字段（可能是 dict 或 str）。"""
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = eval(raw)  # noqa: S307 — CMB 源数据为 Python dict 字面量字符串
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except Exception:
            pass
    return {}


def _format_mc_question(item: dict) -> str:
    """将 CMB / MedQA 选择题格式化为带选项的 question 字符串。"""
    question = item.get("question", "").strip()
    options = _parse_options(item.get("option"))
    if not options:
        for key in ("A", "B", "C", "D", "E"):
            if key in item and item[key]:
                options[key] = item[key]
    if options:
        opts = " ".join(f"{k}. {v}" for k, v in sorted(options.items()))
        return f"{question} {opts}".strip()
    return question


def _load_cmb_exam_questions() -> list[dict]:
    """从 HF mirror / 本地缓存加载 CMB-Exam test 题目。"""
    merge_file = SOURCES_DIR / "cmb_test_questions.json"
    if not merge_file.exists():
        _download(
            "https://hf-mirror.com/datasets/FreedomIntelligence/CMB/resolve/main/"
            "CMB-Exam/CMB-test/CMB-test-choice-question-merge.json",
            merge_file,
            timeout=300,
        )
    if merge_file.exists():
        raw = json.load(open(merge_file, encoding="utf-8"))
        if isinstance(raw, list) and raw:
            print(f"  [local] CMB-Exam test questions: {len(raw)}")
            return raw

    # fallback: HuggingFace datasets
    try:
        import os
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from datasets import load_dataset

        ds = load_dataset("FreedomIntelligence/CMB", "CMB-Exam", split="test")
        print(f"  [hf] CMB-Exam test: {len(ds)} rows")
        return [dict(row) for row in ds]
    except Exception as exc:
        print(f"  [warn] HF load failed: {exc}")
        return []


def _parse_qa_pairs(raw) -> list[dict]:
    """解析 CMB-Clin 的 QA_pairs 字段。"""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw.strip():
        import ast
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return parsed
        except (SyntaxError, ValueError):
            pass
    return []


def _load_cmb_clin() -> list[dict]:
    """加载 CMB-Clin 临床案例 QA（hf-mirror 本地缓存）。"""
    clin_file = SOURCES_DIR / "cmb_clin_qa.json"
    if not clin_file.exists():
        _download(
            "https://hf-mirror.com/datasets/FreedomIntelligence/CMB/resolve/main/"
            "CMB-Clin/CMB-Clin-qa.json",
            clin_file,
            timeout=120,
        )
    if not clin_file.exists():
        print("  [warn] CMB-Clin file unavailable")
        return []

    cases = json.load(open(clin_file, encoding="utf-8"))
    rows: list[dict] = []
    for case in cases:
        title = case.get("title", "clinical")
        desc = (case.get("description") or "").strip()
        # 病例背景拼入 question，便于模型在无 RAG 上下文时作答
        context = f"【{title}】\n{desc[:1200]}".strip() if desc else f"【{title}】"

        for qa in _parse_qa_pairs(case.get("QA_pairs")):
            q = qa.get("question", "").strip()
            a = qa.get("answer", "").strip()
            if not q or not a:
                continue
            full_q = f"{context}\n\n问题：{q}" if context else q

            choice_m = re.search(r"答案[：:]\s*([A-E])\b", a[:30])
            if choice_m and len(a) < 120:
                rows.append({
                    "question": full_q,
                    "reference_answer": choice_m.group(1),
                    "answer_type": "choice",
                    "judge_required": False,
                    "category": title,
                    "source": "CMB-Clin",
                })
            else:
                rows.append({
                    "question": full_q,
                    "reference_answer": a,
                    "answer_type": "open",
                    "judge_required": True,
                    "category": title,
                    "source": "CMB-Clin",
                })

    print(f"  [local] CMB-Clin QA pairs: {len(rows)}")
    return rows


def _load_medqa_zh() -> list[dict]:
    """MedQA 中文 4 选项 test 集（备用）。"""
    cache = SOURCES_DIR / "medqa_zh_test.jsonl"
    if not cache.exists():
        urls = [
            "https://hf-mirror.com/datasets/bigbio/med_qa/resolve/main/data_clean/questions/Mainland/4_options/test.jsonl",
            "https://huggingface.co/datasets/bigbio/med_qa/resolve/main/data_clean/questions/Mainland/4_options/test.jsonl",
        ]
        for url in urls:
            if _download(url, cache, timeout=300):
                break
    if not cache.exists():
        return []

    rows = []
    for line in open(cache, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        answer_idx = item.get("answer_idx", item.get("answer", ""))
        options = item.get("options", {})
        rows.append({
            "question": _format_mc_question({"question": item.get("question", ""), "option": options}),
            "reference_answer": str(answer_idx).strip().upper(),
            "answer_type": "choice",
            "judge_required": False,
            "category": item.get("meta_info", "medqa"),
            "source": "MedQA-zh",
        })
    print(f"  [medqa] loaded {len(rows)} samples")
    return rows


def build_medical_qa_1000() -> list[dict]:
    """构建 medical_qa_1000.jsonl。"""
    print("\n==> medical_qa_1000")
    rng = random.Random(SEED)

    # CMB-Exam 选择题 + CMB-Clin 开放题（优先保留 Clin 全部 ~208 条）
    cmb_questions = _load_cmb_exam_questions()
    answer_map: dict = {}
    ans_file = SOURCES_DIR / "cmb_test_choice_answer.json"
    if not ans_file.exists():
        _download(
            "https://raw.githubusercontent.com/FreedomIntelligence/CMB/main/data/CMB-test-choice-answer.json",
            ans_file,
        )
    if ans_file.exists():
        for item in json.load(open(ans_file, encoding="utf-8")):
            answer_map[item["id"]] = item["answer"]

    # CMB-Exam 选择题 + CMB-Clin 开放题（优先保留 Clin 全部 ~208 条）
    exam_rows: list[dict] = []
    for item in cmb_questions:
        qid = item.get("id")
        ref = answer_map.get(qid) or answer_map.get(int(qid)) if qid is not None else ""
        ref = ref or item.get("answer", "")
        if not ref:
            continue
        # 只取单选/标准选择题
        if len(str(ref)) > 5:
            continue
        qtext = _format_mc_question(item)
        if not qtext:
            continue
        exam_rows.append({
            "question": qtext,
            "reference_answer": str(ref).strip().upper(),
            "answer_type": "choice",
            "judge_required": False,
            "category": item.get("exam_subject", item.get("exam_type", "medical")),
            "source": "CMB-Exam",
        })

    clin_rows = _load_cmb_clin()
    rng.shuffle(exam_rows)

    clin_cap = min(len(clin_rows), 250)  # 保留最多 250 道 Clin 开放题
    clin_selected = clin_rows[:clin_cap]
    exam_needed = MEDICAL_TARGET - len(clin_selected)
    rows = clin_selected + exam_rows[:max(exam_needed, 0)]

    if len(rows) < MEDICAL_TARGET:
        medqa = _load_medqa_zh()
        rng.shuffle(medqa)
        rows.extend(medqa)

    # 去重 + 采样（用完整 question 作 key，避免 Clin 病例前缀相同被误删）
    seen: set[str] = set()
    unique: list[dict] = []
    for r in rows:
        key = r["question"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    rng.shuffle(unique)
    selected = unique[:MEDICAL_TARGET]

    if len(selected) < MEDICAL_TARGET:
        medqa = _load_medqa_zh()
        rng.shuffle(medqa)
        for r in medqa:
            if len(selected) >= MEDICAL_TARGET:
                break
            if r["question"] not in seen:
                seen.add(r["question"])
                selected.append(r)

    # 添加 id
    out = []
    for i, r in enumerate(selected, 1):
        out.append({
            "id": f"med_qa_{i:04d}",
            "category": r.get("category", "medical"),
            "question": r["question"],
            "reference_answer": r["reference_answer"],
            "answer_type": r.get("answer_type", "open"),
            "judge_required": r.get("judge_required", False),
            "source": r.get("source", "unknown"),
        })
    print(f"  [result] {len(out)} samples (target {MEDICAL_TARGET})")
    return out


def build_insurance_qa_500() -> list[dict]:
    """构建 insurance_qa_500.jsonl（训练 holdout）。"""
    print("\n==> insurance_qa_500")
    rng = random.Random(SEED)
    rows: list[dict] = []

    # 1) 人工审核 FAQ + 真实工单（高质量 holdout）
    faq_v1 = json.load(open(INSURANCE_RAW / "faq" / "faq_v1.json", encoding="utf-8"))
    for item in faq_v1:
        rows.append({
            "category": item.get("category", "insurance"),
            "question": item["question"],
            "reference_answer": item["answer"],
            "answer_type": "open",
            "judge_required": True,
            "source": "faq_v1",
        })

    tickets_v1 = json.load(open(INSURANCE_RAW / "tickets" / "tickets_v1.json", encoding="utf-8"))
    for item in tickets_v1:
        rows.append({
            "category": item.get("category", "compliance_qa"),
            "question": item["user_question"],
            "reference_answer": item["agent_answer"],
            "answer_type": "open",
            "judge_required": True,
            "source": "tickets_v1",
        })

    # 2) 合成数据尾部 holdout（index 4500-4999，与训练用 seed=42 的前 4500 条隔离）
    faq_syn = json.load(open(INSURANCE_RAW / "faq" / "faq_synthetic.json", encoding="utf-8"))
    holdout_faq = faq_syn[4500:5000] if len(faq_syn) >= 5000 else faq_syn[-500:]
    for item in holdout_faq:
        rows.append({
            "category": item.get("category", "insurance"),
            "question": item["question"],
            "reference_answer": item["answer"],
            "answer_type": "open",
            "judge_required": True,
            "source": "faq_synthetic_holdout",
        })

    # 3) 若不足 500，从 tickets_synthetic 尾部补充
    if len(rows) < INSURANCE_TARGET:
        tickets_syn = json.load(open(INSURANCE_RAW / "tickets" / "tickets_synthetic.json", encoding="utf-8"))
        need = INSURANCE_TARGET - len(rows)
        extra = tickets_syn[-need:]
        for item in extra:
            rows.append({
                "category": item.get("category", "compliance_qa"),
                "question": item["user_question"],
                "reference_answer": item["agent_answer"],
                "answer_type": "open",
                "judge_required": True,
                "source": "tickets_synthetic_holdout",
            })

    # 去重
    seen: set[str] = set()
    unique: list[dict] = []
    for r in rows:
        key = r["question"][:80]
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    selected = unique[:INSURANCE_TARGET]
    out = []
    for i, r in enumerate(selected, 1):
        out.append({
            "id": f"ins_qa_{i:04d}",
            **r,
        })
    print(f"  [result] {len(out)} samples (target {INSURANCE_TARGET})")
    return out


def build_alpaca_zh_200() -> list[dict]:
    """构建 alpaca_zh_200.jsonl（ChineseAlpacaEval + GPT-4 参考）。"""
    print("\n==> alpaca_zh_200")
    rng = random.Random(SEED)

    instr_file = SOURCES_DIR / "chinese_alpaca_eval.jsonl"
    resp_file = SOURCES_DIR / "chinese_alpaca_gpt4.jsonl"
    if not instr_file.exists():
        _download(
            "https://raw.githubusercontent.com/CrossmodalGroup/ChineseAlpacaEval/main/data/chinese_alpaca_eval.jsonl",
            instr_file,
        )
    if not resp_file.exists():
        _download(
            "https://raw.githubusercontent.com/CrossmodalGroup/ChineseAlpacaEval/main/model_outputs/gpt-4-0613.jsonl",
            resp_file,
        )

    instructions: list[dict] = []
    for line in open(instr_file, encoding="utf-8"):
        line = line.strip()
        if line:
            instructions.append(json.loads(line))

    responses: dict[str, str] = {}
    for line in open(resp_file, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        responses[item["instruction"]] = item.get("response", "")

    pairs: list[dict] = []
    for item in instructions:
        instr_zh = item.get("instruction_zh", item.get("instruction", "")).strip()
        ref = responses.get(instr_zh, "")
        if not instr_zh or not ref:
            continue
        pairs.append({
            "category": item.get("dataset", "general"),
            "question": instr_zh,
            "reference_answer": ref,
            "answer_type": "open",
            "judge_required": True,
            "source": "ChineseAlpacaEval+gpt4",
        })

    rng.shuffle(pairs)
    selected = pairs[:ALPACA_TARGET]
    out = []
    for i, r in enumerate(selected, 1):
        out.append({
            "id": f"alpaca_{i:04d}",
            **r,
        })
    print(f"  [result] {len(out)} samples (target {ALPACA_TARGET})")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build M05 evaluation datasets")
    parser.add_argument(
        "--only", nargs="+",
        choices=["medical", "insurance", "alpaca"],
        default=["medical", "insurance", "alpaca"],
    )
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    if "medical" in args.only:
        rows = build_medical_qa_1000()
        if len(rows) < 100:
            print("  [error] medical dataset too small; check network / CMB sources", file=sys.stderr)
            return 1
        _write_jsonl(EVAL_DIR / "medical_qa_1000.jsonl", rows)

    if "insurance" in args.only:
        rows = build_insurance_qa_500()
        _write_jsonl(EVAL_DIR / "insurance_qa_500.jsonl", rows)

    if "alpaca" in args.only:
        rows = build_alpaca_zh_200()
        if len(rows) < 50:
            print("  [error] alpaca dataset too small; check sources/", file=sys.stderr)
            return 1
        _write_jsonl(EVAL_DIR / "alpaca_zh_200.jsonl", rows)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
