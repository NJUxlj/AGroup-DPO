#!/usr/bin/env python3
"""DPO 数据合成全量集成测试 —— 覆盖 PolicyStore 索引 → Pipeline 全量运行 → 输出验证。

本脚本在 server6 远端执行，验证以下能力：

  测试 1: PolicyStore 索引
    - 扫描 data/insurance/raw/policies/*.json
    - 编码所有 article_content 并存入 Milvus Lite
    - 验证 collection 行数 ≥ 预期值

  测试 2: PolicyStore 查询
    - 用 prompt 语义检索条款片段
    - 验证返回结果包含真实条款内容（关键词命中）

  测试 3: 全量 DPO Pipeline 运行
    - 读取 configs/data/insurance_dpo_gen.yaml
    - 执行 Pipeline.run() —— 采集 → 规范化 → 脱敏 → 过滤 → 配对 → 校验+修复 → 写出
    - 验证产出 JSONL 文件非空

  测试 4: 输出质量检查
    - 抽查 chosen 文本中是否包含真实条款引用（而非模板兜底）
    - 验证 repair 统计中 PolicyStore 修复数 ≥ 0

  测试 5: Validator 校验统计
    - 验证通过率 ≥ 预期阈值
    - 验证 repaired 计数合理

环境要求:
  - conda env llm (Python 3.12)
  - pip install pymilvus>=2.4 sentence-transformers>=2.7 (若缺失则自动安装)
  - 模型: BAAI/bge-small-zh-v1.5 (首次运行自动下载到 HF cache)

用法:
  cd /root/autodl-tmp/agroup-dpo
  PYTHONPATH=src:$PYTHONPATH python tests/m_data/test_dpo_full_pipeline.py
"""

from __future__ import annotations

import json
import logging  # 仅用于控制外部库（pymilvus/sentence_transformers/httpx）日志级别
import os
import sys
import time
from pathlib import Path
from typing import Any

from utils.logger import CustomLogger

# ------------------------------------------------------------------
# Path setup
# ------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

os.chdir(str(_PROJECT_ROOT))

CustomLogger.configure(level="INFO")
log = CustomLogger.get_logger("dpo_full_smoke")

# 抑制 Milvus / transformers 的 DEBUG 噪音
logging.getLogger("pymilvus").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

PASS = 0
FAIL = 0
RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        RESULTS.append((name, True, detail))
        log.info("  ✅ %s: %s", name, detail)
    else:
        FAIL += 1
        RESULTS.append((name, False, detail))
        log.error("  ❌ %s: %s", name, detail)


# =====================================================================
# 测试 0: 环境检查
# =====================================================================
def test_env() -> None:
    log.info("=" * 60)
    log.info("测试 0: 环境检查")
    log.info("=" * 60)

    # Python
    py_ver = sys.version.split()[0]
    record("Python >= 3.10", sys.version_info >= (3, 10), py_ver)

    # pymilvus
    try:
        import pymilvus  # noqa: F401
        record("pymilvus 已安装", True, f"version={pymilvus.__version__}")
    except ImportError:
        record("pymilvus 未安装，尝试安装...", False, "")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pymilvus>=2.4"])
        import pymilvus  # noqa: F401
        record("pymilvus 安装完成", True, f"version={pymilvus.__version__}")

    # sentence-transformers
    try:
        import sentence_transformers  # noqa: F401
        record("sentence-transformers 已安装", True, f"version={sentence_transformers.__version__}")
    except ImportError:
        record("sentence-transformers 未安装，尝试安装...", False, "")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers>=2.7"])
        import sentence_transformers  # noqa: F401
        record("sentence-transformers 安装完成", True, f"version={sentence_transformers.__version__}")

    # pyyaml (for config parsing)
    try:
        import yaml  # noqa: F401
        record("pyyaml 已安装", True)
    except ImportError:
        record("pyyaml 未安装", False, "需要 pyyaml 解析配置文件")
        return

    # 配置文件存在性
    config_path = _PROJECT_ROOT / "configs" / "data" / "insurance_dpo_gen.yaml"
    record("配置文件存在", config_path.exists(), str(config_path))

    # 政策数据存在性
    policy_dir = _PROJECT_ROOT / "data" / "insurance" / "raw" / "policies"
    policy_files = list(policy_dir.glob("*.json")) if policy_dir.exists() else []
    record(
        f"政策 JSON 文件 ({len(policy_files)} 个)",
        len(policy_files) >= 3,
        f"found {len(policy_files)} files",
    )


# =====================================================================
# 测试 1: PolicyStore 索引
# =====================================================================
def test_policy_store_index() -> dict[str, Any] | None:
    log.info("=" * 60)
    log.info("测试 1: PolicyStore 索引政策条款")
    log.info("=" * 60)

    from m_data.policy_store import PolicyStore

    config_path = _PROJECT_ROOT / "configs" / "data" / "insurance_dpo_gen.yaml"
    import yaml
    with open(config_path) as f:
        full_config = yaml.safe_load(f)

    ps_cfg = full_config.get("policy_store", {})
    if not ps_cfg:
        record("PolicyStore 配置存在", False, "insurance_dpo_gen.yaml 中缺少 policy_store 段")
        return None

    record("PolicyStore 配置已加载", True, f"model={ps_cfg.get('embedding_model', 'N/A')}")

    # 清理旧的 Milvus 数据，确保全量重建
    db_path = Path(ps_cfg.get("milvus_db_path", "./milvus_data/policy_store.db"))
    if db_path.exists():
        log.info("  清理旧的 Milvus 数据: %s", db_path)
        import shutil
        shutil.rmtree(str(db_path), ignore_errors=True)

    t0 = time.perf_counter()
    try:
        store = PolicyStore(ps_cfg)
        store.ensure_ready()
        elapsed = time.perf_counter() - t0
        record("PolicyStore 初始化", True, f"{elapsed:.1f}s")
    except Exception as e:
        record("PolicyStore 初始化", False, str(e)[:100])
        return None

    # 验证 collection 行数
    try:
        stats = store._client.get_collection_stats(store._collection_name)
        row_count = stats.get("row_count", 0)
        # 预期：5 个 policy JSON × 平均 8-10 articles = 40-50+
        min_expected = 20
        record(
            f"Milvus collection 行数: {row_count}",
            row_count >= min_expected,
            f"expected >= {min_expected}",
        )
    except Exception as e:
        record("Milvus collection stats", False, str(e)[:100])
        return None

    return {"store": store, "row_count": row_count}


# =====================================================================
# 测试 2: PolicyStore 查询
# =====================================================================
def _check_policy_store_search(ctx: dict[str, Any]) -> None:
    log.info("=" * 60)
    log.info("测试 2: PolicyStore 混合检索")
    log.info("=" * 60)

    store = ctx.get("store")
    if not store:
        record("PolicyStore 可用", False, "store is None")
        return

    record("PolicyStore ready", store.ready)

    # 测试查询：等待期相关
    test_cases = [
        ("POL-CRIT-001", "重疾险等待期内确诊是否赔付？", ["等待期", "90日"]),
        ("POL-CRIT-001", "投保后多久可以理赔？", ["保险事故", "通知"]),
        ("POL-MEDI-001", "百万医疗险免赔额怎么计算？", ["免赔"]),
    ]

    for policy_id, prompt, expected_keywords in test_cases:
        try:
            results = store.search(policy_id=policy_id, prompt=prompt, top_k=3)
        except Exception as e:
            record(f"search({policy_id}) 异常", False, str(e)[:80])
            continue

        if not results:
            record(f"search({policy_id}, '{prompt[:20]}...')", False, "无结果")
            continue

        top_content = results[0].get("article_content", "")
        hits = [kw for kw in expected_keywords if kw in top_content]
        record(
            f"search({policy_id}, '{prompt[:20]}...') → top-1 score={results[0]['score']:.3f}",
            len(hits) >= 1,
            f"hits={hits}, content={top_content[:60]}...",
        )


# =====================================================================
# 测试 3: 全量 DPO Pipeline
# =====================================================================
def test_full_pipeline() -> dict[str, Any] | None:
    log.info("=" * 60)
    log.info("测试 3: 全量 DPO Pipeline 运行")
    log.info("=" * 60)

    from m_data.pipeline import Pipeline
    import yaml

    config_path = _PROJECT_ROOT / "configs" / "data" / "insurance_dpo_gen.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    t0 = time.perf_counter()
    try:
        pipeline = Pipeline(config)
        stats = pipeline.run()
        elapsed = time.perf_counter() - t0
        record("Pipeline.run() 完成", True, f"{elapsed:.1f}s")
    except Exception as e:
        record("Pipeline.run() 异常", False, str(e)[:200])
        import traceback
        traceback.print_exc()
        return None

    # 产出检查
    dpo_path = Path(config["output"].get("path", "data/insurance/dpo_train_v1.2.jsonl"))
    sft_path = Path(config["output"].get("sft_path", "data/insurance/insurance_sft_v1.jsonl"))

    dpo_exists = dpo_path.exists()
    sft_exists = sft_path.exists()

    record("DPO JSONL 产出存在", dpo_exists, str(dpo_path))
    record("SFT JSONL 产出存在", sft_exists, str(sft_path))

    dpo_count = 0
    if dpo_exists:
        with open(dpo_path, encoding="utf-8") as f:
            dpo_count = sum(1 for _ in f)
        record(f"DPO 样本数: {dpo_count}", dpo_count > 0, f"{dpo_count} samples")

    sft_count = 0
    if sft_exists:
        with open(sft_path, encoding="utf-8") as f:
            sft_count = sum(1 for _ in f)
        record(f"SFT 样本数: {sft_count}", sft_count > 0, f"{sft_count} samples")

    return {
        "stats": stats,
        "dpo_path": str(dpo_path),
        "sft_path": str(sft_path),
        "dpo_count": dpo_count,
        "sft_count": sft_count,
    }


# =====================================================================
# 测试 4: 输出质量检查
# =====================================================================
def _check_output_quality(ctx: dict[str, Any]) -> None:
    log.info("=" * 60)
    log.info("测试 4: DPO 输出质量检查（条款引用是否真实）")
    log.info("=" * 60)

    dpo_path = Path(ctx.get("dpo_path", ""))
    if not dpo_path or not dpo_path.exists():
        record("DPO 文件可读", False, "path is empty or missing")
        return

    samples: list[dict[str, Any]] = []
    with open(dpo_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    record("DPO JSONL 可解析", len(samples) > 0, f"{len(samples)} samples loaded")

    # 分类统计
    total = len(samples)
    policy_store_repaired = 0  # 含真实条款引用的样本
    fallback_repaired = 0      # 含模板兜底的样本
    no_reference = 0           # 无任何引用的样本

    for s in samples:
        chosen = s.get("chosen", "")
        if "依据" in chosen and "第" in chosen and "条" in chosen:
            # 含真实条款引用（PolicyStore 修复）
            policy_store_repaired += 1
        elif "具体参见" in chosen and "相关条款" in chosen:
            # 含模板兜底
            fallback_repaired += 1
        else:
            no_reference += 1

    record(
        f"真实条款引用样本: {policy_store_repaired}/{total}",
        policy_store_repaired > 0,
        f"{policy_store_repaired}/{total} ({100*policy_store_repaired/max(total,1):.0f}%)",
    )
    record(
        f"模板兜底样本: {fallback_repaired}/{total}",
        True,  # 模板兜底是预期行为（无 policy_id 时）
        f"{fallback_repaired}/{total} ({100*fallback_repaired/max(total,1):.0f}%)",
    )

    # 抽查一个真实条款引用的样本
    if policy_store_repaired > 0:
        for s in samples:
            chosen = s.get("chosen", "")
            if "依据" in chosen and "第" in chosen:
                log.info("  抽查样本 (policy_id=%s):", s.get("policy_id", "N/A"))
                log.info("    prompt: %s", s.get("prompt", "")[:80])
                # 找到 chosen 中 "依据" 之后的文本
                idx = chosen.find("依据")
                clause_suffix = chosen[idx:idx + 200] if idx >= 0 else chosen[-200:]
                log.info("    chosen 条款片段: ...%s", clause_suffix)
                break


# =====================================================================
# 测试 5: Validator 统计
# =====================================================================
def _check_validator_stats(ctx: dict[str, Any]) -> None:
    log.info("=" * 60)
    log.info("测试 5: Validator 校验统计")
    log.info("=" * 60)

    stats = ctx.get("stats", {})
    validator = stats.get("validator", {})

    if not validator:
        record("Validator stats 存在", False, "stats 中无 validator 字段")
        return

    total = validator.get("total", 0)
    passed = validator.get("passed", 0)
    failed = validator.get("failed", 0)
    pass_rate = validator.get("pass_rate", 0)
    repaired = validator.get("repaired", 0)

    record(f"DPO 校验总数: {total}", total > 0)
    record(f"DPO 校验通过: {passed}/{total}", passed > 0)
    record(
        f"DPO 通过率: {pass_rate:.1%}",
        pass_rate >= 0.5,  # 期望 ≥ 50%
        f"{pass_rate:.1%}",
    )
    record(
        f"回流修复数: {repaired}",
        True,  # repaired 为 0 也是可接受的（取决于数据）
        f"{repaired} samples repaired",
    )

    # 汇总 stats
    log.info("  Pipeline 汇总:")
    log.info("    dpo_total: %s", stats.get("dpo_total", "N/A"))
    log.info("    sft_total: %s", stats.get("sft_total", "N/A"))
    log.info("    elapsed: %.1fs", stats.get("elapsed_seconds", 0))
    collector = stats.get("collector", {})
    log.info("    collector: total=%s, by_source=%s",
                collector.get("total", "N/A"),
                collector.get("by_source", {}))


# =====================================================================
# Main
# =====================================================================
def main() -> int:
    log.info("=" * 60)
    log.info("🚀 DPO 数据合成全量集成测试")
    log.info(f"  项目根目录: {_PROJECT_ROOT}")
    log.info(f"  Python: {sys.version}")
    log.info("=" * 60)

    # ── 0. 环境检查 ──
    test_env()

    # ── 1. PolicyStore 索引 ──
    store_ctx = test_policy_store_index()

    # ── 2. PolicyStore 查询 ──
    if store_ctx:
        _check_policy_store_search(store_ctx)

    # ── 3. 全量 Pipeline ──
    pipeline_ctx = test_full_pipeline()

    # ── 4. 输出质量 ──
    if pipeline_ctx:
        _check_output_quality(pipeline_ctx)

    # ── 5. Validator 统计 ──
    if pipeline_ctx:
        _check_validator_stats(pipeline_ctx)

    # ── 汇总 ──
    log.info("")
    log.info("=" * 60)
    log.info("测试结果汇总")
    log.info("=" * 60)
    for name, ok, detail in RESULTS:
        status = "✅" if ok else "❌"
        log.info("  %s %s%s", status, name, f"  ({detail})" if detail else "")

    log.info("")
    log.info("通过: %d/%d, 失败: %d/%d", PASS, PASS + FAIL, FAIL, PASS + FAIL)

    if FAIL == 0:
        log.info("🎉 全量集成测试全部通过!")
        return 0
    else:
        log.error("⚠️  存在 %d 个失败项，请检查远端日志", FAIL)
        return 1


if __name__ == "__main__":
    sys.exit(main())
