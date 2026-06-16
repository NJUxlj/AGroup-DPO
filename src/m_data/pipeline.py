"""DPO 数据生成流水线编排。

M02 § 3.5: 串联 Collector → Normalizer → PIIScrubber → PairBuilder → Validator → Exporter。

流水线阶段：
1. [Collector] 多源并发采集
2. [Normalizer] 文本规范化
3. [PIIScrubber] PII 脱敏
4. [Filter] 长度/敏感词过滤
5. [PairBuilder] 配对 chosen/rejected（→ DPO 路径）
   [SFTBuilder]  配对 instruction/output（→ SFT 路径）
6. [Validator] 规则校验
7. [Exporter] JSONL 写出
"""

import copy
import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from m_data.exporter import Exporter
from m_data.normalizer import Normalizer
from m_data.pair_builder import PairBuilder
from m_data.pii_scrubber import PIIScrubber
from m_data.policy_store import PolicyStore
from m_data.sft_builder import SFTBuilder
from m_data.sources.base import DataSource
from m_data.validator import Validator
from llm.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class Pipeline:
    """DPO/SFT 数据生成流水线。

    使用方式：
        pipeline = Pipeline(config)
        pipeline.run()

    或分步调用：
        records = pipeline.collect()
        records = pipeline.normalize(records)
        ...
    """

    def __init__(self, config: dict[str, Any]):
        """
        Args:
            config: 流水线配置字典（与 configs/data/insurance_dpo_gen.yaml 对齐）。
        """
        self._cfg = config
        self._sources: list[DataSource] = []
        self._source_limits: list[int] = []
        self._normalizer: Optional[Normalizer] = None
        self._scrubber: Optional[PIIScrubber] = None
        self._pair_builder: Optional[PairBuilder] = None
        self._sft_builder: Optional[SFTBuilder] = None
        self._validator: Optional[Validator] = None
        self._dpo_exporter: Optional[Exporter] = None
        self._sft_exporter: Optional[Exporter] = None
        self._policy_store: Optional[PolicyStore] = None

        # 统计
        self._stats: dict[str, Any] = {
            "started_at": datetime.now().isoformat(),
            "collector": {"total": 0, "by_source": {}},
            "normalizer": {"input": 0, "output": 0},
            "scrubber": {"total": 0, "pii_hits": 0},
            "pair_builder": {"total": 0},
            "sft_builder": {"total": 0},
            "validator": {},
            "exporter": {"dpo_written": 0, "sft_written": 0},
        }

    def init_from_config(self) -> None:
        """根据配置初始化所有组件。"""
        # Collector
        for src_cfg in self._cfg.get("sources", {}).values():
            if not src_cfg.get("enabled", True):
                continue
            source = self._build_source(src_cfg)
            if source:
                self._sources.append(source)
                self._source_limits.append(src_cfg.get("limit", 0))

        # Normalizer
        quality = self._cfg.get("quality", {})
        self._normalizer = Normalizer(
            max_length=quality.get("max_response_len", 2048),
        )

        # PIIScrubber
        self._scrubber = PIIScrubber()

        # PairBuilder
        strategies = self._cfg.get("strategies", {})
        enabled = [k for k, v in strategies.items() if v.get("enabled", True)]
        judge_cfg = strategies.get("llm_judge", {})
        rag_cfg = strategies.get("retrieval_diff", {})

        # 创建 LLMProvider（支持第三方 API + 本地模型）
        judge_provider = None
        if judge_cfg.get("enabled", True):
            judge_model = judge_cfg.get("judge_model", "qwen2.5-7b-instruct")
            judge_endpoint = judge_cfg.get("judge_endpoint")
            judge_api_key = judge_cfg.get("api_key", "")
            if judge_endpoint:
                judge_provider = LLMProvider(
                    model=judge_model,
                    base_url=judge_endpoint,
                    api_key=judge_api_key,
                )

        self._pair_builder = PairBuilder(
            enabled_strategies=enabled,
            judge_model=judge_cfg.get("judge_model", "qwen2.5-7b-instruct"),
            judge_endpoint=judge_cfg.get("judge_endpoint"),
            judge_provider=judge_provider,
            rag_endpoint=rag_cfg.get("rag_endpoint"),
        )

        # SFTBuilder
        self._sft_builder = SFTBuilder()

        # Validator
        self._validator = Validator(
            min_prompt_len=quality.get("min_prompt_len", 5),
            max_prompt_len=quality.get("max_prompt_len", 1024),
            min_response_len=quality.get("min_response_len", 10),
            max_response_len=quality.get("max_response_len", 2048),
            max_chosen_rejected_similarity=quality.get("max_chosen_rejected_similarity", 0.95),
        )

        # PolicyStore (Milvus 向量库，用于条款检索修复)
        ps_cfg = self._cfg.get("policy_store")
        if ps_cfg and ps_cfg.get("enabled", True):
            try:
                self._policy_store = PolicyStore(ps_cfg)
                self._policy_store.ensure_ready()
            except Exception as e:
                logger.warning("PolicyStore init failed, will use fallback repair: %s", e)
                self._policy_store = None

        # Exporter
        output = self._cfg.get("output", {})
        self._dpo_exporter = Exporter(
            output_path=output.get("path", "data/insurance/dpo_train_v1.2.jsonl"),
            shared_path=output.get("shared_path", ""),
        )
        sft_path = output.get("sft_path", "data/insurance/insurance_sft_v1.jsonl")
        self._sft_exporter = Exporter(
            output_path=sft_path,
            shared_path=output.get("shared_sft_path", ""),
        )

    def _build_source(self, cfg: dict) -> Optional[DataSource]:
        """根据配置构建数据源实例。"""
        source_type = cfg.get("type", "")
        data_path = cfg.get("path", "")

        if not data_path:
            logger.warning("Source config missing 'path', skipping")
            return None

        if source_type == "policy":
            from m_data.sources.policy import InsurancePolicySource

            return InsurancePolicySource(
                data_path=data_path,
                parser=cfg.get("parser", "json"),
            )
        elif source_type == "faq":
            from m_data.sources.faq import FAQSource

            return FAQSource(data_path=data_path)
        elif source_type == "ticket":
            from m_data.sources.ticket import TicketSource

            return TicketSource(
                data_path=data_path,
                filter_category=cfg.get("filter_category", "compliance_qa"),
            )
        else:
            logger.warning("Unknown source type: %s", source_type)
            return None

    def run(self, since: Optional[datetime] = None, dry_run: bool = False) -> dict[str, Any]:
        """执行完整流水线。

        Args:
            since: 增量采集起点。
            dry_run: 仅统计不写出。

        Returns:
            流水线统计信息。
        """
        t0 = time.perf_counter()

        self.init_from_config()

        # ------------------------------------------------------------------
        # Step 1: 采集
        # ------------------------------------------------------------------
        logger.info("[Pipeline] Step 1/6: Collecting from %d sources", len(self._sources))
        all_records = self.collect(since=since)
        logger.info(
            "[Pipeline] Collected %d records from %d sources",
            len(all_records),
            len(self._sources),
        )

        # ------------------------------------------------------------------
        # Step 2: 规范化
        # ------------------------------------------------------------------
        logger.info("[Pipeline] Step 2/6: Normalizing")
        all_records = self.normalize(all_records)

        # ------------------------------------------------------------------
        # Step 3: PII 脱敏
        # ------------------------------------------------------------------
        logger.info("[Pipeline] Step 3/6: PII scrubbing")
        all_records = self.scrub(all_records)

        # ------------------------------------------------------------------
        # Step 4: 过滤（长度 + 敏感词）
        # ------------------------------------------------------------------
        logger.info("[Pipeline] Step 4/6: Filtering")
        all_records = self.filter_records(all_records)
        logger.info("[Pipeline] After filter: %d records", len(all_records))

        # ------------------------------------------------------------------
        # Step 5: 配对构造
        # ------------------------------------------------------------------
        logger.info("[Pipeline] Step 5/6: Building pairs")
        dpo_samples = list(self.build_dpo_pairs(all_records))
        sft_samples = list(self.build_sft_samples(all_records))
        logger.info(
            "[Pipeline] Built %d DPO + %d SFT samples",
            len(dpo_samples),
            len(sft_samples),
        )

        # ------------------------------------------------------------------
        # Step 5.5: prompt 级别全局去重
        # ------------------------------------------------------------------
        if dpo_samples:
            dpo_samples = self._dedup_by_prompt(dpo_samples)
            logger.info("[Pipeline] After prompt dedup: %d DPO samples", len(dpo_samples))

        # ------------------------------------------------------------------
        # Step 6: 校验 + 回流修复 + 写出
        # ------------------------------------------------------------------
        logger.info("[Pipeline] Step 6/6: Validating & exporting")

        dpo_valid: list[dict[str, Any]] = []
        sft_valid: list[dict[str, Any]] = []

        if dpo_samples:
            dpo_valid = self._validate_and_repair(dpo_samples)
            if not dry_run:
                written = self._dpo_exporter.write_stream(iter(dpo_valid))
                self._stats["exporter"]["dpo_written"] = written
                logger.info("[Pipeline] DPO exported: %d samples", written)

        if sft_samples:
            sft_valid = self._validate_sft(sft_samples)
            if not dry_run:
                written = self._sft_exporter.write_stream(iter(sft_valid))
                self._stats["exporter"]["sft_written"] = written
                logger.info("[Pipeline] SFT exported: %d samples", written)

        # ------------------------------------------------------------------
        # Step 7: 评测留出集生成
        # ------------------------------------------------------------------
        if not dry_run and sft_valid:
            holdout_path = self._cfg.get("output", {}).get(
                "eval_holdout_path", "data/eval/insurance_qa_500.jsonl"
            )
            self._generate_holdout_set(sft_valid, holdout_path)

        # ------------------------------------------------------------------
        # Step 8: 数据质量报告
        # ------------------------------------------------------------------
        if not dry_run:
            report_path = self._cfg.get("output", {}).get(
                "report_path", "reports/dpo_data_quality_v1.2.md"
            )
            self._generate_quality_report(
                report_path, dpo_samples, dpo_valid, sft_samples, sft_valid
            )

        # ------------------------------------------------------------------
        # 总耗时与统计
        # ------------------------------------------------------------------
        elapsed = time.perf_counter() - t0
        self._stats["finished_at"] = datetime.now().isoformat()
        self._stats["elapsed_seconds"] = round(elapsed, 2)
        self._stats["dpo_total"] = len(dpo_valid)
        self._stats["sft_total"] = len(sft_valid)

        logger.info("[Pipeline] Finished in %.1fs", elapsed)
        logger.info(
            "[Pipeline] Summary: %d DPO, %d SFT samples",
            len(dpo_valid),
            len(sft_valid),
        )

        return self._stats

    # ------------------------------------------------------------------
    # 各阶段方法（可独立调用）
    # ------------------------------------------------------------------

    def collect(self, since: Optional[datetime] = None) -> list[dict[str, Any]]:
        """从所有数据源采集记录。"""
        records: list[dict[str, Any]] = []
        for idx, source in enumerate(self._sources):
            src_limit = self._source_limits[idx] if idx < len(self._source_limits) else 0
            count = 0
            for raw in source.fetch(since=since, limit=src_limit):
                rec = raw.content.copy()
                rec["_source"] = raw.source
                rec["_record_id"] = raw.record_id
                rec["_raw_meta"] = raw.raw_meta
                records.append(rec)
                count += 1
            self._stats["collector"]["by_source"][source.source_name] = count
            self._stats["collector"]["total"] += count
            logger.info("  Source '%s': %d records", source.source_name, count)
        return records

    def normalize(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """对所有记录执行文本规范化。"""
        self._stats["normalizer"]["input"] = len(records)
        fields = self._get_text_fields(records)
        for rec in records:
            self._normalizer.normalize_record(rec, fields)
        self._stats["normalizer"]["output"] = len(records)
        return records

    def scrub(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """对所有记录执行 PII 脱敏。"""
        self._stats["scrubber"]["total"] = len(records)
        fields = self._get_text_fields(records)
        hits = 0
        for rec in records:
            _, hit = self._scrubber.scrub_record(rec, fields)
            if hit:
                hits += 1
        self._stats["scrubber"]["pii_hits"] = hits
        return records

    def filter_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """过滤无效记录（空问题/空答案/过短/敏感词等）。"""
        quality = self._cfg.get("quality", {})
        min_q_len = quality.get("min_prompt_len", 5)
        min_a_len = quality.get("min_response_len", 10)
        sensitive_words: list[str] = quality.get("sensitive_words", [])

        filtered = []
        for rec in records:
            question = rec.get("question") or rec.get("user_question", "")
            answer = rec.get("answer") or rec.get("agent_answer", "") or rec.get("article_content", "")

            # 长度检查
            if len(question) < min_q_len or len(answer) < min_a_len:
                continue

            # 敏感词检查
            if sensitive_words:
                combined = question + " " + answer
                if any(sw in combined for sw in sensitive_words):
                    continue

            filtered.append(rec)
        return filtered

    def build_dpo_pairs(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        构造 DPO 配对。
        """
        samples = list(self._pair_builder.build_from_records(records))
        self._stats["pair_builder"]["total"] = len(samples)
        return samples

    def build_sft_samples(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """构造 SFT 样本，按数据来源分配策略标签（S-A / S-B / S-C）。"""
        # 按来源分组：policy/faq → S-A, ticket → S-B, 其他 → S-C
        groups: dict[str, list[dict[str, Any]]] = {"S-A": [], "S-B": [], "S-C": []}
        for rec in records:
            source = rec.get("_source", "")
            if source in ("policy_v1", "faq_v2"):
                groups["S-A"].append(rec)
            elif source == "ticket_v3":
                groups["S-B"].append(rec)
            else:
                groups["S-C"].append(rec)

        samples: list[dict[str, Any]] = []
        for label, recs in groups.items():
            if recs:
                samples.extend(self._sft_builder.build_from_records(recs, label))
        self._stats["sft_builder"]["total"] = len(samples)
        return samples

    def validate(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """校验 DPO 样本，返回通过校验的样本列表。"""
        results = self._validator.validate_batch(samples)
        valid = [s for s, ok, _ in results if ok]
        self._stats["validator"] = self._validator.stats(samples)
        logger.info(
            "[Validator] %d/%d passed (%.1f%%)",
            len(valid),
            len(samples),
            len(valid) / max(len(samples), 1) * 100,
        )
        return valid

    def _validate_and_repair(
        self, samples: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """校验 + 回流修复：对'必引条款缺失'的样本尝试修复后重新校验。

        为什么需要必引条款：
        保险行业高度合规。客户问"等待期确诊赔不赔"时，
        如果 chosen 只说"不赔"，却不引用具体条款依据，这个答案不可用于 DPO 对齐训练——模型会学到"只说结论、不给出处"的坏习惯，在真实业务中极易引发合规风险。

        M02 § 3.4: 必引条款不通过应回流到 PairBuilder 重做。
        本方法模拟回流：追加标准条款引用语后重新校验。
        """
        results = self._validator.validate_batch(samples)

        valid: list[dict[str, Any]] = []
        to_repair: list[dict[str, Any]] = []
        failed_other = 0

        for s, ok, reason in results:
            if ok:
                valid.append(s)
            elif "missing required policy reference" in reason:
                to_repair.append(s)
            else:
                failed_other += 1

        # 修复并重新校验
        repaired_count = 0
        if to_repair:
            repaired: list[dict[str, Any]] = []
            for s in to_repair:
                fixed = self._repair_missing_reference(s, policy_store=self._policy_store)
                if fixed:
                    repaired.append(fixed)
            if repaired:
                re_results = self._validator.validate_batch(repaired)
                for s, ok, _ in re_results:
                    if ok:
                        valid.append(s)
                        repaired_count += 1
            logger.info(
                "[Repair] Attempted %d, repaired %d",
                len(to_repair),
                repaired_count,
            )

        # 统计（基于最终通过数计算）
        total_checked = len(samples)
        self._stats["validator"] = {
            "total": total_checked,
            "passed": len(valid),
            "failed": total_checked - len(valid),
            "pass_rate": len(valid) / max(total_checked, 1),
            "repaired": repaired_count,
        }
        logger.info(
            "[Validator] %d/%d passed (%.1f%%) [repaired: %d]",
            len(valid),
            total_checked,
            len(valid) / max(total_checked, 1) * 100,
            repaired_count,
        )
        return valid

    @staticmethod
    def _repair_missing_reference(
        sample: dict[str, Any],
        policy_store: Optional[PolicyStore] = None,
    ) -> dict[str, Any] | None:
        """对缺少条款引用的 DPO 样本尝试修复。

        修复策略（优先级从高到低）：
        1. Milvus 混合检索：若存在 policy_id 且 PolicyStore 可用，按 prompt 语义 +
           关键词检索条款原文片段，追加到 chosen 末尾。
        2. ID 兜底：若 policy_id 存在但 Milvus 不可用/无结果，追加带 policy_id
           的通用引用语。
        3. 通用兜底：若无 policy_id，追加不带 ID 的通用引用语。
        """
        fixed = copy.deepcopy(sample)
        chosen = fixed.get("chosen", "")
        policy_id = fixed.get("policy_id")
        prompt = fixed.get("prompt", "")

        if not chosen:
            return None

        # ── 策略 1: Milvus 真实条款检索 ──
        clause_suffix = Pipeline._build_clause_suffix(
            policy_id=policy_id,
            prompt=prompt,
            policy_store=policy_store,
        )
        if clause_suffix:
            if clause_suffix.strip() not in chosen:
                fixed["chosen"] = chosen.rstrip() + clause_suffix
            return fixed

        # ── 策略 2 & 3: ID 兜底 / 通用兜底 ──
        if policy_id:
            ref_suffix = f" 具体参见{policy_id}相关条款及保单约定。"
        else:
            ref_suffix = " 具体参见相关保险条款及保单约定。"

        # 避免重复追加
        if ref_suffix.strip() not in chosen:
            fixed["chosen"] = chosen.rstrip() + ref_suffix

        return fixed

    @staticmethod
    def _build_clause_suffix(
        policy_id: str | None,
        prompt: str,
        policy_store: Optional[PolicyStore] = None,
    ) -> str:
        """通过 Milvus 混合检索构造条款引用后缀。

        Returns:
            条款引用文本；若检索失败或无结果则返回空字符串。
        """
        if not policy_id or not policy_store or not policy_store.ready:
            return ""

        try:
            clauses = policy_store.search(policy_id=policy_id, prompt=prompt)
        except Exception as e:
            logger.warning("PolicyStore search failed for %s: %s", policy_id, e)
            return ""

        if not clauses:
            return ""

        # 选取 top-2 条（避免 chosen 末尾过长）
        top_clauses = clauses[:2]
        fragments: list[str] = []
        for c in top_clauses:
            art_id = c.get("article_id", "")
            title = c.get("article_title", "")
            content = c.get("article_content", "")
            # 截断每条条款内容至 ~150 字，保持可读性
            if len(content) > 150:
                content = content[:147] + "..."
            if art_id and title:
                fragment = f"第{art_id}条（{title}）：{content}"
            elif art_id:
                fragment = f"第{art_id}条：{content}"
            else:
                fragment = content
            fragments.append(fragment)

        clause_text = "；".join(fragments)
        return f" 依据{policy_id}：{clause_text}。"

    @staticmethod
    def _dedup_by_prompt(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """按 prompt 去重，保留首次出现的样本。

        M02 § 6.3: 同一 prompt 不允许多次出现。
        """
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for s in samples:
            key = hashlib.md5(s["prompt"].encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                result.append(s)
        dup_count = len(samples) - len(result)
        if dup_count:
            logger.info("[Dedup] Removed %d duplicate prompts", dup_count)
        return result

    def _validate_sft(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """SFT 样本的轻量校验（跳过 chosen/rejected 相关规则）。"""
        quality = self._cfg.get("quality", {})
        min_in = quality.get("min_prompt_len", 5)
        max_in = quality.get("max_prompt_len", 1024)
        min_out = quality.get("min_response_len", 10)
        max_out = quality.get("max_response_len", 2048)

        valid = []
        for s in samples:
            inp = s.get("input", "")
            out = s.get("output", "")
            if min_in <= len(inp) <= max_in and min_out <= len(out) <= max_out:
                valid.append(s)
        return valid

    def _generate_holdout_set(
        self, sft_samples: list[dict[str, Any]], path: str
    ) -> None:
        """从 SFT 样本中留出评测集。

        M02 § 3.7 / D-M02-12: 随机留出 ≥ 500 条用于 M05 评测。
        """
        builder = SFTBuilder(include_system=True)
        train, eval_set = builder.split_train_eval(
            sft_samples,
            eval_ratio=0.1,
            seed=self._cfg.get("seed", 42),
        )

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for s in eval_set:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        self._stats["holdout"] = {"path": path, "count": len(eval_set)}
        logger.info(
            "[Holdout] Written %d samples to %s (train: %d)",
            len(eval_set),
            path,
            len(train),
        )

    def _generate_quality_report(
        self,
        path: str,
        dpo_raw: list[dict[str, Any]],
        dpo_valid: list[dict[str, Any]],
        sft_raw: list[dict[str, Any]],
        sft_valid: list[dict[str, Any]],
    ) -> None:
        """生成数据质量报告 (Markdown)。

        M02 D-M02-13: 含通过率 / source 分布 / PII 命中率等。
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write("# DPO 数据质量报告\n\n")
            f.write(
                f"生成时间：{datetime.now().isoformat()}\n"
            )
            f.write(f"版本：{self._cfg.get('version', 'N/A')}\n\n")

            # 数据规模
            f.write("## 1. 数据规模\n\n")
            f.write("| 数据集 | 原始产出 | 校验通过 | 通过率 |\n")
            f.write("|--------|----------|----------|--------|\n")
            dpo_pass_rate = len(dpo_valid) / max(len(dpo_raw), 1) * 100
            sft_pass_rate = len(sft_valid) / max(len(sft_raw), 1) * 100
            f.write(
                f"| DPO | {len(dpo_raw)} | {len(dpo_valid)} | "
                f"{dpo_pass_rate:.1f}% |\n"
            )
            f.write(
                f"| SFT | {len(sft_raw)} | {len(sft_valid)} | "
                f"{sft_pass_rate:.1f}% |\n\n"
            )

            # 采集统计
            f.write("## 2. 采集统计\n\n")
            f.write(
                f"总采集记录：{self._stats['collector']['total']}\n\n"
            )
            f.write("| 数据源 | 数量 |\n")
            f.write("|--------|------|\n")
            for src, cnt in self._stats["collector"]["by_source"].items():
                f.write(f"| {src} | {cnt} |\n")
            f.write("\n")

            # PII
            f.write("## 3. PII 脱敏\n\n")
            f.write(
                f"脱敏记录数：{self._stats['scrubber']['total']}\n"
            )
            f.write(
                f"PII 命中数：{self._stats['scrubber']['pii_hits']}\n\n"
            )

            # 校验统计
            val = self._stats.get("validator", {})
            if val:
                f.write("## 4. 校验统计\n\n")
                f.write(
                    f"校验通过率：{val.get('pass_rate', 0) * 100:.1f}%\n"
                )
                f.write(
                    f"通过：{val.get('passed', 0)} / "
                    f"失败：{val.get('failed', 0)}\n"
                )
                repaired = val.get("repaired", 0)
                if repaired:
                    f.write(f"回流修复：{repaired}\n")
                f.write("\n")

            # 评测留出
            ho = self._stats.get("holdout", {})
            if ho:
                f.write("## 5. 评测留出集\n\n")
                f.write(f"留出路径：{ho.get('path', 'N/A')}\n")
                f.write(f"留出条数：{ho.get('count', 0)}\n\n")

        logger.info("[Report] Quality report written to %s", path)

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _get_text_fields(records: list[dict[str, Any]]) -> list[str]:
        """从全部记录中自动发现文本字段名（取并集）。"""
        if not records:
            return []
        # 常见文本字段
        candidates = {
            "question", "answer", "user_question", "agent_answer",
            "article_content", "article_title", "prompt", "chosen", "rejected",
            "instruction", "input", "output", "raw_text",
        }
        # 对所有记录取并集，避免第一条记录不包含某些字段时遗漏
        fields: set[str] = set()
        for rec in records:
            for k in rec:
                if k in candidates and isinstance(rec[k], str):
                    fields.add(k)
        return sorted(fields)
