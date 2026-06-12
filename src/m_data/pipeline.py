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

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from m_data.exporter import Exporter
from m_data.normalizer import Normalizer
from m_data.pair_builder import PairBuilder
from m_data.pii_scrubber import PIIScrubber
from m_data.sft_builder import SFTBuilder
from m_data.sources.base import DataSource
from m_data.validator import Validator

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
        self._normalizer: Optional[Normalizer] = None
        self._scrubber: Optional[PIIScrubber] = None
        self._pair_builder: Optional[PairBuilder] = None
        self._sft_builder: Optional[SFTBuilder] = None
        self._validator: Optional[Validator] = None
        self._dpo_exporter: Optional[Exporter] = None
        self._sft_exporter: Optional[Exporter] = None

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
        self._pair_builder = PairBuilder(
            enabled_strategies=enabled,
            judge_model=judge_cfg.get("judge_model", "qwen2.5-7b-instruct"),
            judge_endpoint=judge_cfg.get("judge_endpoint"),
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
        # Step 4: 过滤
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
        # Step 6: 校验 + 写出
        # ------------------------------------------------------------------
        logger.info("[Pipeline] Step 6/6: Validating & exporting")

        if dpo_samples:
            dpo_valid = self.validate(dpo_samples)
            if not dry_run:
                written = self._dpo_exporter.write_stream(iter(dpo_valid))
                self._stats["exporter"]["dpo_written"] = written
                logger.info("[Pipeline] DPO exported: %d samples", written)

        if sft_samples:
            # SFT 样本也走一次 Validator（跳过 chosen/rejected 特定规则）
            sft_valid = self._validate_sft(sft_samples)
            if not dry_run:
                written = self._sft_exporter.write_stream(iter(sft_valid))
                self._stats["exporter"]["sft_written"] = written
                logger.info("[Pipeline] SFT exported: %d samples", written)

        # ------------------------------------------------------------------
        # 总耗时与统计
        # ------------------------------------------------------------------
        elapsed = time.perf_counter() - t0
        self._stats["finished_at"] = datetime.now().isoformat()
        self._stats["elapsed_seconds"] = round(elapsed, 2)
        self._stats["dpo_total"] = len(dpo_samples)
        self._stats["sft_total"] = len(sft_samples)

        logger.info("[Pipeline] Finished in %.1fs", elapsed)
        logger.info(
            "[Pipeline] Summary: %d DPO, %d SFT samples",
            len(dpo_samples),
            len(sft_samples),
        )

        return self._stats

    # ------------------------------------------------------------------
    # 各阶段方法（可独立调用）
    # ------------------------------------------------------------------

    def collect(self, since: Optional[datetime] = None) -> list[dict[str, Any]]:
        """从所有数据源采集记录。"""
        records: list[dict[str, Any]] = []
        for source in self._sources:
            count = 0
            for raw in source.fetch(since=since, limit=self._cfg.get("limit", 0)):
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
        """过滤无效记录（空问题/空答案/过短等）。"""
        quality = self._cfg.get("quality", {})
        min_q_len = quality.get("min_prompt_len", 5)
        min_a_len = quality.get("min_response_len", 10)

        filtered = []
        for rec in records:
            question = rec.get("question") or rec.get("user_question", "")
            answer = rec.get("answer") or rec.get("agent_answer", "") or rec.get("article_content", "")
            if len(question) >= min_q_len and len(answer) >= min_a_len:
                filtered.append(rec)
        return filtered

    def build_dpo_pairs(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """构造 DPO 配对。"""
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
