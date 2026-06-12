#!/usr/bin/env python3
"""合成数据生成器——用 vLLM (Qwen2.5-1.5B-Instruct) 大批量生成 FAQ 和工单数据。

用法：
    # 生成 FAQ 数据（默认 3000 条）
    python scripts/gen_synthetic_data.py --mode faq --total 3000 --output data/insurance/raw/faq/faq_synthetic.json

    # 生成工单数据（默认 1000 条）
    python scripts/gen_synthetic_data.py --mode ticket --total 1000 --output data/insurance/raw/tickets/tickets_synthetic.json

    # 同时生成两种
    python scripts/gen_synthetic_data.py --mode all --faq-count 3000 --ticket-count 1000
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# FAQ 生成模板
# =============================================================================

FAQ_CATEGORIES = [
    {
        "category": "重大疾病保险",
        "topics": [
            "等待期规定和例外情况",
            "疾病覆盖范围和理赔条件",
            "轻症/中症/重疾的区别",
            "保费豁免条件",
            "多次赔付机制",
            "保额如何选择",
            "健康告知注意事项",
            "既往症承保条件",
            "原位癌理赔",
            "甲状腺癌分级赔付",
            "特定疾病额外赔付",
            "少儿/成人重疾险差异",
            "癌症多次赔付间隔期",
            "身故责任与重疾责任关系",
            "遗传性疾病免责",
        ],
    },
    {
        "category": "百万医疗保险",
        "topics": [
            "免赔额计算规则",
            "社保报销抵扣免赔额",
            "住院/门诊/特需费用报销",
            "进口药/自费药报销",
            "保证续保与非保证续保区别",
            "保费随年龄调整机制",
            "理赔后能否续保",
            "等待期差异（疾病/特定疾病）",
            "异地就医报销",
            "多份医疗险理赔顺序",
            "中医/康复治疗报销",
            "住院津贴附加险",
        ],
    },
    {
        "category": "意外伤害保险",
        "topics": [
            "猝死是否属于意外",
            "伤残评定标准（1-10级）",
            "免责条款（酒驾/高风险运动等）",
            "等待期和生效时间",
            "意外医疗额度与免赔额",
            "职业变更通知义务",
            "多份意外险叠加赔付",
            "境外意外保障",
            "意外住院津贴",
            "交通事故理赔",
            "动物咬伤/食物中毒理赔",
        ],
    },
    {
        "category": "定期寿险",
        "topics": [
            "定期寿险与终身寿险区别",
            "保额计算方法（年收入倍数）",
            "等待期和自杀条款",
            "受益人指定和变更",
            "多份寿险叠加赔付",
            "到期后转保选项",
            "全残责任覆盖",
            "有房贷者的寿险配置",
            "家庭主妇/儿童是否需要寿险",
        ],
    },
    {
        "category": "养老保险",
        "topics": [
            "养老年金领取起始年龄",
            "现金价值与退保损失",
            "保证领取期含义",
            "缴费期限选择策略",
            "保证利益与非保证利益",
            "社保与商业养老互补",
            "身故后受益人领取",
            "保单贷款功能",
            "年金险分红机制",
        ],
    },
    {
        "category": "投保须知",
        "topics": [
            "如实告知义务（保险法第十六条）",
            "不可抗辩条款（两年期）",
            "犹豫期权益（10-15天）",
            "宽限期（60天）和复效",
            "电子保单法律效力",
            "投保年龄限制",
            "体检要求与免体检限额",
            "保险利益原则（为家人投保）",
            "既往症与健康告知关系",
            "怀孕/乙肝携带等特殊情形投保",
            "高风险职业投保",
        ],
    },
    {
        "category": "理赔须知",
        "topics": [
            "理赔时效规定（30日核定）",
            "赔付时限（10日履行）",
            "拒赔后的复核/投诉/诉讼",
            "理赔材料清单与补交",
            "医疗险发票原件要求",
            "异地就医理赔流程",
            "多份保险理赔顺序",
            "先天性疾病免责理赔",
            "未如实告知拒赔抗辩",
            "第三方责任事故理赔",
        ],
    },
]

TICKET_SCENARIOS = [
    {"cat": "compliance_qa", "topic": "等待期内体检发现异常，担心影响后续理赔"},
    {"cat": "compliance_qa", "topic": "投保时忘记告知几年前的小病史"},
    {"cat": "compliance_qa", "topic": "理赔被以未如实告知为由拒赔，询问维权途径"},
    {"cat": "compliance_qa", "topic": "异地就医后如何申请理赔"},
    {"cat": "compliance_qa", "topic": "多份保险发生事故后不知如何申请理赔"},
    {"cat": "compliance_qa", "topic": "怀疑保险公司理赔金额计算有误"},
    {"cat": "compliance_qa", "topic": "被车撞伤，肇事方保险和自己的保险如何协调"},
    {"cat": "compliance_qa", "topic": "保单过期后才发现之前出险了"},
    {"cat": "compliance_qa", "topic": "买保险时医生说没事的病被要求告知"},
    {"cat": "compliance_qa", "topic": "百万医疗险续保时保费大幅上涨"},
    {"cat": "compliance_qa", "topic": "重疾理赔被要求补充多年前的病历"},
    {"cat": "compliance_qa", "topic": "意外险职业变更后出险被拒赔"},
    {"cat": "compliance_qa", "topic": "家人代为投保后不同意，要求退保"},
    {"cat": "compliance_qa", "topic": "保险受益人已故，理赔金归属不明"},
    {"cat": "compliance_qa", "topic": "与保险公司对疾病定义有争议"},
    {"cat": "compliance_qa", "topic": "退休后继续交医疗险保费是否合理"},
    {"cat": "compliance_qa", "topic": "住院费用报销后被要求退还部分款项"},
    {"cat": "compliance_qa", "topic": "出国期间发生意外，国内保险公司是否赔付"},
    {"cat": "compliance_qa", "topic": "体检报告异常想买保险，咨询核保可能性"},
    {"cat": "compliance_qa", "topic": "宽限期最后一天交费但因系统问题未到账"},
]

# =============================================================================
# vLLM 批量生成
# =============================================================================


def build_faq_prompt(category: str, topic: str, count: int) -> str:
    """构造 FAQ 生成 prompt。"""
    return f"""你是AI财保助理的资深客服专家。请为「{category}」类别生成 {count} 个真实的保险客户常见问题及答案。

话题约束：{topic}

要求：
1. 问题要像真实客户提出的（口语化、带具体场景）
2. 答案要专业、准确、符合中国保险法规和行业惯例
3. 答案中涉及「赔付/等待期/免赔/告知」等业务关键词时，必须引用具体条款或法规
4. 输出严格 JSON 数组，每个元素格式：{{"question": "...", "answer": "..."}}

只输出 JSON 数组，不要其他文字。
直接输出 [{{
"""


def build_ticket_prompt(topic: str, count: int) -> str:
    """构造工单生成 prompt。"""
    return f"""你是AI财保助理的客服坐席。请生成 {count} 个真实的历史工单对话，场景为：

话题：{topic}

每个工单包含一个客户问题和一个专业的客服回答。要求：
1. 客户问题要具体、有细节（如投保时间、具体疾病名称等）
2. 客服回答要合规、引用条款或法规，体现专业保险公司客服水平
3. 输出严格 JSON 数组，每个元素格式：{{"user_question": "...", "agent_answer": "..."}}

只输出 JSON 数组，不要其他文字。
直接输出 [{{
"""


def generate_with_vllm(prompts: list[str], max_tokens: int = 512) -> list[str]:
    """用 vLLM 批量生成。"""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
    )

    params = SamplingParams(temperature=0.8, max_tokens=max_tokens, top_p=0.95)
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text.strip() for o in outputs]


def parse_json_response(text: str) -> list[dict]:
    """从模型输出中解析 JSON 数组。"""
    # 尝试找到 JSON 数组边界
    text = text.strip()
    if text.startswith("```"):
        # 去掉 markdown 代码块
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("\n", 1)[0]

    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    except json.JSONDecodeError:
        # 尝试逐行修复
        logger.warning("JSON parse failed, trying line-by-line repair...")
    return []


def generate_faq(batch_size: int = 20, total: int = 3000, output: str = "") -> list[dict]:
    """生成 FAQ 数据。"""
    all_items: list[dict] = []
    prompts: list[str] = []
    metadata: list[dict] = []

    # 构造所有 prompt
    topics_pool = []
    for cat_info in FAQ_CATEGORIES:
        for topic in cat_info["topics"]:
            topics_pool.append((cat_info["category"], topic))

    random.shuffle(topics_pool)

    needed = total
    idx = 0
    while needed > 0 and topics_pool:
        cat, topic = topics_pool[idx % len(topics_pool)]
        count = min(batch_size, needed)
        prompts.append(build_faq_prompt(cat, topic, count))
        metadata.append({"category": cat, "topic": topic})
        needed -= count
        idx += 1

    logger.info("FAQ: %d prompts prepared, target %d samples", len(prompts), total)

    # 用 vLLM 批量生成
    t0 = time.perf_counter()
    responses = generate_with_vllm(prompts, max_tokens=4096)
    elapsed = time.perf_counter() - t0
    logger.info("FAQ: vLLM generation took %.1fs", elapsed)

    # 解析响应
    for resp, meta in zip(responses, metadata):
        items = parse_json_response(resp)
        for item in items:
            item["category"] = meta["category"]
            item["topic"] = meta["topic"]
        all_items.extend(items)
        if items:
            logger.debug("  Parsed %d items from topic '%s'", len(items), meta["topic"])
        else:
            logger.warning("  Failed to parse items from topic '%s'", meta["topic"])

    logger.info("FAQ: total generated %d items", len(all_items))

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(all_items, f, ensure_ascii=False, indent=2)
        logger.info("FAQ: saved to %s", output)

    return all_items


def generate_tickets(batch_size: int = 30, total: int = 1000, output: str = "") -> list[dict]:
    """生成工单数据。"""
    all_items: list[dict] = []
    prompts: list[str] = []

    needed = total
    idx = 0
    while needed > 0 and TICKET_SCENARIOS:
        scenario = TICKET_SCENARIOS[idx % len(TICKET_SCENARIOS)]
        count = min(batch_size, needed)
        prompts.append(build_ticket_prompt(scenario["topic"], count))
        needed -= count
        idx += 1

    logger.info("Ticket: %d prompts prepared, target %d samples", len(prompts), total)

    t0 = time.perf_counter()
    responses = generate_with_vllm(prompts, max_tokens=4096)
    elapsed = time.perf_counter() - t0
    logger.info("Ticket: vLLM generation took %.1fs", elapsed)

    for resp in responses:
        items = parse_json_response(resp)
        for i, item in enumerate(items):
            item["ticket_id"] = f"TK-SYN-{len(all_items) + i + 1:04d}"
            item["category"] = "compliance_qa"
            item["created_at"] = "2026-06-10T00:00:00"
        all_items.extend(items)

    logger.info("Ticket: total generated %d items", len(all_items))

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(all_items, f, ensure_ascii=False, indent=2)
        logger.info("Ticket: saved to %s", output)

    return all_items


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="合成保险数据生成器")
    parser.add_argument("--mode", choices=["faq", "ticket", "all"], default="all")
    parser.add_argument("--faq-count", type=int, default=3000, help="FAQ 目标条数")
    parser.add_argument("--ticket-count", type=int, default=1000, help="工单目标条数")
    parser.add_argument("--faq-output", type=str, default="data/insurance/raw/faq/faq_synthetic.json")
    parser.add_argument("--ticket-output", type=str, default="data/insurance/raw/tickets/tickets_synthetic.json")
    parser.add_argument("--batch-size", type=int, default=15, help="每次生成的条数")
    parser.add_argument("--dry-run", action="store_true", help="不调用模型，只打印 prompt 数量")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("Dry run mode: only counting prompts")
        prompts, _ = 0, []
        needed = args.faq_count
        idx = 0
        pool = []
        for cat_info in FAQ_CATEGORIES:
            for topic in cat_info["topics"]:
                pool.append((cat_info["category"], topic))
        while needed > 0 and pool:
            needed -= min(args.batch_size, needed)
            idx += 1
        logger.info("FAQ prompts needed: ~%d", idx)
        logger.info("Ticket prompts needed: ~%d", max(args.ticket_count // args.batch_size, 1))
        return

    results: dict[str, Any] = {}

    if args.mode in ("faq", "all"):
        items = generate_faq(
            batch_size=args.batch_size,
            total=args.faq_count,
            output=args.faq_output,
        )
        results["faq_count"] = len(items)

    if args.mode in ("ticket", "all"):
        items = generate_tickets(
            batch_size=args.batch_size,
            total=args.ticket_count,
            output=args.ticket_output,
        )
        results["ticket_count"] = len(items)

    logger.info("Done! Summary: %s", json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
