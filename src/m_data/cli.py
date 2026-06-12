"""M02 数据流水线 CLI 入口。

使用方式：
    PYTHONPATH=src python -m m_data.cli --config configs/data/insurance_dpo_gen.yaml
    PYTHONPATH=src python -m m_data.cli --config configs/data/insurance_dpo_gen.yaml --dry-run
    PYTHONPATH=src python -m m_data.cli --config configs/data/insurance_dpo_gen.yaml --since 2026-06-01
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

from m_data.pipeline import Pipeline

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M02 DPO/SFT 数据集生成流水线",
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="流水线配置文件路径（YAML）",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="增量采集起点日期（YYYY-MM-DD）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干跑模式，仅统计不写出文件",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        print(f"Error: config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error: failed to parse YAML config: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)
    logger.info("Loaded config from %s", args.config)

    since = None
    if args.since:
        try:
            since = datetime.strptime(args.since, "%Y-%m-%d")
        except ValueError:
            print(
                f"Error: invalid --since date '{args.since}'. "
                f"Expected format: YYYY-MM-DD",
                file=sys.stderr,
            )
            sys.exit(1)
        logger.info("Incremental mode: since=%s", since.isoformat())

    pipeline = Pipeline(config)
    stats = pipeline.run(since=since, dry_run=args.dry_run)

    # 输出摘要
    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    print(f"  DPO samples:  {stats.get('dpo_total', 0)}")
    print(f"  SFT samples:  {stats.get('sft_total', 0)}")
    print(f"  Elapsed:      {stats.get('elapsed_seconds', 0):.1f}s")
    val = stats.get("validator", {})
    if val:
        print(f"  Validator:    {val.get('passed', 0)}/{val.get('total', 0)} "
              f"passed ({val.get('pass_rate', 0)*100:.1f}%)")
    print("=" * 60)

    if args.dry_run:
        print("[DRY-RUN] No files were written.")
    else:
        dpo_w = stats.get("exporter", {}).get("dpo_written", 0)
        sft_w = stats.get("exporter", {}).get("sft_written", 0)
        print(f"  DPO written:  {dpo_w}")
        print(f"  SFT written:  {sft_w}")


if __name__ == "__main__":
    main()
