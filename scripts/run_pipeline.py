#!/usr/bin/env python3
"""M02 流水线运行脚本 —— 在 server2 上执行。"""
import sys
import json
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from m_data.pipeline import Pipeline

config_path = Path(__file__).resolve().parent.parent / "configs" / "data" / "insurance_dpo_gen.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

pipeline = Pipeline(config)
stats = pipeline.run()

# Save stats
log_dir = Path(__file__).resolve().parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
with open(log_dir / "pipeline_stats.json", "w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2, default=str)

print("=== PIPELINE COMPLETE ===")
print(f"DPO samples: {stats.get('dpo_total', 0)}")
print(f"SFT samples: {stats.get('sft_total', 0)}")
print(f"Elapsed: {stats.get('elapsed_seconds', 0):.1f}s")
print(f"Collector total: {stats.get('collector', {}).get('total', 0)}")
print(f"Validator DPO passed: {stats.get('validator', {})}")
