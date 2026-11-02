"""Settings from configs/excise.yaml (report Appendix A), exposed as plain constants."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(open(ROOT / "configs" / "excise.yaml"))

DATASET = CONFIG["dataset"]
BACKBONE = CONFIG["backbone"]
SPLITS = CONFIG["splits"]
WINDOW = CONFIG["windows"]["length"]
STRIDE = CONFIG["windows"]["stride"]
TRAIN = CONFIG["train"]
DECODE = CONFIG["decode"]
