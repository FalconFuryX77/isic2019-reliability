#!/usr/bin/env python
"""
Create the lesion-aware stratified train/val/test split.
Usage:
    python scripts/run_split.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._common import load_config  # noqa: E402
from src.data.split import create_splits  # noqa: E402


def main():
    cfg = load_config()
    create_splits(cfg, Path(cfg.dataset.raw_dir), Path(cfg.paths.splits_dir))


if __name__ == "__main__":
    main()