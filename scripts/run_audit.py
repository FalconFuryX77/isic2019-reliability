#!/usr/bin/env python
"""
Run the ISIC 2019 dataset audit (read-only).
Usage:
    python scripts/run_audit.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._common import load_config  # noqa: E402
from src.data.audit import audit_dataset  # noqa: E402


def main():
    cfg = load_config()
    print("Auditing raw ISIC 2019 data...")
    report = audit_dataset(cfg.dataset.raw_dir, cfg.paths.results_dir)
    return report


if __name__ == "__main__":
    main()