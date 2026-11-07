#!/usr/bin/env python
"""Audit the exact split protocol: counts, distributions, leakage checks."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

proj = Path(__file__).resolve().parent.parent
splits = {s: pd.read_csv(proj / "data" / "splits" / f"{s}.csv")
          for s in ["train", "validation", "test"]}

print("=== counts ===")
for s, df in splits.items():
    print(f"  {s}: {len(df)}")

print("=== class distribution (counts) ===")
for s, df in splits.items():
    print(f"  {s}: {df['label'].value_counts().to_dict()}")

print("=== class distribution (proportions) ===")
for s, df in splits.items():
    vc = df["label"].value_counts(normalize=True).to_dict()
    print(f"  {s}: {dict(sorted(vc.items()))}")

print("=== lesion/group counts ===")
for s, df in splits.items():
    df["group_key"] = df["lesion_id"].fillna(df["image_id"] + "_own")
    print(f"  {s}: unique lesion groups = {df['group_key'].nunique()}")

print("=== cross-split IMAGE overlap (must be 0) ===")
all_ids = {s: set(df["image_id"]) for s, df in splits.items()}
for a in ["train", "validation", "test"]:
    for b in ["train", "validation", "test"]:
        if a < b:
            print(f"  {a} & {b}: {len(all_ids[a] & all_ids[b])}")

print("=== cross-split LESION/group overlap (must be 0) ===")
for s, df in splits.items():
    df["group_key"] = df["lesion_id"].fillna(df["image_id"] + "_own")
all_groups = {s: set(df["group_key"]) for s, df in splits.items()}
for a in ["train", "validation", "test"]:
    for b in ["train", "validation", "test"]:
        if a < b:
            print(f"  {a} & {b}: {len(all_groups[a] & all_groups[b])}")

print("=== images without lesion_id (own group) ===")
for s, df in splits.items():
    n = int(df["lesion_id"].isna().sum())
    print(f"  {s}: {n}")

print("=== split_summary.json ===")
p = proj / "data" / "splits" / "split_summary.json"
print(json.dumps(json.loads(p.read_text()), indent=2) if p.exists() else "missing")