"""
Data Splitting — ISIC 2019
==========================
Performs a lesion/patient-aware, class-stratified split BEFORE any
augmentation. Writes permanent split CSVs to data/splits/.

  train.csv
  validation.csv
  test.csv

Columns: image_id, image_path, label, lesion_id, sex, age_approx

The split is reproducible (fixed seed) and is NOT regenerated unless the
split files are deleted.
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ISIC_CLASSES

SEED = 42


def _resolve_gt_label(row, class_cols):
    """Return the single diagnostic class for a GT row (one-hot)."""
    for c in class_cols:
        if row.get(c, 0) == 1:
            return c
    return None


def create_splits(cfg, raw_dir: Path, splits_dir: Path):
    splits_dir.mkdir(parents=True, exist_ok=True)

    gt = pd.read_csv(raw_dir / cfg.dataset.train_gt_csv)
    meta = pd.read_csv(raw_dir / cfg.dataset.train_metadata_csv)

    class_cols = [c for c in ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"] if c in gt.columns]

    gt = gt.merge(meta, on="image", how="left")

    # Ground-truth label
    gt["label"] = gt.apply(lambda r: _resolve_gt_label(r, class_cols), axis=1)

    if cfg.split.drop_unk:
        before = len(gt)
        gt = gt.dropna(subset=["label"]).copy()
        print(f"  Dropped {before - len(gt):,} UNK/no-GT images")

    # Grouping key: lesion_id if available, else image id (no leakage)
    gt["group_key"] = gt["lesion_id"].fillna("img_" + gt["image"])

    print(f"  Total labeled images: {len(gt):,}")
    print(f"  Unique lesion/group keys: {gt['group_key'].nunique():,}")

    # ── Group-level stratified split ──
    groups = gt.groupby("group_key").agg(
        label=("label", "first"),
        n=("image", "count"),
    ).reset_index()

    label_of = dict(zip(groups["group_key"], groups["label"]))

    frac = cfg.split
    train_frac, val_frac, test_frac = frac.train_frac, frac.val_frac, frac.test_frac
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    rng = random.Random(SEED)
    keys = list(groups["group_key"])
    rng.shuffle(keys)

    n = len(keys)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    # Stratify by class within each bin (greedy class-aware assignment)
    # to keep class proportions balanced across splits.
    from collections import Counter, defaultdict

    buckets = {"train": [], "val": [], "test": []}
    bucket_counts = {b: Counter() for b in buckets}
    total_counts = Counter(label_of[k] for k in keys)
    max_per_class = {c: total_counts[c] for c in total_counts}
    targets = {
        "train": {c: int(total_counts[c] * train_frac) for c in total_counts},
        "val": {c: int(total_counts[c] * val_frac) for c in total_counts},
        "test": {c: total_counts[c] - int(total_counts[c] * train_frac)
                 - int(total_counts[c] * val_frac) for c in total_counts},
    }

    for k in keys:
        c = label_of[k]
        # Choose the bucket where this class is most "under target"
        deficits = {}
        for b in buckets:
            ratio = bucket_counts[b][c] / targets[b][c] if targets[b][c] > 0 else 1.0
            deficits[b] = ratio
        best = min(deficits, key=deficits.get)
        buckets[best].append(k)
        bucket_counts[best][c] += 1

    # ── Build per-split dataframes ──
    imgs = []
    for img_dir, split in [(raw_dir / cfg.dataset.train_input_dir, "train")]:
        _ = (img_dir, split)

    split_dfs = {}
    for split_name, group_keys in buckets.items():
        mask = gt["group_key"].isin(group_keys)
        df = gt[mask].copy()
        df["image_path"] = df["image"].apply(
            lambda i: str(raw_dir / cfg.dataset.train_input_dir / f"{i}.jpg"))
        df = df.rename(columns={"image": "image_id"})
        cols = ["image_id", "image_path", "label", "lesion_id", "group_key", "sex", "age_approx"]
        df = df[[c for c in cols if c in df.columns]]
        split_dfs[split_name] = df.reset_index(drop=True)

    # ── Verify no group overlaps across splits ──
    all_keys = {s: set(d["group_key"]) for s, d in split_dfs.items()}
    assert all_keys["train"].isdisjoint(all_keys["val"]), "train/val overlap!"
    assert all_keys["train"].isdisjoint(all_keys["test"]), "train/test overlap!"
    assert all_keys["val"].isdisjoint(all_keys["test"]), "val/test overlap!"
    print("  Verified: no lesion/group overlaps between splits.")

    # ── Save ──
    names = {"train": "train.csv", "val": "validation.csv", "test": "test.csv"}
    for split_name, df in split_dfs.items():
        path = splits_dir / names[split_name]
        df.drop(columns=["group_key"]).to_csv(path, index=False)
        print(f"  Saved {split_name}: {len(df):,} images -> {path}")

    # ── Distribution summary ──
    print()
    print(f"{'class':6s} {'train':>8s} {'val':>8s} {'test':>8s} {'total':>8s}")
    for c in ISIC_CLASSES:
        row = []
        for s in ["train", "val", "test"]:
            row.append(int((split_dfs[s]["label"] == c).sum()))
        print(f"{c:6s} {row[0]:8d} {row[1]:8d} {row[2]:8d} {sum(row):8d}")

    # Save split summary
    summary = {}
    for split_name, df in split_dfs.items():
        summary[split_name] = {
            "n_images": int(len(df)),
            "distribution": {c: int((df["label"] == c).sum()) for c in ISIC_CLASSES},
        }
    summary_path = splits_dir / "split_summary.json"
    import json
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved split summary: {summary_path}")

    return split_dfs


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.config import load_config
    cfg = load_config()
    create_splits(cfg, Path(cfg.dataset.raw_dir), Path(cfg.paths.splits_dir))