"""
Dataset Audit — ISIC 2019
=========================
Inspects the raw ISIC 2019 data WITHOUT modifying it, and produces:

  dataset_audit.json
  dataset_class_distribution.csv
  dataset_class_distribution.png

plus a terminal summary. Reproducible and read-only.
"""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _file_hash(path, block_size=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def audit_dataset(raw_dir: str, output_dir: str):
    raw = Path(raw_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_gt_path = raw / "ISIC_2019_Training_GroundTruth.csv"
    train_meta_path = raw / "ISIC_2019_Training_Metadata.csv"
    test_gt_path = raw / "ISIC_2019_Test_GroundTruth.csv"
    test_meta_path = raw / "ISIC_2019_Test_Metadata.csv"
    train_img_dir = raw / "ISIC_2019_Training_Input"
    test_img_dir = raw / "ISIC_2019_Test_Input"

    report = {"dataset": "ISIC 2019", "generated": None}
    for p in [train_gt_path, train_meta_path, test_gt_path, test_meta_path]:
        report[p.stem] = {"exists": p.exists()}

    if not train_gt_path.exists():
        print("[ERROR] Training GroundTruth CSV missing. Cannot audit.")
        return

    train_gt = pd.read_csv(train_gt_path)
    train_meta = pd.read_csv(train_meta_path) if train_meta_path.exists() else None
    test_gt = pd.read_csv(test_gt_path) if test_gt_path.exists() else None
    test_meta = pd.read_csv(test_meta_path) if test_meta_path.exists() else None

    class_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

    # ── 1. Counts ──
    n_total_images = len(train_gt)
    n_labeled = int((train_gt[class_cols].sum(axis=1) > 0).sum())
    n_unk = int(train_gt["UNK"].sum()) if "UNK" in train_gt.columns else 0

    # ── 2. Class distribution (exclude UNK) ──
    label_counts = {}
    for c in class_cols:
        label_counts[c] = int(train_gt[c].sum())
    labeled_total = sum(label_counts[c] for c in class_cols[:-1])
    pct = {c: (label_counts[c] / labeled_total * 100 if labeled_total else 0.0)
           for c in class_cols[:-1]}

    # ── 3. Missing / malformed rows ──
    n_missing_gt = int((train_gt[class_cols].sum(axis=1) == 0).sum())

    # ── 4. Files on disk ──
    img_files = sorted(train_img_dir.glob("*.jpg")) if train_img_dir.exists() else []
    n_files = len(img_files)
    file_ids = {f.stem for f in img_files}

    n_gt_ids_not_on_disk = len(set(train_gt["image"]) - file_ids)
    n_files_not_in_gt = len(file_ids - set(train_gt["image"]))

    # ── 5. Duplicate filenames ──
    all_names = [f.name for f in img_files]
    dup_names = [k for k, v in Counter(all_names).items() if v > 1]

    # ── 6. Corrupted / unreadable images + dimensions/channels ──
    corrupted = []
    sizes = Counter()
    channels = Counter()
    sample = img_files[:500]  # hash + dims on a representative sample for speed
    hash_map = defaultdict(list)

    for f in img_files:
        try:
            with Image.open(f) as im:
                im.load()
                sizes[im.size] += 1
                channels[("RGB" if im.mode == "RGB" else im.mode)] += 1
        except Exception as e:
            corrupted.append({"file": f.name, "error": str(e)})

    # hash the sample (full dataset hashing is expensive)
    for f in sample:
        try:
            h = _file_hash(f)
            hash_map[h].append(f.name)
        except Exception:
            pass
    dup_hashes = {h: names for h, names in hash_map.items() if len(names) > 1}

    # ── 7. Metadata ──
    meta_summary = {}
    if train_meta is not None:
        meta_summary = {
            "rows": int(len(train_meta)),
            "columns": list(train_meta.columns),
            "lesion_id_nonempty": int(train_meta["lesion_id"].notna().sum()),
            "lesion_id_unique": int(train_meta["lesion_id"].dropna().nunique()),
            "sex_counts": train_meta["sex"].value_counts(dropna=False).to_dict(),
            "age_missing": int(train_meta["age_approx"].isna().sum()),
            "anatom_site_counts": train_meta["anatom_site_general"]
                .value_counts(dropna=False).to_dict(),
        }

    # ── 8. Test set summary (counts only; untouched) ──
    test_summary = {}
    if test_gt is not None:
        test_summary = {"rows": int(len(test_gt))}
        if test_meta is not None:
            test_summary["metadata_rows"] = int(len(test_meta))

    report.update({
        "train_images_in_gt": n_total_images,
        "train_images_labeled": n_labeled,
        "train_images_unk": n_unk,
        "train_images_no_gt": n_missing_gt,
        "files_on_disk_train": n_files,
        "gt_ids_missing_on_disk": n_gt_ids_not_on_disk,
        "files_not_in_gt": n_files_not_in_gt,
        "duplicate_filenames": len(dup_names),
        "corrupted_images": len(corrupted),
        "corrupted_sample": corrupted[:10],
        "image_size_counts": {f"{k[0]}x{k[1]}": v for k, v in sizes.most_common(10)},
        "image_mode_counts": {k: v for k, v in channels.most_common()},
        "duplicate_hash_groups": len(dup_hashes),
        "duplicate_hash_sample": {k: v for k, v in list(dup_hashes.items())[:5]},
        "metadata": meta_summary,
        "test_set": test_summary,
        "class_counts": label_counts,
        "class_percentages": pct,
    })

    # ── Save distribution CSV ──
    dist_df = pd.DataFrame({
        "class": class_cols[:-1],
        "count": [label_counts[c] for c in class_cols[:-1]],
        "percentage": [round(pct[c], 4) for c in class_cols[:-1]],
    })
    dist_csv = out / "dataset_class_distribution.csv"
    dist_df.to_csv(dist_csv, index=False)

    # ── Save JSON ──
    audit_json = out / "dataset_audit.json"
    with open(audit_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ── PNG distribution ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        classes = class_cols[:-1]
        counts = [label_counts[c] for c in classes]
        bars = ax.bar(classes, counts, color="steelblue", edgecolor="black")
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    f"{c:,}", ha="center", va="bottom", fontsize=8)
        ax.set_title("ISIC 2019 — Training Class Distribution")
        ax.set_xlabel("Class")
        ax.set_ylabel("Number of images")
        ax.tick_params(axis="x", rotation=0)
        plt.tight_layout()
        png_path = out / "dataset_class_distribution.png"
        plt.savefig(png_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [WARN] Could not create distribution PNG: {e}")

    # ── Terminal summary ──
    print("=" * 62)
    print("  ISIC 2019 DATASET AUDIT")
    print("=" * 62)
    print(f"  Images in training GT:    {n_total_images:,}")
    print(f"  Labeled images:           {n_labeled:,}")
    print(f"  UNK (no diagnosis):       {n_unk:,}")
    print(f"  Images missing GT:        {n_missing_gt:,}")
    print(f"  Files on disk (train):    {n_files:,}")
    print(f"  GT ids missing on disk:   {n_gt_ids_not_on_disk:,}")
    print(f"  Duplicate filenames:      {len(dup_names):,}")
    print(f"  Corrupted images:         {len(corrupted):,}")
    print(f"  Duplicate hash groups:    {len(dup_hashes):,}")
    print()
    print("  Class distribution (labeled only):")
    for c in class_cols[:-1]:
        print(f"    {c:5s}  {label_counts[c]:6,}  ({pct[c]:5.2f}%)")
    print()
    if meta_summary:
        print(f"  Metadata rows:            {meta_summary.get('rows', 0):,}")
        print(f"  lesion_id nonempty:       {meta_summary.get('lesion_id_nonempty', 0):,}")
        print(f"  lesion_id unique:         {meta_summary.get('lesion_id_unique', 0):,}")
    print()
    print(f"  Saved: {audit_json}")
    print(f"  Saved: {dist_csv}")
    print(f"  Saved: {png_path if 'png_path' in dir() else dist_csv}")
    print("=" * 62)

    return report


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.config import load_config
    cfg = load_config()
    audit_dataset(cfg.dataset.raw_dir, cfg.paths.results_dir)