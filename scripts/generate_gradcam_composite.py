#!/usr/bin/env python
"""Generate paper-ready Grad-CAM composite from individual experiment artifacts."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PROJ = Path(r"")
SRC = PROJ / "experiments" / "resnet50_baseline" / "gradcam"
OUT = PROJ / "paper_results" / "resnet50_gradcam_examples.png"

# Order: 4 categories x 3 examples -> 12 panels
# Categories as saved by run_explain.py
groups = [
    ("high_conf_correct", 3, "High-conf Correct"),
    ("high_conf_incorrect", 3, "High-conf Incorrect"),
    ("low_conf_correct", 3, "Low-conf Correct"),
    ("low_conf_incorrect", 3, "Low-conf Incorrect"),
]

# Collect files in deterministic order
panels = []
labels = []
for prefix, n, title in groups:
    files = sorted(SRC.glob(f"{prefix}_*.png"))
    for f in files[:n]:
        panels.append(f)
        # parse true/pred from filename: e.g. high_conf_correct_0_idx0_trueNV_predNV.png
        parts = f.stem.split("_")
        # find true/pred tokens
        try:
            ti = parts.index("true")
            pi = parts.index("pred")
            true = parts[ti + 1]
            pred = parts[pi + 1]
            labels.append(f"{true}->{pred}")
        except Exception:
            labels.append(title)

# Layout: 4 rows x 3 cols = 12
fig, axes = plt.subplots(4, 3, figsize=(9, 12))
fig.suptitle("ResNet-50 Grad-CAM — representative examples\n(threshold locked on validation; heatmap = model attention)",
             fontsize=11, y=0.98)
for idx, (ax, f, lab) in enumerate(zip(axes.flat, panels, labels)):
    img = Image.open(f).convert("RGB")
    ax.imshow(img)
    ax.set_title(lab, fontsize=8)
    ax.axis("off")

# Row titles on left
row_titles = [g[2] for g in groups]
for r, title in enumerate(row_titles):
    axes[r, 0].text(-0.15, 0.5, title, transform=axes[r, 0].transAxes,
                    rotation=90, va="center", ha="center", fontsize=9, weight="bold")

plt.tight_layout(rect=[0.08, 0, 1, 0.96])
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved composite Grad-CAM to {OUT} ({OUT.stat().st_size/1024:.0f} KB) from {len(panels)} panels")
