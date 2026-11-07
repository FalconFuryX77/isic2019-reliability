#!/usr/bin/env python
"""Revised methodology diagram showing which data split feeds which component."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_revision" / "Images" / "isic_methodology_diagram_v2.png"

C = {"data": "#e8eef7", "train": "#fde9d9", "val": "#e2efda", "test": "#ededed", "off": "#fff2cc", "edge": "#404040"}
fig, ax = plt.subplots(figsize=(13, 4.6))
ax.set_xlim(0, 130)
ax.set_ylim(0, 46)
ax.axis("off")


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                fc=color, ec=C["edge"], lw=1.0))
    ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=11.5, fontweight="bold")
    ax.text(x + w / 2, y + h - 4.8, body, ha="center", va="top", fontsize=9.6, linespacing=1.3)


def arrow(x0, y0, x1, y1, label=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color=C["edge"], shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.8, label, ha="center", va="bottom", fontsize=9, style="italic")


# data
box(1, 30, 20, 14, "ISIC 2019 labelled", "25,331 images, 8 classes\nHAM10000, BCN_20000,\nMSK", C["data"])
box(25, 30, 24, 14, "Lesion-grouped split", "lesion IDs (91.8% of images)\nexact + near-duplicate audit\nclass-stratified 80/10/10", C["data"])
box(53, 37, 15, 7, "Train", "20,235", C["train"])
box(53, 28.5, 15, 7, "Validation", "2,573", C["val"])
box(53, 20, 15, 7, "Internal test", "2,523", C["test"])
box(1, 3, 28, 14, "Official ISIC 2019 test", "8,238 images incl. 2,047 UNK\nnever used in development\n(shift + out-of-distribution)", C["off"])
arrow(21, 37, 25, 37)
arrow(49, 40.5, 53, 40.5)
arrow(49, 34, 53, 32)
arrow(49, 31.5, 53, 23.5)

# training
box(73, 34, 26, 11.5, "Training", "ResNet-50: plain (E0) and\nclass-weighted (E1) CE, 5 seeds\nEfficientNet-B3: E1, 3 seeds", C["train"])
arrow(68, 40.5, 73, 40.5)
# validation-only fitting
box(73, 14, 26, 18.5, "Fitted on validation only", "early stopping / checkpoint\narchitecture choice (macro-F1)\ntemperature, vector scaling,\nprior correction\nthresholds $\\tau$, SGR, conformal $q$\nscore combination", C["val"])
arrow(68, 32, 73, 27, "val.")
arrow(86, 34, 86, 32.8)
# evaluation
box(104, 14, 25, 31, "Locked evaluation", "discrimination\n(acc., bal. acc., macro-F1, AUC)\n\ncalibration\n(ECE, aECE, cwECE, Brier, NLL)\n\nerror detection\n(MSP, entropy, MC-Dropout,\nensembles; AUROC, AURC)\n\nselective + conformal prediction\n\nGrad-CAM sanity checks", C["test"])
arrow(99, 23, 104, 23)
ax.plot([68, 70.5, 70.5, 101.5, 101.5], [23.5, 23.5, 11, 11, 18], color=C["edge"], lw=1.1)
arrow(101.5, 18, 104, 18)
ax.plot([29, 101.5, 101.5], [8, 8, 11], color=C["edge"], lw=1.1)
ax.text(65, 9, "applied once with all components locked", fontsize=9, style="italic", ha="center")
ax.text(84, 12, "internal test (secondary evaluation)", fontsize=9, style="italic", ha="center")
plt.tight_layout()
plt.savefig(OUT, dpi=220, bbox_inches="tight")
print("saved", OUT)
