#!/usr/bin/env python
"""Regenerate IEEE-paper-ready Grad-CAM composite: 4 examples x 3 panels, 2x2 layout."""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import load_config, get_device
from src.data.dataset import get_dataloaders
from src.config import ISIC_CLASSES
from src.models.builders import build_model
from src.explainability.gradcam import gradcam, overlay_heatmap
from src.data.dataset import make_transforms, IMAGENET_MEAN, IMAGENET_STD

PROJ = Path(r"")
EXP = PROJ / "experiments" / "resnet50_baseline"
OUT = PROJ / "paper_results" / "resnet50_gradcam_examples.png"

REPRS = [
    ("high_conf_correct", 0),
    ("high_conf_incorrect", 19),
    ("low_conf_correct", 5),
    ("low_conf_incorrect", 11),
]

def main():
    cfg = load_config()
    cfg.training.num_workers = 0
    device = get_device()
    model = build_model(backbone_name="resnet50", num_classes=len(ISIC_CLASSES),
                        pretrained=True, dense_size=cfg.head.dense_size,
                        dropout=cfg.head.dropout, use_batchnorm=cfg.head.use_batchnorm, device=device)
    ckpt = EXP / "best_resnet50.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    from src.models.builders import get_gradcam_target_layer
    target_layer = get_gradcam_target_layer(model)

    from src.evaluation.evaluate import get_predictions
    _, val_loader, test_loader = get_dataloaders(cfg, ISIC_CLASSES)
    probs, y_true = get_predictions(model, test_loader, device)
    y_pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    dataset = test_loader.dataset
    norm = make_transforms(cfg.preprocessing.image_size, augment=False,
                           mean=IMAGENET_MEAN, std=IMAGENET_STD)

    fig = plt.figure(figsize=(7.8, 5.2))
    gs = gridspec.GridSpec(2, 7, width_ratios=[1, 1, 1, 0.12, 1, 1, 1],
                           wspace=0.04, hspace=0.35,
                           left=0.02, right=0.98, top=0.86, bottom=0.06)

    cat_titles = {
        "high_conf_correct": "High-confidence Correct",
        "high_conf_incorrect": "High-confidence Incorrect",
        "low_conf_correct": "Low-confidence Correct",
        "low_conf_incorrect": "Low-confidence Incorrect",
    }
    panel_titles = ["Original", "Grad-CAM", "Overlay"]

    axes_grid = []
    for r in range(2):
        row = []
        for c in range(7):
            ax = fig.add_subplot(gs[r, c])
            row.append(ax)
        axes_grid.append(row)

    for r in range(2):
        axes_grid[r][3].axis("off")

    for ex_idx, (cat, idx) in enumerate(REPRS):
        r = ex_idx // 2
        left_col = 0 if ex_idx % 2 == 0 else 4
        axes = [axes_grid[r][left_col + i] for i in range(3)]

        img_path = dataset.paths[idx]
        true_c = ISIC_CLASSES[y_true[idx]]
        pred_c = ISIC_CLASSES[y_pred[idx]]
        cval = float(conf[idx])

        pil_img = Image.open(img_path).convert("RGB")
        input_tensor = norm(pil_img).unsqueeze(0).to(device)
        heatmap, _ = gradcam(model, input_tensor, target_layer, target_class=int(y_pred[idx]), method=cfg.explainability.method)

        resized_np = np.array(pil_img.resize((cfg.preprocessing.image_size, cfg.preprocessing.image_size), Image.BICUBIC)).astype(np.float32)
        resized_uint8 = resized_np.astype(np.uint8)
        overlay = overlay_heatmap(resized_np, heatmap)

        panels = [
            (resized_uint8, None),
            (heatmap, "jet"),
            (overlay, None),
        ]
        for ax, (data, cmap) in zip(axes, panels):
            if cmap == "jet":
                ax.imshow(data, cmap="jet", vmin=0, vmax=1)
            else:
                ax.imshow(data)
            ax.axis("off")

        for ax, ttl in zip(axes, panel_titles):
            ax.set_title(ttl, fontsize=8.5, pad=3, weight="bold")

        axes[1].text(0.5, 1.32, f"{cat_titles[cat]}", transform=axes[1].transAxes,
                     ha="center", va="bottom", fontsize=9.5, weight="bold", linespacing=1.2)
        axes[1].text(0.5, 1.18, f"True: {true_c}  \u2192  Pred: {pred_c}  (conf {cval:.2f})",
                     transform=axes[1].transAxes, ha="center", va="bottom", fontsize=7.5, style="italic", color="#333333")

    plt.savefig(OUT, dpi=350, bbox_inches="tight")
    print(f"Saved IEEE Grad-CAM composite to {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"Figure size: {fig.get_size_inches()} inches, dpi 350")

if __name__ == "__main__":
    main()
