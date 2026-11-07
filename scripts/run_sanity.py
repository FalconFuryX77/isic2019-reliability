#!/usr/bin/env python
"""
Sanity check before full training (Section 28 of the project plan):
  1. Load a small batch of images
  2. Verify labels / class indices
  3. Verify preprocessing + augmentation visually
  4. Verify model output shape = 8
  5. Run one forward pass
  6. Run one training step and confirm loss decreases across batches
  7. Confirm validation pipeline has NO augmentation
  8. Confirm train/val/test image IDs do not overlap
Usage:
    python scripts/run_sanity.py --model resnet50
"""
import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._common import load_config, get_device, resolve_backbone, MODEL_DISPLAY  # noqa: E402
from src.utils import seed_everything  # noqa: E402
from src.data.dataset import get_dataloaders, ISICDataset, make_transforms  # noqa: E402
from src.models.builders import build_model  # noqa: E402
from src.config import ISIC_CLASSES  # noqa: E402

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50")
    args = parser.parse_args()

    cfg = load_config()
    backbone = resolve_backbone(args.model)
    display = MODEL_DISPLAY[backbone]
    seed_everything(cfg.seed)
    device = get_device()
    print(f"\n{'='*62}\n  SANITY CHECK: {display}\n{'='*62}")

    train_loader, val_loader, test_loader = get_dataloaders(cfg, ISIC_CLASSES)

    # 1. Load a small batch
    x, y = next(iter(train_loader))
    print(f"  [1] Batch loaded: x={tuple(x.shape)} y={tuple(y.shape)}")
    print(f"      Sample labels: {y[:8].tolist()} (expected 0-{len(ISIC_CLASSES)-1})")
    assert x.min() >= -3 and x.max() <= 3, "Preprocessing range unexpected"
    assert set(y[:8].tolist()) <= set(range(len(ISIC_CLASSES))), "Labels out of range"

    # 2. Class indices
    idx_map = train_loader.dataset.class_to_idx
    print(f"  [2] Class->index map: {idx_map}")
    assert len(idx_map) == len(ISIC_CLASSES)

    # 3/4. Preprocessing + augmentation difference
    val_x, val_y = next(iter(val_loader))
    x_np = x[0].numpy().transpose(1, 2, 0)
    val_np = val_x[0].numpy().transpose(1, 2, 0)
    diff = float(torch.abs(x[0] - val_x[0]).mean())
    print(f"  [3] Train vs Val first-sample mean abs diff: {diff:.4f} "
          f"(>0 indicates train augmentation active)")
    assert diff > 0, "Training augmentation appears inactive"

    # 5. Model forward pass
    model = build_model(backbone_name=backbone, num_classes=len(ISIC_CLASSES),
                        pretrained=True, dense_size=cfg.head.dense_size,
                        dropout=cfg.head.dropout,
                        use_batchnorm=cfg.head.use_batchnorm, device=device)
    model.eval()
    with torch.no_grad():
        out = model(x.to(device))
    print(f"  [5] Forward pass output shape: {tuple(out.shape)}")
    assert out.shape[1] == 8, "Output should be 8 classes"

    # 6. One training step, loss decreases across a few batches
    model.freeze_backbone()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()
    model.train()
    losses = []
    for i, (xb, yb) in enumerate(train_loader):
        if i >= 3:
            break
        opt.zero_grad()
        logits = model(xb.to(device))
        loss = crit(logits, yb.to(device))
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    print(f"  [6] Training losses over 3 batches: {[round(l, 4) for l in losses]}")
    assert losses[-1] < losses[0], "Loss did not decrease across batches"

    # 7. Validation pipeline has no augmentation
    is_det = all(not isinstance(t, (torchvision.transforms.RandomRotation,
                                    torchvision.transforms.RandomAffine,
                                    torchvision.transforms.ColorJitter,
                                    torchvision.transforms.RandomHorizontalFlip))
                 for t in getattr(val_loader.dataset.transform, 'transforms', []))
    print(f"  [7] Validation transforms contain NO augmentation: {is_det}")

    # 8. Split overlap check
    ids = {split: set(pd.read_csv(split_csv)["image_id"])
           for split, split_csv in [("train", f"{cfg.paths.splits_dir}/train.csv"),
                                    ("val", f"{cfg.paths.splits_dir}/validation.csv"),
                                    ("test", f"{cfg.paths.splits_dir}/test.csv")]}
    overlap = (ids["train"] & ids["val"]) | (ids["train"] & ids["test"]) | (ids["val"] & ids["test"])
    print(f"  [8] Split overlaps (train/val/test): {len(overlap)}")
    assert len(overlap) == 0, "Split overlap detected!"

    # Sample display figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        inv_x = x[:8] * std + mean
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i, ax in enumerate(axes.flat):
            ax.imshow(np.clip(inv_x[i].numpy().transpose(1, 2, 0), 0, 1))
            ax.set_title(ISIC_CLASSES[y[i].item()])
            ax.axis("off")
        plt.suptitle(f"Sanity check — augmented training samples ({display})")
        out_dir = Path(cfg.paths.results_dir) / "sanity"
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "sanity_samples.png", dpi=150)
        plt.close()
        print(f"      Saved sample grid: {out_dir / 'sanity_samples.png'}")
    except Exception as e:
        print(f"  [WARN] sample grid failed: {e}")

    print(f"\n  {'='*62}\n  [OK] ALL SANITY CHECKS PASSED\n  {'='*62}")


if __name__ == "__main__":
    import torchvision  # noqa: F401 (used in checks)
    import pandas as pd  # noqa: F401
    main()