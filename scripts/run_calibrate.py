#!/usr/bin/env python
"""
Run temperature scaling calibration on a trained model.
Usage:
    python scripts/run_calibrate.py --model resnet50 --exp-id resnet50_baseline
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._common import load_config, get_device, resolve_backbone, MODEL_DISPLAY  # noqa: E402
from src.utils import seed_everything  # noqa: E402
from src.data.dataset import get_dataloaders  # noqa: E402
from src.models.builders import build_model  # noqa: E402
from src.calibration.temperature import run_calibration  # noqa: E402
from src.config import ISIC_CLASSES  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--exp-id", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    backbone = resolve_backbone(args.model)
    display = MODEL_DISPLAY[backbone]
    seed_everything(cfg.seed)
    device = get_device()

    exp_id = args.exp_id or f"{backbone}_baseline"
    exp_dir = Path(cfg.paths.experiments_dir) / exp_id
    checkpoint = args.checkpoint or exp_dir / f"best_{backbone}.pt"
    if not Path(checkpoint).exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint}")
        sys.exit(1)

    train_loader, val_loader, test_loader = get_dataloaders(cfg, ISIC_CLASSES)
    model = build_model(backbone_name=backbone, num_classes=len(ISIC_CLASSES),
                        pretrained=True, dense_size=cfg.head.dense_size,
                        dropout=cfg.head.dropout,
                        use_batchnorm=cfg.head.use_batchnorm, device=device)
    import torch
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    print(f"\n  Calibrating {display} (T fitted on validation only)...")
    results = run_calibration(cfg, model, val_loader, test_loader,
                              ISIC_CLASSES, device, exp_dir)
    print(f"\n  Saved: {exp_dir / 'calibration_metrics.json'}")


if __name__ == "__main__":
    main()