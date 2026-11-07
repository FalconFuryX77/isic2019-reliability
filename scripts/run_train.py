#!/usr/bin/env python
"""
Train a baseline model on ISIC 2019.
Usage:
    python scripts/run_train.py --model resnet50
    python scripts/run_train.py --model efficientnet_b3
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._common import load_config, get_device, resolve_backbone, MODEL_DISPLAY  # noqa: E402
from src.utils import seed_everything, save_json, count_parameters  # noqa: E402
from src.experiment import create_experiment_dir, save_training_curves  # noqa: E402
from src.data.dataset import get_dataloaders  # noqa: E402
from src.models.builders import build_model  # noqa: E402
from src.training.train import train_model  # noqa: E402
from src.config import ISIC_CLASSES  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="resnet50 or efficientnet_b3")
    parser.add_argument("--config", default=None)
    parser.add_argument("--exp-id", default=None)
    parser.add_argument("--epochs1", type=int, default=None)
    parser.add_argument("--epochs2", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs1:
        cfg.training.phase1_epochs = args.epochs1
    if args.epochs2:
        cfg.training.phase2_epochs = args.epochs2

    backbone = resolve_backbone(args.model)
    display = MODEL_DISPLAY[backbone]
    exp_id = args.exp_id or f"{backbone}_baseline"
    seed_everything(cfg.seed)
    device = get_device()
    print(f"\n  Backbone: {display} | Device: {device} | Seed: {cfg.seed}")

    train_loader, val_loader, test_loader = get_dataloaders(cfg, ISIC_CLASSES)
    print(f"  Train: {len(train_loader.dataset):,} | "
          f"Val: {len(val_loader.dataset):,} | "
          f"Test: {len(test_loader.dataset):,}")

    model = build_model(backbone_name=backbone, num_classes=len(ISIC_CLASSES),
                        pretrained=True, dense_size=cfg.head.dense_size,
                        dropout=cfg.head.dropout,
                        use_batchnorm=cfg.head.use_batchnorm,
                        device=device)
    total, trainable = count_parameters(model)
    print(f"  Parameters: total={total:,} trainable={trainable:,}")

    exp_dir = create_experiment_dir(cfg, exp_id)
    checkpoint = train_model(cfg, model, train_loader, val_loader,
                             ISIC_CLASSES, device, backbone, exp_dir)

    # Save curves
    save_training_curves(exp_dir / "training_history.csv", exp_dir, display)

    print(f"\n  [OK] {display} training complete.")
    print(f"  Experiment dir: {exp_dir}")


if __name__ == "__main__":
    main()