#!/usr/bin/env python
"""
Run MC Dropout uncertainty analysis on a trained model's test split.
Usage:
    python scripts/run_uncertainty.py --model resnet50 --exp-id resnet50_baseline
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._common import load_config, get_device, resolve_backbone, MODEL_DISPLAY  # noqa: E402
from src.utils import seed_everything  # noqa: E402
from src.data.dataset import get_dataloaders  # noqa: E402
from src.models.builders import build_model  # noqa: E402
from src.uncertainty.mc_dropout import run_uncertainty_analysis  # noqa: E402
from src.config import ISIC_CLASSES  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--exp-id", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--passes", type=int, default=None)
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

    passes = args.passes or cfg.uncertainty.mc_dropout_passes

    train_loader, val_loader, test_loader = get_dataloaders(cfg, ISIC_CLASSES)
    model = build_model(backbone_name=backbone, num_classes=len(ISIC_CLASSES),
                        pretrained=True, dense_size=cfg.head.dense_size,
                        dropout=cfg.head.dropout,
                        use_batchnorm=cfg.head.use_batchnorm, device=device)
    import torch
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Load calibrated temperature if available
    temperature = 1.0
    cal_path = exp_dir / "calibration_metrics.json"
    if cal_path.exists() and cfg.uncertainty.use_temperature:
        import json
        with open(cal_path) as f:
            temperature = json.load(f).get("temperature", 1.0)
        print(f"  Using calibrated temperature T = {temperature:.4f}")

    print(f"\n  Running MC Dropout uncertainty ({passes} passes) on test split...")
    df, stats = run_uncertainty_analysis(
        cfg, model, test_loader, ISIC_CLASSES, device, exp_dir,
        temperature=temperature)
    print(f"\n  Saved: {exp_dir / 'uncertainty_results.csv'}")
    print(f"  Entropy mean: {stats['predictive_entropy']['mean']:.4f} "
          f"(correct {stats['predictive_entropy']['mean_correct']:.4f} vs "
          f"incorrect {stats['predictive_entropy']['mean_incorrect']:.4f})")


if __name__ == "__main__":
    main()