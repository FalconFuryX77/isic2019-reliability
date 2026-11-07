#!/usr/bin/env python
"""
Run selective prediction / abstention analysis on a trained model.
Usage:
    python scripts/run_selective.py --model resnet50 --exp-id resnet50_baseline
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._common import load_config, get_device, resolve_backbone, MODEL_DISPLAY  # noqa: E402
from src.utils import seed_everything  # noqa: E402
from src.data.dataset import get_dataloaders  # noqa: E402
from src.models.builders import build_model  # noqa: E402
from src.evaluation.selective import run_selective_analysis  # noqa: E402
from src.evaluation.evaluate import get_predictions  # noqa: E402
from src.config import ISIC_CLASSES  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--exp-id", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--calibrated", action="store_true",
                        help="use calibrated probabilities if available")
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

    val_probs, val_y_true = get_predictions(model, val_loader, device)
    test_probs, test_y_true = get_predictions(model, test_loader, device)

    # Apply calibrated temperature if requested (to BOTH val and test)
    if args.calibrated:
        cal_path = exp_dir / "calibration_metrics.json"
        if cal_path.exists():
            import json
            from scipy.special import softmax
            with open(cal_path) as f:
                T = json.load(f).get("temperature", 1.0)
            _, val_logits, val_y_true = get_predictions(model, val_loader, device, return_logits=True)
            _, test_logits, test_y_true = get_predictions(model, test_loader, device, return_logits=True)
            val_probs = softmax(val_logits / T, axis=1)
            test_probs = softmax(test_logits / T, axis=1)
            print(f"  Using calibrated probabilities (T={T:.4f})")

    print(f"\n  Selective prediction: thresholds locked on VALIDATION, applied to TEST...")
    df, summary = run_selective_analysis(
        cfg, val_probs, val_y_true, test_probs, test_y_true,
        exp_dir, target_coverages=(0.85, 0.90, 0.95))
    print(f"\n  Saved: {exp_dir / 'selective_protocol.csv'}, "
          f"{exp_dir / 'coverage_vs_risk.png'}")


if __name__ == "__main__":
    main()