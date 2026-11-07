#!/usr/bin/env python
"""
Evaluate a trained model on the held-out test split.
Usage:
    python scripts/run_evaluate.py --model resnet50 --exp-id resnet50_baseline
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._common import load_config, get_device, resolve_backbone, MODEL_DISPLAY  # noqa: E402
from src.utils import seed_everything  # noqa: E402
from src.data.dataset import get_dataloaders  # noqa: E402
from src.models.builders import build_model  # noqa: E402
from src.evaluation.evaluate import evaluate_model, print_summary  # noqa: E402
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
    if not exp_dir.exists():
        print(f"[ERROR] Experiment dir not found: {exp_dir}")
        sys.exit(1)

    checkpoint = args.checkpoint or exp_dir / f"best_{backbone}.pt"
    if not Path(checkpoint).exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint}")
        sys.exit(1)

    _, _, test_loader = get_dataloaders(cfg, ISIC_CLASSES)
    model = build_model(backbone_name=backbone, num_classes=len(ISIC_CLASSES),
                        pretrained=True, dense_size=cfg.head.dense_size,
                        dropout=cfg.head.dropout,
                        use_batchnorm=cfg.head.use_batchnorm, device=device)
    import torch
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    print(f"\n  Evaluating {display} on test split...")
    metrics, pred_df = evaluate_model(cfg, model, test_loader, ISIC_CLASSES,
                                      device, exp_dir, split_name="test")
    print_summary(metrics, ISIC_CLASSES, f"{display} — TEST SPLIT")
    print(f"\n  Saved: {exp_dir / 'metrics.json'}, {exp_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()