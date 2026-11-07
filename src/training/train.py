"""
Training loop with two phases and full per-epoch logging.

Phase 1: backbone frozen, train head only, LR ~1e-3.
Phase 2: unfreeze last `fraction` of backbone, LR ~1e-5.

Saves:
  training_history.csv
  best_<model>.pt  (checkpoint)
  accuracy_curve.png / loss_curve.png / macro_f1_curve.png
  class_weights.json
"""
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.training.metrics import evaluate_epoch, compute_class_weights_from_loader
from src.utils import save_json


class EarlyStopping:
    def __init__(self, patience=6, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best = -np.inf if mode == "max" else np.inf
        self.counter = 0
        self.best_epoch = -1

    def __call__(self, value, epoch):
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.counter = 0
            self.best_epoch = epoch
            return True
        self.counter += 1
        return self.counter >= self.patience


def _build_optimizer(model, lr, weight_decay, name="adamw"):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)


HISTORY_FIELDS = ["phase", "epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy",
                  "val_precision_macro", "val_recall_macro", "val_f1_macro"]


def train_phase(model, loader, val_loader, device, cfg, phase_name, phase_lr,
                epochs, optimizer, criterion, class_weights, history_writer,
                num_classes, checkpoint_path, start_epoch=0, global_best=None):
    """Train a single phase. Returns final val metrics dict.

    Revision note (2026-11): the original protocol saves `checkpoint_path` whenever
    EarlyStopping.__call__ returns True, i.e. on improvement AND on the early-stopping
    epoch, so the restored weights are those of the stopping epoch. That behaviour is
    kept unchanged (it defines the published models). In addition, the genuinely best
    epoch across both phases (highest validation macro-F1) is saved to
    `best_true_<model>.pt` for a corrected-protocol sensitivity analysis.
    """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.training.reduce_lr_factor,
        patience=cfg.training.reduce_lr_patience)
    early_stop = EarlyStopping(patience=cfg.training.early_stop_patience, mode="max")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        train_metrics, _, _ = evaluate_epoch(model, loader, device, num_classes)
        val_metrics, _, _ = evaluate_epoch(model, val_loader, device, num_classes, criterion=criterion)

        global_epoch = start_epoch + epoch
        row = {
            "phase": phase_name,
            "epoch": global_epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_metrics.get("loss", float("nan")), 5),
            "train_accuracy": round(train_metrics["accuracy"], 5),
            "val_accuracy": round(val_metrics["accuracy"], 5),
            "val_precision_macro": round(val_metrics["precision_macro"], 5),
            "val_recall_macro": round(val_metrics["recall_macro"], 5),
            "val_f1_macro": round(val_metrics["f1_macro"], 5),
        }
        # (bug fix: a plain csv.writer given a dict wrote only the keys)
        history_writer.writerow([row[k] for k in HISTORY_FIELDS])
        print(f"  [{phase_name}] epoch {global_epoch:3d} | "
              f"tr_loss {train_loss:.4f} | val_loss {val_metrics.get('loss', float('nan')):.4f} | "
              f"tr_acc {train_metrics['accuracy']:.4f} | val_acc {val_metrics['accuracy']:.4f} | "
              f"val_f1_macro {val_metrics['f1_macro']:.4f}")

        # Checkpoint on best val macro-F1
        monitor = cfg.training.save_best_monitor
        current = val_metrics.get(monitor.replace("val_", ""), val_metrics["f1_macro"])
        if early_stop(current, global_epoch):
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    * best {monitor}: {early_stop.best:.4f} at epoch {global_epoch} -> saved {checkpoint_path}")
        if global_best is not None and current > global_best["value"]:
            global_best.update(value=current, epoch=global_epoch)
            true_path = Path(checkpoint_path).with_name(Path(checkpoint_path).name.replace("best_", "best_true_"))
            torch.save(model.state_dict(), true_path)
            print(f"    * true best {monitor}: {current:.4f} at epoch {global_epoch} -> saved {true_path}")

        scheduler.step(current)

        if early_stop.counter >= cfg.training.early_stop_patience:
            print(f"  Early stopping at epoch {global_epoch}")
            break

    return val_metrics


def train_model(cfg, model, train_loader, val_loader, class_names, device,
                backbone_name, exp_dir):
    """Full two-phase training. Saves history + checkpoint + class weights."""
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    num_classes = len(class_names)

    class_weights = compute_class_weights_from_loader(train_loader, num_classes, device)
    save_json({class_names[i]: float(w) for i, w in enumerate(class_weights.tolist())},
              exp_dir / "class_weights.json")
    print(f"  Class weights (from train split): {dict(zip(class_names, class_weights.tolist()))}")

    use_weights = bool(getattr(cfg.training, "use_class_weights", True))
    weight_arg = class_weights if use_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight_arg,
                                    label_smoothing=cfg.training.label_smoothing)
    checkpoint_path = exp_dir / f"best_{backbone_name}.pt"
    history_path = exp_dir / "training_history.csv"

    global_best = {"value": -np.inf, "epoch": -1}
    with open(history_path, "w", newline="", buffering=1) as f:
        writer = csv.writer(f)
        writer.writerow(HISTORY_FIELDS)

        # ── PHASE 1: frozen backbone ──
        model.freeze_backbone()
        optimizer = _build_optimizer(model, cfg.training.phase1_lr,
                                     cfg.training.weight_decay, cfg.training.optimizer)
        print("\n" + "=" * 62)
        print(f"  PHASE 1: Frozen backbone, LR={cfg.training.phase1_lr}")
        print("=" * 62)
        train_phase(model, train_loader, val_loader, device, cfg,
                    "phase1_frozen", cfg.training.phase1_lr,
                    cfg.training.phase1_epochs, optimizer, criterion,
                    class_weights, writer, num_classes, checkpoint_path,
                    start_epoch=0, global_best=global_best)

        # ── PHASE 2: fine-tune ──
        model.unfreeze_fraction(cfg.training.phase2_unfreeze_fraction)
        optimizer = _build_optimizer(model, cfg.training.phase2_lr,
                                     cfg.training.weight_decay, cfg.training.optimizer)
        print("\n" + "=" * 62)
        print(f"  PHASE 2: Unfreeze last {cfg.training.phase2_unfreeze_fraction:.0%}, LR={cfg.training.phase2_lr}")
        print("=" * 62)
        train_phase(model, train_loader, val_loader, device, cfg,
                    "phase2_finetune", cfg.training.phase2_lr,
                    cfg.training.phase2_epochs, optimizer, criterion,
                    class_weights, writer, num_classes, checkpoint_path,
                    start_epoch=cfg.training.phase1_epochs, global_best=global_best)

    save_json({"true_best_epoch": global_best["epoch"], "true_best_val_macro_f1": global_best["value"]},
              exp_dir / "true_best_epoch.json")

    # Reload best weights (original protocol: the last checkpoint written above)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"  Restored best checkpoint: {checkpoint_path}")

    return checkpoint_path