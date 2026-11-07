"""
Comprehensive model evaluation.
================================
Computes and saves:
  predictions.csv               (per-sample probabilities + labels)
  metrics.json                  (aggregate metrics)
  classification_report.csv     (per-class precision/recall/f1)
  confusion_matrix.csv
  confusion_matrix.png
  roc_curves.png / pr_curves.png
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve, auc)

from src.utils import save_json


@torch.no_grad()
def get_predictions(model, loader, device, return_logits=False):
    """Return (probs, y_true) — or (probs, logits, y_true) if return_logits.

    Raw logits are required for temperature scaling.
    """
    model.eval()
    all_probs, all_logits, all_labels = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    if return_logits:
        return probs, np.concatenate(all_logits), labels
    return probs, labels


def compute_all_metrics(y_true, probs, class_names):
    """Compute every required metric from predicted probabilities."""
    y_pred = np.argmax(probs, axis=1)
    n_classes = len(class_names)

    macro = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # Per-class
    report = {}
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(n_classes)))
    per_class_rec = recall_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(n_classes)))
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(n_classes)))
    for i, c in enumerate(class_names):
        report[c] = {
            "precision": float(per_class_prec[i]),
            "recall": float(per_class_rec[i]),
            "f1": float(per_class_f1[i]),
            "support": int(np.sum(y_true == i)),
        }

    # One-vs-rest ROC-AUC and PR-AUC
    roc_auc = {}
    pr_auc = {}
    for i, c in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        try:
            roc_auc[c] = float(roc_auc_score(y_bin, probs[:, i]))
        except Exception:
            roc_auc[c] = float("nan")
        try:
            pr_auc[c] = float(average_precision_score(y_bin, probs[:, i]))
        except Exception:
            pr_auc[c] = float("nan")
        report[c]["auc_roc"] = roc_auc[c]
        report[c]["auc_pr"] = pr_auc[c]

    macro["auc_roc_macro"] = float(np.nanmean(list(roc_auc.values())))
    macro["auc_pr_macro"] = float(np.nanmean(list(pr_auc.values())))

    return {"macro": macro, "per_class": report, "confusion_matrix": cm.tolist()}


def save_predictions_csv(df, exp_dir):
    path = Path(exp_dir) / "predictions.csv"
    df.to_csv(path, index=False)
    return path


def evaluate_model(cfg, model, loader, class_names, device, exp_dir,
                   split_name="test", df_meta=None):
    """Run full evaluation and save all artifacts."""
    probs, y_true = get_predictions(model, loader, device)
    y_pred = np.argmax(probs, axis=1)
    metrics = compute_all_metrics(y_true, probs, class_names)

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # predictions.csv
    rows = []
    n = len(y_true)
    for i in range(n):
        row = {
            "split": split_name,
            "true_label": class_names[y_true[i]],
            "predicted_label": class_names[y_pred[i]],
            "correct": int(y_true[i] == y_pred[i]),
            "max_confidence": float(probs[i].max()),
        }
        for j, c in enumerate(class_names):
            row[f"prob_{c}"] = float(probs[i, j])
        rows.append(row)
    pred_df = pd.DataFrame(rows)
    save_predictions_csv(pred_df, exp_dir)

    # metrics.json
    save_json(metrics, exp_dir / "metrics.json")

    # classification_report.csv
    rep_rows = []
    for c, d in metrics["per_class"].items():
        rep_rows.append({"class": c, **d})
    pd.DataFrame(rep_rows).to_csv(exp_dir / "classification_report.csv", index=False)

    # confusion_matrix.csv
    pd.DataFrame(metrics["confusion_matrix"], index=class_names, columns=class_names).to_csv(
        exp_dir / "confusion_matrix.csv")

    # ── Figures ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Confusion matrix
        fig, ax = plt.subplots(figsize=(9, 7))
        cm = np.array(metrics["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.figure.colorbar(im, ax=ax)
        ax.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names)), labels=class_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        ax.set_title(f"Confusion Matrix ({split_name})")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        plt.tight_layout()
        plt.savefig(exp_dir / "confusion_matrix.png", dpi=150)
        plt.close()

        # ROC + PR curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for i, c in enumerate(class_names):
            y_bin = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, probs[:, i])
            precision, recall_, _ = precision_recall_curve(y_bin, probs[:, i])
            axes[0].plot(fpr, tpr, label=f"{c} ({metrics['per_class'][c]['auc_roc']:.3f})")
            axes[1].plot(recall_, precision, label=f"{c} ({metrics['per_class'][c]['auc_pr']:.3f})")
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[0].set_title("One-vs-Rest ROC"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
        axes[0].legend(fontsize=7, loc="lower right")
        axes[1].set_title("One-vs-Rest PR"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
        axes[1].legend(fontsize=7, loc="lower left")
        plt.tight_layout()
        plt.savefig(exp_dir / "roc_pr_curves.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [WARN] figure generation failed: {e}")

    return metrics, pred_df


def print_summary(metrics, class_names, title="EVALUATION"):
    m = metrics["macro"]
    print(f"\n  {'=' * 62}")
    print(f"  {title}")
    print(f"  {'=' * 62}")
    for k, v in m.items():
        print(f"    {k:<24} {v:.4f}")
    print()
    print(f"  {'class':6s} {'precision':>10s} {'recall':>8s} {'f1':>8s} {'support':>8s}")
    for c in class_names:
        d = metrics["per_class"][c]
        print(f"  {c:6s} {d['precision']:10.4f} {d['recall']:8.4f} {d['f1']:8.4f} {d['support']:8d}")
    print(f"  {'=' * 62}")