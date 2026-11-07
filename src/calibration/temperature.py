"""
Confidence Calibration — Temperature Scaling.
==============================================
Temperature T is optimized on the VALIDATION set only (never test).
Then calibrated probabilities are evaluated on the untouched test set.

Metrics: ECE, MCE, Brier score, NLL + reliability diagrams before/after.
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import softmax

from src.utils import save_json


# ── Calibration metrics ──
def compute_ece(y_true, probs, n_bins=15):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            ece += np.abs(accuracies[in_bin].mean() - confidences[in_bin].mean()) * in_bin.mean()
    return float(ece)


def compute_mce(y_true, probs, n_bins=15):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    max_gap = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            max_gap = max(max_gap, abs(accuracies[in_bin].mean() - confidences[in_bin].mean()))
    return float(max_gap)


def compute_brier(y_true, probs):
    y_onehot = np.eye(probs.shape[1])[y_true]
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def compute_nll(y_true, probs, eps=1e-12):
    probs = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true])))


def calibration_metrics(y_true, probs, n_bins=15):
    return {
        "ECE": compute_ece(y_true, probs, n_bins),
        "MCE": compute_mce(y_true, probs, n_bins),
        "Brier": compute_brier(y_true, probs),
        "NLL": compute_nll(y_true, probs),
    }


# ── Temperature scaling ──
class TemperatureScaler:
    def __init__(self, bounds=(0.1, 10.0)):
        self.bounds = bounds
        self.temperature = 1.0

    def _nll(self, T, logits, y_true):
        scaled = logits / T
        probs = softmax(scaled, axis=1)
        probs = np.clip(probs, 1e-12, 1 - 1e-12)
        return -np.mean(np.log(probs[np.arange(len(y_true)), y_true]))

    def fit(self, val_logits, val_y_true):
        res = minimize_scalar(self._nll, bounds=self.bounds,
                              args=(val_logits, val_y_true), method="bounded")
        self.temperature = float(res.x)
        return self.temperature

    def calibrate(self, logits):
        return softmax(logits / self.temperature, axis=1)


# ── Reliability diagram ──
def plot_reliability_diagram(y_true, probs, path, title="", n_bins=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    centers, accs, counts = [], [], []
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
            accs.append(accuracies[in_bin].mean())
            counts.append(in_bin.sum())

    ece = compute_ece(y_true, probs, n_bins)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax1.bar(centers, accs, width=1.0 / n_bins, alpha=0.5, color="#3498db", edgecolor="#2c3e50")
    ax1.plot(centers, accs, "o-", color="#e74c3c", lw=2, ms=6)
    ax1.set_title(f"{title}\nECE = {ece:.4f}")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.set_xlabel("Confidence"); ax1.set_ylabel("Accuracy")
    ax1.legend(loc="upper left"); ax1.grid(alpha=0.3)

    ax2.bar(centers, counts, width=1.0 / n_bins, alpha=0.7, color="#95a5a6", edgecolor="#2c3e50")
    ax2.set_xlim(0, 1); ax2.set_xlabel("Confidence"); ax2.set_ylabel("Count")

    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def run_calibration(cfg, model, val_loader, test_loader, class_names, device, exp_dir):
    """Full calibration pipeline: fit T on val, evaluate on test."""
    from src.evaluation.evaluate import get_predictions

    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    val_probs, val_logits, val_y_true = get_predictions(model, val_loader, device, return_logits=True)
    test_probs, test_logits, test_y_true = get_predictions(model, test_loader, device, return_logits=True)

    before = calibration_metrics(test_y_true, test_probs, cfg.calibration.ece_bins)

    scaler = TemperatureScaler(bounds=cfg.calibration.temperature_bounds)
    T = scaler.fit(val_logits, val_y_true)
    cal_probs = scaler.calibrate(test_logits)
    after = calibration_metrics(test_y_true, cal_probs, cfg.calibration.ece_bins)

    # Use ece_bins for all reliability figures so titles match canonical 15-bin ECE (Fig.5 caption)
    n_bins = cfg.calibration.ece_bins
    plot_reliability_diagram(test_y_true, test_probs,
                             exp_dir / "reliability_before.png",
                             title=f"{class_names[0] and 'Raw Softmax'} — Before Calibration", n_bins=n_bins)
    plot_reliability_diagram(test_y_true, cal_probs,
                             exp_dir / "reliability_after.png",
                             title=f"Temperature Scaling (T={T:.3f}) — After", n_bins=n_bins)

    results = {
        "temperature": T,
        "n_val_samples": int(len(val_y_true)),
        "n_test_samples": int(len(test_y_true)),
        "before_calibration": before,
        "after_calibration": after,
    }
    save_json(results, exp_dir / "calibration_metrics.json")

    # Side-by-side comparison figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, probs, title in [(axes[0], test_probs, f"Before (ECE={before['ECE']:.4f})"),
                                  (axes[1], cal_probs, f"After T={T:.3f} (ECE={after['ECE']:.4f})")]:
            conf = np.max(probs, axis=1); pred = np.argmax(probs, axis=1)
            acc = (pred == test_y_true).astype(float)
            bounds = np.linspace(0, 1, n_bins + 1)
            cs, ac = [], []
            for i in range(n_bins):
                m = (conf > bounds[i]) & (conf <= bounds[i + 1])
                if m.sum() > 0:
                    cs.append((bounds[i] + bounds[i + 1]) / 2); ac.append(acc[m].mean())
            ax.plot([0, 1], [0, 1], "k--")
            ax.plot(cs, ac, "o-", color="#e74c3c", lw=2)
            ax.set_title(title); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(exp_dir / "reliability_comparison.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [WARN] comparison figure failed: {e}")

    # Print
    print(f"\n  TEMPERATURE SCALING: T = {T:.4f} (fit on validation)")
    print(f"\n  {'metric':10s} {'before':>10s} {'after':>10s} {'delta':>10s}")
    for k in ["ECE", "MCE", "Brier", "NLL"]:
        b, a = before[k], after[k]
        print(f"  {k:10s} {b:10.4f} {a:10.4f} {a - b:+10.4f}")

    return results