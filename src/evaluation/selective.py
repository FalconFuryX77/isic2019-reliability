"""
Selective Prediction / Abstention (methodologically correct protocol).
=======================================================================
Protocol (no test-set tuning):
  1. Calibrated probabilities are computed for VALIDATION and TEST.
  2. Confidence thresholds that hit target coverage levels are selected
     on the VALIDATION set ONLY.
  3. Those locked thresholds are applied to the TEST set and reported
     (coverage, selective accuracy, risk) without further adjustment.

Stores selective_prediction.csv, coverage_vs_risk.png, selective_summary.json.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import save_json


def _coverage_df(conf, correct):
    """Coverage/acc/risk for a grid of thresholds on one split."""
    thresholds = np.linspace(0.0, 1.0, 101)
    rows = []
    for th in thresholds:
        accepted = conf >= th
        cov = float(accepted.mean())
        if accepted.sum() > 0:
            acc = float(correct[accepted].mean())
            rows.append({
                "threshold": round(float(th), 4),
                "coverage": round(cov, 4),
                "selective_accuracy": round(acc, 4),
                "risk": round(1.0 - acc, 4),
                "rejection_rate": round(1.0 - cov, 4),
                "n_accepted": int(accepted.sum()),
            })
    return pd.DataFrame(rows)


def _threshold_for_coverage(conf, target_coverage):
    """Largest threshold whose coverage >= target (on a given split)."""
    targets = np.linspace(0.0, 1.0, 101)
    below = targets[targets <= target_coverage]
    if len(below) == 0:
        return 0.0
    best = below[np.argmin(np.abs(below - target_coverage))]
    # find threshold achieving at least best coverage
    ths = np.linspace(0.0, 1.0, 1001)
    coverages = np.array([float((conf >= t).mean()) for t in ths])
    valid = coverages >= best
    if not valid.any():
        return 0.0
    return float(ths[np.where(valid)[0][-1]])


def run_selective_analysis(cfg, val_probs, val_y_true, test_probs, test_y_true,
                           exp_dir, target_coverages=(0.85, 0.90, 0.95)):
    """Select thresholds on validation, evaluate locked thresholds on test."""
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    val_conf = np.max(val_probs, axis=1)
    val_pred = np.argmax(val_probs, axis=1)
    val_correct = (val_pred == val_y_true).astype(float)

    test_conf = np.max(test_probs, axis=1)
    test_pred = np.argmax(test_probs, axis=1)
    test_correct = (test_pred == test_y_true).astype(float)

    # ── 1. Full curve per split (for plotting) ──
    val_df = _coverage_df(val_conf, val_correct)
    test_df = _coverage_df(test_conf, test_correct)
    test_df.to_csv(exp_dir / "selective_prediction.csv", index=False)

    # ── 2. Select thresholds on validation, lock, evaluate on test ──
    summary = {}
    rows_detail = []
    for target in target_coverages:
        th = _threshold_for_coverage(val_conf, target)
        # Validate that threshold achieves target coverage on val
        val_cov = float((val_conf >= th).mean())
        # Apply locked threshold to test
        test_mask = test_conf >= th
        test_cov = float(test_mask.mean())
        test_acc = float(test_correct[test_mask].mean()) if test_mask.sum() > 0 else float("nan")
        val_acc = float(val_correct[val_conf >= th].mean()) if (val_conf >= th).sum() > 0 else float("nan")
        summary[f"cov{int(target*100)}"] = {
            "target_coverage": float(target),
            "locked_threshold": round(th, 4),
            "val_coverage": round(val_cov, 4),
            "val_accuracy": round(val_acc, 4) if not np.isnan(val_acc) else None,
            "test_coverage": round(test_cov, 4),
            "test_accuracy": round(test_acc, 4) if not np.isnan(test_acc) else None,
            "test_risk": round(1.0 - test_acc, 4) if not np.isnan(test_acc) else None,
            "test_n_accepted": int(test_mask.sum()),
            "test_n_rejected": int((~test_mask).sum()),
        }
        rows_detail.append({
            "split": "val", "target_coverage": float(target), "locked_threshold": round(th, 4),
            "coverage": round(val_cov, 4), "selective_accuracy": round(val_acc, 4),
            "risk": round(1.0 - val_acc, 4),
        })
        rows_detail.append({
            "split": "test", "target_coverage": float(target), "locked_threshold": round(th, 4),
            "coverage": round(test_cov, 4), "selective_accuracy": round(test_acc, 4),
            "risk": round(1.0 - test_acc, 4),
        })
    pd.DataFrame(rows_detail).to_csv(exp_dir / "selective_protocol.csv", index=False)
    save_json({"protocol": "threshold selected on VALIDATION, locked, applied to TEST",
               "summary": summary}, exp_dir / "selective_summary.json")

    # ── 3. Coverage vs risk plot (test curve) ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(8, 6))
        valid = test_df.dropna(subset=["risk"])
        ax1.plot(valid["coverage"], valid["risk"], "o-", color="#e74c3c", lw=2, label="Test curve")
        for tgt, s in summary.items():
            ax1.plot(s["test_coverage"], s["test_risk"], "o", color="#2c3e50", ms=8,
                     label=f"locked th={s['locked_threshold']:.2f}")
        ax1.axvline(0.85, color="gray", ls="--", alpha=0.5)
        ax1.set_xlabel("Coverage (fraction accepted)"); ax1.set_ylabel("Risk (1 - selective accuracy)")
        ax1.set_title("Selective Prediction — Coverage vs Risk (test, thresholds locked from val)")
        ax1.legend(); ax1.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(exp_dir / "coverage_vs_risk.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [WARN] selective plot failed: {e}")

    print(f"\n  SELECTIVE PREDICTION (threshold selected on VALIDATION, applied to TEST):")
    for k, s in summary.items():
        print(f"    {k:6s} target_cov={s['target_coverage']:.2f} th={s['locked_threshold']:.3f} | "
              f"val cov={s['val_coverage']:.3f} | test cov={s['test_coverage']:.3f} "
              f"test acc={s['test_accuracy']:.3f} risk={s['test_risk']:.3f} "
              f"(accepted {s['test_n_accepted']}/{s['test_n_accepted'] + s['test_n_rejected']})")
    return test_df, summary