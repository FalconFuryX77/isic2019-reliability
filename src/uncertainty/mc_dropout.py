"""
Uncertainty Estimation.
=======================
Methods:
  1. Predictive entropy of the mean MC probability
  2. Max softmax probability (confidence)
  3. MC Dropout — N stochastic forward passes with dropout enabled
     -> mean predictive probability, predictive entropy,
        mutual information (epistemic uncertainty), predictive variance.

Stores uncertainty_results.csv and uncertainty_analysis.png.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils import save_json


def predictive_entropy(probs, eps=1e-12, axis=-1):
    """Entropy of a probability distribution (per sample), nat units."""
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=axis)


def run_mc_dropout(model, loader, device, n_passes=30, temperature=1.0,
                   return_all=False):
    """Perform MC dropout inference.

    Returns dict with:
      mc_mean_probs   (N, C)
      mc_std_probs    (N, C)
      predictive_entropy  (N,)
      mutual_info         (N,)
      predictive_variance (N,)
    """
    model.eval()
    model.enable_dropout(True)
    all_pass_probs = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        pass_probs = []
        for _ in range(n_passes):
            with torch.no_grad():
                logits = model(x)
            probs = torch.softmax(logits / temperature, dim=1).cpu().numpy()
            pass_probs.append(probs)
        all_pass_probs.append(np.stack(pass_probs, axis=0))  # (passes, batch, C)

    model.enable_dropout(False)
    model.eval()

    pass_probs = np.concatenate(all_pass_probs, axis=1)  # (passes, N, C)
    mean_probs = pass_probs.mean(axis=0)
    std_probs = pass_probs.std(axis=0)

    # Predictive entropy of the mean
    H_pred = predictive_entropy(mean_probs)

    # Expected entropy (average per-pass entropy)
    H_exp = predictive_entropy(pass_probs).mean(axis=0)

    # Mutual information = H[mean] - E[H[pass]]
    mutual_info = H_pred - H_exp

    # Predictive variance = mean across classes of per-class variance
    pred_variance = std_probs.mean(axis=1)

    results = {
        "mc_mean_probs": mean_probs,
        "mc_std_probs": std_probs,
        "predictive_entropy": H_pred,
        "mutual_info": mutual_info,
        "predictive_variance": pred_variance,
    }
    if return_all:
        results["all_pass_probs"] = pass_probs
    return results


def run_uncertainty_analysis(cfg, model, loader, class_names, device, exp_dir,
                             temperature=1.0, y_true=None):
    """Full uncertainty analysis on a split. Saves CSV + figure."""
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Single deterministic pass for labels
    from src.evaluation.evaluate import get_predictions
    det_probs, det_y_true = get_predictions(model, loader, device)
    if y_true is None:
        y_true = det_y_true

    n_passes = cfg.uncertainty.mc_dropout_passes
    mc = run_mc_dropout(model, loader, device, n_passes=n_passes,
                        temperature=temperature)
    mean_probs = mc["mc_mean_probs"]
    y_pred = np.argmax(mean_probs, axis=1)
    correct = (y_pred == y_true).astype(int)
    max_conf = np.max(mean_probs, axis=1)

    rows = []
    n = len(y_true)
    for i in range(n):
        rows.append({
            "true_label": class_names[y_true[i]],
            "predicted_label": class_names[y_pred[i]],
            "correct": int(correct[i]),
            "max_confidence": float(max_conf[i]),
            "predictive_entropy": float(mc["predictive_entropy"][i]),
            "mutual_information": float(mc["mutual_info"][i]),
            "predictive_variance": float(mc["predictive_variance"][i]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(exp_dir / "uncertainty_results.csv", index=False)

    # ── Analysis stats ──
    from scipy import stats as sp_stats
    from sklearn.metrics import roc_auc_score

    stats = {}
    for metric in ["predictive_entropy", "mutual_information", "predictive_variance", "max_confidence"]:
        vals = df[metric].values
        inc = df["correct"] == 0
        cor = df["correct"] == 1
        entry = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "median": float(np.median(vals)),
            "mean_correct": float(vals[cor].mean()) if cor.any() else float("nan"),
            "mean_incorrect": float(vals[inc].mean()) if inc.any() else float("nan"),
        }
        if cor.any() and inc.any():
            a = vals[cor]
            b = vals[inc]
            # Mann-Whitney U (one-sided: does error group have higher value?)
            u, p = sp_stats.mannwhitneyu(a, b, alternative="less")
            entry["mannwhitney_u"] = float(u)
            entry["mannwhitney_p"] = float(p)
            # Cohen's d effect size (pooled)
            n1, n2 = len(a), len(b)
            pooled = np.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2))
            entry["cohens_d"] = float((b.mean() - a.mean()) / pooled) if pooled > 0 else float("nan")
            # AUROC for "error detection" — higher metric value should flag errors.
            # For entropy/MI/variance larger => more uncertain => error. For confidence larger => less uncertain.
            score = -vals if metric == "max_confidence" else vals
            try:
                entry["error_detection_auroc"] = float(roc_auc_score(df["correct"], score))
            except Exception:
                entry["error_detection_auroc"] = float("nan")
        stats[metric] = entry
    save_json(stats, exp_dir / "uncertainty_metrics.json")

    # ── Figure ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes[0, 0].hist(df["max_confidence"][df["correct"] == 1],
                        bins=30, alpha=0.6, label="Correct", color="green")
        axes[0, 0].hist(df["max_confidence"][df["correct"] == 0],
                        bins=30, alpha=0.6, label="Incorrect", color="red")
        axes[0, 0].set_title("Confidence by Correctness")
        axes[0, 0].legend()

        axes[0, 1].hist(df["predictive_entropy"][df["correct"] == 1],
                        bins=30, alpha=0.6, label="Correct", color="green")
        axes[0, 1].hist(df["predictive_entropy"][df["correct"] == 0],
                        bins=30, alpha=0.6, label="Incorrect", color="red")
        axes[0, 1].set_title("Predictive Entropy by Correctness")
        axes[0, 1].legend()

        axes[1, 0].hist(df["mutual_information"][df["correct"] == 1],
                        bins=30, alpha=0.6, label="Correct", color="green")
        axes[1, 0].hist(df["mutual_information"][df["correct"] == 0],
                        bins=30, alpha=0.6, label="Incorrect", color="red")
        axes[1, 0].set_title("Mutual Information by Correctness")
        axes[1, 0].legend()

        axes[1, 1].scatter(df["max_confidence"], df["predictive_entropy"],
                           c=df["correct"], cmap="RdYlGn", alpha=0.4, s=8)
        axes[1, 1].set_title("Confidence vs Entropy")
        axes[1, 1].set_xlabel("Max Confidence"); axes[1, 1].set_ylabel("Predictive Entropy")

        plt.tight_layout()
        plt.savefig(exp_dir / "uncertainty_analysis.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [WARN] uncertainty figure failed: {e}")

    return df, stats