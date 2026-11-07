#!/usr/bin/env python
"""Compute statistical separation of uncertainty metrics from existing CSVs.

Reads experiments/<exp>/uncertainty_results.csv and computes, for each metric:
  - Mann-Whitney U test (one-sided, errors have higher value)
  - Cohen's d effect size
  - Error-detection AUROC (higher metric => more likely to be an error;
    for max_confidence the sign is flipped)
Writes results/tables/uncertainty_separation.csv and prints a summary.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJ = Path(__file__).resolve().parent.parent
EXPS = {
    "resnet50_baseline": "ResNet-50",
    "efficientnet_b3_baseline": "EfficientNet-B3",
}
METRICS = ["predictive_entropy", "mutual_information", "predictive_variance", "max_confidence"]


def compute_separation(vals, correct, is_confidence):
    """Return dict of statistics for one metric (correct vs incorrect)."""
    cor = vals[correct == 1]
    inc = vals[correct == 0]
    out = {"mean_correct": float(cor.mean()), "mean_incorrect": float(inc.mean()),
           "n_correct": int(len(cor)), "n_incorrect": int(len(inc))}
    if len(cor) > 0 and len(inc) > 0:
        # One-sided: is the error group's value higher? (True for entropy/MI/variance)
        u, p = sp_stats.mannwhitneyu(cor, inc, alternative="less")
        out["mannwhitney_u"] = float(u)
        out["mannwhitney_p"] = float(p)
        n1, n2 = len(cor), len(inc)
        pooled = np.sqrt(((n1 - 1) * cor.std(ddof=1) ** 2 + (n2 - 1) * inc.std(ddof=1) ** 2) / (n1 + n2 - 2))
        out["cohens_d"] = float((inc.mean() - cor.mean()) / pooled) if pooled > 0 else float("nan")
        # AUROC for ERROR DETECTION: positive class = "is an error".
        # Higher score must mean "more likely an error":
        #   entropy/MI/variance: higher = more uncertain = error -> score = vals
        #   confidence: lower = more uncertain = error -> score = -vals
        error = (correct == 0).astype(int)
        score = -vals if is_confidence else vals
        try:
            out["error_detection_auroc"] = float(roc_auc_score(error, score))
        except Exception:
            out["error_detection_auroc"] = float("nan")
        # Bootstrap 95% CI on the AUROC (2000 resamples)
        rng = np.random.default_rng(42)
        idx = rng.integers(0, len(error), size=(2000, len(error)))
        try:
            aucs = [roc_auc_score(error[i], score[i]) for i in idx]
            out["auroc_ci95"] = [round(float(np.percentile(aucs, 2.5)), 4),
                                 round(float(np.percentile(aucs, 97.5)), 4)]
        except Exception:
            out["auroc_ci95"] = None
    else:
        out.update({"mannwhitney_u": None, "mannwhitney_p": None, "cohens_d": None,
                    "error_detection_auroc": None, "auroc_ci95": None})
    return out


rows = []
for exp, display in EXPS.items():
    p = PROJ / "experiments" / exp / "uncertainty_results.csv"
    if not p.exists():
        print(f"[SKIP] missing {p}")
        continue
    df = pd.read_csv(p)
    correct = df["correct"].values.astype(int)
    n_err = int((correct == 0).sum())
    print(f"\n=== {display}  (n_test={len(df)}, errors={n_err}, acc={1 - n_err / len(df):.3f}) ===")
    for metric in METRICS:
        vals = df[metric].values.astype(float)
        s = compute_separation(vals, correct, is_confidence=(metric == "max_confidence"))
        rows.append({"model": display, "exp": exp, "metric": metric, **s})
        ci = s["auroc_ci95"]
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a"
        print(f"  {metric:22s} cor={s['mean_correct']:.3f} inc={s['mean_incorrect']:.3f} "
              f"| U p={s['mannwhitney_p']:.2e} d={s['cohens_d']:.3f} "
              f"| AUC={s['error_detection_auroc']:.3f} 95%CI={ci_str}")

out_dir = PROJ / "results" / "tables"
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out_dir / "uncertainty_separation.csv", index=False)
print(f"\nSaved {out_dir / 'uncertainty_separation.csv'}")