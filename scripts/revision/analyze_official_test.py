#!/usr/bin/env python
"""
Shift / OOD evaluation on the OFFICIAL ISIC 2019 test set (8,238 images incl. the
UNK class that has no training examples). This set was never used during model
development; every threshold, temperature and score below is locked on the
INTERNAL validation split beforehand.

Outputs -> results/revision/official_test/
"""
import sys
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common import (  # noqa: E402
    ROOT, OUT, CLASSES, K, PAPER_RUNS, load_dump, fit_temperature, calib_metrics, cls_metrics,
    uncertainty_scores, aurc, ci, save_json, per_class)
from src.evaluation.selective import _threshold_for_coverage  # noqa: E402

D = OUT / "official_test"
D.mkdir(parents=True, exist_ok=True)
UNK = 8
MEL = 0


def main(n_boot=1000):
    gt = pd.read_csv(ROOT / "data" / "raw" / "ISIC_2019_Test_GroundTruth.csv")
    excl = set()
    dup_csv = OUT / "integrity" / "near_duplicate_candidates_official_vs_internal.csv"
    if dup_csv.exists():
        dd = pd.read_csv(dup_csv)
        if len(dd):
            excl = set(dd[dd["strong"]]["official"])
    rows, ood_rows, sel_rows, chal_rows, conf_rows, pc_rows = [], [], [], [], [], []
    for k, path in PAPER_RUNS.items():
        r = load_dump(path)
        o = r["official_test"]
        assert (o["ids"] == gt["image"].values).all()
        T = fit_temperature(r["val"]["logits"], r["val"]["y"])
        y = o["y"]
        subsets = {"all_8238": np.ones(len(y), bool),
                   "scored_subset": gt["score_weight"].values == 1,
                   "all_minus_near_duplicates": ~gt["image"].isin(excl).values}
        s, p = uncertainty_scores(o["logits"], o["mc_logits"], T)
        sv, pv = uncertainty_scores(r["val"]["logits"], r["val"]["mc_logits"], T)
        tau = _threshold_for_coverage(1 - sv["MSP_T"], 0.85)          # paper rule, internal val
        tau_ent = np.quantile(sv["MC_entropy_T"], 0.85)              # 85% val coverage for MC entropy
        pred = p["det"].argmax(1)
        for sname, sub in subsets.items():
            known = sub & (y < UNK)
            yk, zk = y[known], o["logits"][known]
            pr, pt = softmax(zk, 1), softmax(zk / T, 1)
            row = {"model": k, "subset": sname, "n_total": int(sub.sum()), "n_known": int(known.sum()),
                   "n_UNK": int((sub & (y == UNK)).sum()), "T_internal_val": T,
                   **cls_metrics(yk, pr),
                   **{f"{m}_before": v for m, v in calib_metrics(yk, pr).items()},
                   **{f"{m}_afterT": v for m, v in calib_metrics(yk, pt).items()}}
            rows.append(row)
            # OOD: UNK vs known
            is_unk = (y[sub] == UNK).astype(int)
            for n, sc in s.items():
                ood_rows.append({"model": k, "subset": sname, "score": n,
                                 "AUROC_UNK_vs_known": roc_auc_score(is_unk, sc[sub])})
            # selective at locked thresholds (known classes + UNK acceptance)
            for rule, accept in [("MSP_T>=tau_val85", 1 - s["MSP_T"] >= tau),
                                 ("MC_entropy<=q_val85", s["MC_entropy_T"] <= tau_ent)]:
                a_k = accept[known]
                sel_rows.append({"model": k, "subset": sname, "rule": rule,
                                 "threshold": tau if rule.startswith("MSP") else float(tau_ent),
                                 "coverage_known": a_k.mean(),
                                 "accepted_acc_known": (pred[known][a_k] == yk[a_k]).mean(),
                                 "overall_acc_known": (pred[known] == yk).mean(),
                                 "rejected_acc_known": (pred[known][~a_k] == yk[~a_k]).mean(),
                                 "UNK_accepted_frac": accept[sub & (y == UNK)].mean(),
                                 "MEL_sens_all": (pred[known & (y == MEL)] == MEL).mean(),
                                 "MEL_sens_accepted": (pred[known & (y == MEL) & accept] == MEL).mean()})
                # challenge-style 9-class balanced accuracy: rejected -> UNK
                pred9 = np.where(accept, pred, UNK)[sub]
                chal_rows.append({"model": k, "subset": sname, "rule": rule,
                                  "bal_acc_9class_reject_as_UNK": balanced_accuracy_score(y[sub], pred9),
                                  "UNK_recall": (pred9[y[sub] == UNK] == UNK).mean()})
            pred9_never = pred[sub]
            chal_rows.append({"model": k, "subset": sname, "rule": "never_predict_UNK",
                              "bal_acc_9class_reject_as_UNK": balanced_accuracy_score(y[sub], pred9_never),
                              "UNK_recall": 0.0})
            # conformal (internal-val calibrated LAC), coverage under shift on known classes
            pvT = softmax(r["val"]["logits"] / T, 1)
            yv = r["val"]["y"]
            for alpha in [0.10, 0.05]:
                scv = 1 - pvT[np.arange(len(yv)), yv]
                q = np.quantile(scv, min(1, ceil((len(yv) + 1) * (1 - alpha)) / len(yv)), method="higher")
                S = (1 - pt) <= q
                conf_rows.append({"model": k, "subset": sname, "alpha": alpha,
                                  "coverage_known": S[np.arange(len(yk)), yk].mean(),
                                  "avg_set_size_known": S.sum(1).mean(),
                                  "avg_set_size_UNK": ((1 - softmax(o["logits"][sub & (y == UNK)] / T, 1)) <= q).sum(1).mean()})
            if sname == "all_8238":
                t = per_class(yk, pr)
                t.insert(0, "model", k)
                pc_rows.append(t)
        # bootstrap CI (image-level; no lesion ids for the official set) on key metrics, all images
        if k == "resnet50_E1":
            rng = np.random.default_rng(0)
            known_idx = np.where(y < UNK)[0]
            all_idx = np.arange(len(y))
            bs = {"bal_acc_known": [], "ECE_afterT_known": [], "AUROC_UNK_MSP": [], "AUROC_UNK_MCentropy": [],
                  "AUROC_UNK_MI": []}
            from scripts.revision.common import ece_equal_width
            for _ in range(n_boot):
                ik = rng.choice(known_idx, len(known_idx))
                ia = rng.choice(all_idx, len(all_idx))
                bs["bal_acc_known"].append(balanced_accuracy_score(y[ik], pred[ik]))
                bs["ECE_afterT_known"].append(ece_equal_width(y[ik], softmax(o["logits"][ik] / T, 1)))
                u = (y[ia] == UNK).astype(int)
                bs["AUROC_UNK_MSP"].append(roc_auc_score(u, s["MSP_T"][ia]))
                bs["AUROC_UNK_MCentropy"].append(roc_auc_score(u, s["MC_entropy_T"][ia]))
                bs["AUROC_UNK_MI"].append(roc_auc_score(u, s["MC_mutual_info_T"][ia]))
            save_json({m: ci(v) for m, v in bs.items()}, D / "bootstrap_ci_resnet50_E1.json")
    pd.DataFrame(rows).to_csv(D / "classification_calibration_known_classes.csv", index=False)
    pd.DataFrame(ood_rows).to_csv(D / "ood_unk_detection.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(D / "selective_under_shift.csv", index=False)
    pd.DataFrame(chal_rows).to_csv(D / "challenge_style_balanced_accuracy.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(D / "conformal_under_shift.csv", index=False)
    pd.concat(pc_rows).to_csv(D / "per_class_known.csv", index=False)
    save_json({"n_excluded_near_duplicates": len(excl)}, D / "meta.json")
    print("done ->", D)


if __name__ == "__main__":
    main()
