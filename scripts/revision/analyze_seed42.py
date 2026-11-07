#!/usr/bin/env python
"""
Reviewer-revision analyses on the seed-42 runs reported in the paper.
Inference dumps only -- no retraining. Outputs -> results/revision/seed42/

Sections
  A  reproduction of the submitted numbers from the dumps
  B  validation-only model selection (ResNet-50 vs EfficientNet-B3)
  C  lesion-cluster bootstrap CIs + paired differences (E1-E0, ResNet-EffNet)
  D  calibration: E0/E1/EffNet before/after T, adaptive + class-wise ECE, Brier,
     vector scaling, prior (logit) correction + T, CIs
  E  uncertainty scores vs. errors: MSP, softmax entropy, MC entropy/MI/expected
     entropy, AUROC/pAUROC + CIs + paired diffs vs MSP, correlations, combined score
  F  selective prediction: AURC/E-AURC vs random, val-vs-test coverage, raw vs
     calibrated thresholds, SGR risk-controlled thresholds, class-wise effects
  G  split-conformal prediction sets (LAC, APS, class-conditional LAC)
"""
import sys
from math import ceil, log2
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import beta, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common import (  # noqa: E402
    OUT, CLASSES, K, PAPER_RUNS, load_dump, split_groups, fit_temperature, fit_vector_scaling,
    calib_metrics, cls_metrics, per_class, uncertainty_scores, aurc, eaurc,
    risk_coverage_curve, cluster_bootstrap_indices, ci, save_json, ece_equal_width, nll)

D = OUT / "seed42"
D.mkdir(parents=True, exist_ok=True)
N_BOOT = 2000
MEL = CLASSES.index("MEL")


def class_weights_of(exp):
    import json
    w = json.loads((Path(PAPER_RUNS[exp]).parent / "class_weights.json").read_text())
    return np.array([w[c] for c in CLASSES])


def main():
    runs = {k: load_dump(p) for k, p in PAPER_RUNS.items()}
    test_ids, test_groups = split_groups("test")
    for k, r in runs.items():
        assert (r["test"]["ids"] == test_ids).all(), k
    res = {}

    # ---------------- A + B: point metrics on val and test, T fitted on val
    T = {k: fit_temperature(r["val"]["logits"], r["val"]["y"]) for k, r in runs.items()}
    rows = []
    for k, r in runs.items():
        for split in ["val", "test"]:
            y, z = r[split]["y"], r[split]["logits"]
            m = cls_metrics(y, softmax(z, 1))
            cb = calib_metrics(y, softmax(z, 1))
            ca = calib_metrics(y, softmax(z / T[k], 1))
            rows.append({"model": k, "split": split, **m,
                         **{f"{a}_before": b for a, b in cb.items()},
                         **{f"{a}_afterT": b for a, b in ca.items()}, "T": T[k]})
    point = pd.DataFrame(rows)
    point.to_csv(D / "A_point_metrics_val_test.csv", index=False)
    res["temperatures"] = T

    # model selection on validation only
    v = point[point.split == "val"].set_index("model")
    sel = v.loc[["resnet50_E1", "effnetb3_E1"], ["accuracy", "balanced_accuracy", "macro_f1",
                                                  "roc_auc_macro_ovr", "NLL_afterT", "ECE_afterT"]]
    sel.to_csv(D / "B_validation_model_selection.csv")
    res["validation_selection"] = {
        "criterion": "validation macro-F1 (same criterion as checkpointing), tie-break validation balanced accuracy",
        "resnet50_val_macro_f1": float(v.loc["resnet50_E1", "macro_f1"]),
        "effnet_val_macro_f1": float(v.loc["effnetb3_E1", "macro_f1"]),
        "resnet50_val_bal_acc": float(v.loc["resnet50_E1", "balanced_accuracy"]),
        "effnet_val_bal_acc": float(v.loc["effnetb3_E1", "balanced_accuracy"]),
        "selected": "resnet50_E1" if v.loc["resnet50_E1", "macro_f1"] >= v.loc["effnetb3_E1", "macro_f1"] else "effnetb3_E1",
    }
    # per-class (test) for primary
    pc = []
    for k in runs:
        for split in ["val", "test"]:
            t = per_class(runs[k][split]["y"], softmax(runs[k][split]["logits"], 1))
            t.insert(0, "split", split)
            t.insert(0, "model", k)
            pc.append(t)
    pd.concat(pc).to_csv(D / "A_per_class_metrics.csv", index=False)

    # ---------------- C: cluster bootstrap on test (paired across models)
    yt = runs["resnet50_E1"]["test"]["y"]
    probs_raw = {k: softmax(r["test"]["logits"], 1) for k, r in runs.items()}
    probs_T = {k: softmax(r["test"]["logits"] / T[k], 1) for k, r in runs.items()}
    unc = {}
    for k, r in runs.items():
        s, p = uncertainty_scores(r["test"]["logits"], r["test"]["mc_logits"], T[k])
        unc[k] = (s, p)
    boot = {k: {} for k in runs}
    diffs = {"E1_minus_E0": {}, "ResNet_minus_EffNet": {}}

    def stats_for(k, idx):
        y = yt[idx]
        pr, pt = probs_raw[k][idx], probs_T[k][idx]
        out = cls_metrics(y, pr)
        out["ECE_before"] = ece_equal_width(y, pr)
        out["ECE_afterT"] = ece_equal_width(y, pt)
        out["NLL_before"] = nll(y, pr)
        out["NLL_afterT"] = nll(y, pt)
        s, p = unc[k]
        err_mc = (p["mc_mean"][idx].argmax(1) != y).astype(int)
        err_det = (pt.argmax(1) != y).astype(int)
        out["AUROC_MCentropy_vs_MCerrors"] = roc_auc_score(err_mc, s["MC_entropy_T"][idx]) if 0 < err_mc.sum() < len(idx) else np.nan
        out["AUROC_MSP_vs_deterr"] = roc_auc_score(err_det, s["MSP_T"][idx])
        out["AUROC_MCentropy_vs_deterr"] = roc_auc_score(err_det, s["MC_entropy_T"][idx])
        out["AURC_MSP"] = aurc(err_det, s["MSP_T"][idx])
        conf = pt.max(1)
        acc_mask = conf >= sel_tau[k]
        out["sel_coverage"] = acc_mask.mean()
        out["sel_accepted_acc"] = (pt.argmax(1)[acc_mask] == y[acc_mask]).mean() if acc_mask.any() else np.nan
        return out

    # validation-locked thresholds (target val coverage 0.85, as in the paper)
    sel_tau = {}
    for k, r in runs.items():
        cv = softmax(r["val"]["logits"] / T[k], 1).max(1)
        sel_tau[k] = paper_threshold(cv, 0.85)
    res["selective_tau_cov85"] = sel_tau
    full = np.arange(len(yt))
    pts = {k: stats_for(k, full) for k in runs}
    for b, idx in enumerate(cluster_bootstrap_indices(test_groups, N_BOOT, seed=42)):
        cur = {k: stats_for(k, idx) for k in runs}
        for k in runs:
            for m, val in cur[k].items():
                boot[k].setdefault(m, []).append(val)
        for m in cur["resnet50_E1"]:
            diffs["E1_minus_E0"].setdefault(m, []).append(cur["resnet50_E1"][m] - cur["resnet50_E0"][m])
            diffs["ResNet_minus_EffNet"].setdefault(m, []).append(cur["resnet50_E1"][m] - cur["effnetb3_E1"][m])
    ci_rows = []
    for k in runs:
        for m in pts[k]:
            lo, hi = ci(boot[k][m])
            ci_rows.append({"model": k, "metric": m, "point": pts[k][m], "ci_low": lo, "ci_high": hi})
    for name, dd in diffs.items():
        a, b = ("resnet50_E1", "resnet50_E0") if name == "E1_minus_E0" else ("resnet50_E1", "effnetb3_E1")
        for m, vals in dd.items():
            vals = np.array(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            lo, hi = ci(vals)
            p_two = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
            ci_rows.append({"model": name, "metric": m, "point": pts[a][m] - pts[b][m],
                            "ci_low": lo, "ci_high": hi, "bootstrap_p_two_sided": min(1.0, p_two)})
    pd.DataFrame(ci_rows).to_csv(D / "C_bootstrap_ci_test.csv", index=False)

    # ---------------- D: calibration variants (fitted on val, evaluated on test)
    cal_rows = []
    for k, r in runs.items():
        zv, yv, zt = r["val"]["logits"], r["val"]["y"], r["test"]["logits"]
        variants = {"uncalibrated": softmax(zt, 1), "temperature": softmax(zt / T[k], 1)}
        w, b = fit_vector_scaling(zv, yv)
        variants["vector_scaling"] = softmax(zt * w + b, 1)
        if k.endswith("E1"):
            cw = class_weights_of(k)
            adj_v, adj_t = zv - np.log(cw), zt - np.log(cw)   # undo the class-weight prior shift
            T_adj = fit_temperature(adj_v, yv)
            variants["prior_correction"] = softmax(adj_t, 1)
            variants["prior_correction+T"] = softmax(adj_t / T_adj, 1)
            res.setdefault("T_after_prior_correction", {})[k] = T_adj
        for name, p in variants.items():
            row = {"model": k, "variant": name, **calib_metrics(yt, p), **cls_metrics(yt, p)}
            cal_rows.append(row)
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(D / "D_calibration_variants_test.csv", index=False)
    # class-wise reliability gaps (primary, after T): per-class ECE
    from scripts.revision.common import ece_classwise
    pcl = []
    for k in runs:
        for name, p in [("uncalibrated", probs_raw[k]), ("temperature", probs_T[k])]:
            for c in range(K):
                sub = np.zeros_like(p)
                edges = np.linspace(0, 1, 16)
                pk, tk = p[:, c], (yt == c).astype(float)
                e = 0.0
                for i in range(15):
                    m = (pk > edges[i]) & (pk <= edges[i + 1]) if i else (pk >= 0) & (pk <= edges[1])
                    if m.any():
                        e += abs(tk[m].mean() - pk[m].mean()) * m.mean()
                pcl.append({"model": k, "variant": name, "class": CLASSES[c], "classwise_ECE": e,
                            "mean_prob": pk.mean(), "prevalence": tk.mean()})
    pd.DataFrame(pcl).to_csv(D / "D_per_class_ECE_test.csv", index=False)

    # ---------------- E: uncertainty scores
    unc_rows, corr_rows = [], []
    for k, r in runs.items():
        s, p = unc[k]
        y = yt
        err_det = (p["det"].argmax(1) != y).astype(int)
        err_mc = (p["mc_mean"].argmax(1) != y).astype(int)
        for name, score in s.items():
            target = err_mc if name.startswith("MC_") else err_det
            unc_rows.append({"model": k, "score": name,
                             "target": "MC-mean prediction errors" if name.startswith("MC_") else "deterministic prediction errors",
                             "AUROC_own_errors": roc_auc_score(target, score),
                             "AUROC_det_errors": roc_auc_score(err_det, score),
                             "pAUROC_fpr0.2_det_errors": roc_auc_score(err_det, score, max_fpr=0.2),
                             "AUPR_error_det": average_precision_score(err_det, score),
                             "AURC_det": aurc(err_det, score), "EAURC_det": eaurc(err_det, score),
                             "mean_correct": score[err_det == 0].mean(), "mean_incorrect": score[err_det == 1].mean()})
        rho = spearmanr(s["MSP_T"], s["MC_entropy_T"]).correlation
        rho2 = spearmanr(s["MSP_T"], s["MC_mutual_info_T"]).correlation
        rho3 = spearmanr(s["softmax_entropy_T"], s["MC_entropy_T"]).correlation
        corr_rows.append({"model": k, "spearman_MSP_vs_MCentropy": rho, "spearman_MSP_vs_MI": rho2,
                          "spearman_softmaxEntropy_vs_MCentropy": rho3,
                          "frac_pred_changed_by_MC": float((p["mc_mean"].argmax(1) != p["det"].argmax(1)).mean())})
    unc_df = pd.DataFrame(unc_rows)
    unc_df.to_csv(D / "E_uncertainty_scores_test.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(D / "E_score_correlations_test.csv", index=False)

    # paired bootstrap: each score minus MSP (deterministic errors), primary model
    k = "resnet50_E1"
    s, p = unc[k]
    err_det = (p["det"].argmax(1) != yt).astype(int)
    names = list(s.keys())
    bvals = {n: [] for n in names}
    for idx in cluster_bootstrap_indices(test_groups, N_BOOT, seed=7):
        e = err_det[idx]
        base = roc_auc_score(e, s["MSP_T"][idx])
        for n in names:
            bvals[n].append(roc_auc_score(e, s[n][idx]) - base)
    pdiff = []
    for n in names:
        vals = np.array(bvals[n])
        lo, hi = ci(vals)
        pdiff.append({"score": n, "AUROC": roc_auc_score(err_det, s[n]),
                      "diff_vs_MSP_T": roc_auc_score(err_det, s[n]) - roc_auc_score(err_det, s["MSP_T"]),
                      "ci_low": lo, "ci_high": hi,
                      "p_two_sided": min(1.0, 2 * min((vals <= 0).mean(), (vals >= 0).mean()))})
    pd.DataFrame(pdiff).to_csv(D / "E_auroc_diff_vs_MSP_resnet50_E1.csv", index=False)

    # combined score: logistic regression on validation errors, evaluated on test
    comb = []
    for k, r in runs.items():
        sv, pv = uncertainty_scores(r["val"]["logits"], r["val"]["mc_logits"], T[k])
        ev = (pv["det"].argmax(1) != r["val"]["y"]).astype(int)
        st, pt = unc[k]
        et = (pt["det"].argmax(1) != yt).astype(int)
        for feats in [["MSP_T"], ["MC_entropy_T"], ["MSP_T", "MC_entropy_T"],
                      ["MSP_T", "MC_mutual_info_T"], ["MSP_T", "MC_entropy_T", "MC_mutual_info_T"]]:
            Xv = np.column_stack([sv[f] for f in feats])
            Xt = np.column_stack([st[f] for f in feats])
            mu, sd = Xv.mean(0), Xv.std(0) + 1e-12
            lr = LogisticRegression(C=1.0, max_iter=1000).fit((Xv - mu) / sd, ev)
            sc = lr.predict_proba((Xt - mu) / sd)[:, 1]
            comb.append({"model": k, "features": "+".join(feats), "test_AUROC": roc_auc_score(et, sc),
                         "test_AURC": aurc(et, sc)})
    pd.DataFrame(comb).to_csv(D / "E_combined_scores_test.csv", index=False)

    # ---------------- F: selective prediction
    sel_rows, rc_curves = [], {}
    for k, r in runs.items():
        for split in ["val", "test"]:
            s_, p_ = uncertainty_scores(r[split]["logits"], r[split]["mc_logits"], T[k])
            y = r[split]["y"]
            e = (p_["det"].argmax(1) != y).astype(int)
            for n in ["MSP_T", "MSP_raw", "softmax_entropy_T", "MC_entropy_T", "MC_mutual_info_T", "MC_MSP_T"]:
                sel_rows.append({"model": k, "split": split, "score": n, "AURC": aurc(e, s_[n]),
                                 "E_AURC": eaurc(e, s_[n]), "overall_risk(random-rejection AURC)": e.mean()})
                c_, rk_ = risk_coverage_curve(e, s_[n])
                rc_curves[f"{k}|{split}|{n}"] = (c_.tolist(), rk_.tolist())
    pd.DataFrame(sel_rows).to_csv(D / "F_aurc_val_test.csv", index=False)
    save_json(rc_curves, D / "F_risk_coverage_curves.json")

    # thresholds: target coverage on validation -> test (calibrated vs raw, MSP vs MC entropy)
    thr_rows = []
    for k, r in runs.items():
        sv, pv = uncertainty_scores(r["val"]["logits"], r["val"]["mc_logits"], T[k])
        st, pt = unc[k]
        yv = r["val"]["y"]
        for n in ["MSP_T", "MSP_raw", "MC_entropy_T"]:
            for target in [0.70, 0.80, 0.85, 0.90, 0.95]:
                if n == "MSP_T" or n == "MSP_raw":
                    conf_v = 1 - sv[n]
                    conf_t = 1 - st[n]
                    tau = paper_threshold(conf_v, target)
                    acc_v, acc_t = conf_v >= tau, conf_t >= tau
                else:
                    tau = np.quantile(sv[n], target)
                    acc_v, acc_t = sv[n] <= tau, st[n] <= tau
                predv = pv["det"].argmax(1)
                predt = pt["det"].argmax(1)
                rej = ~acc_t
                row = {"model": k, "score": n, "target_val_coverage": target, "threshold": float(tau),
                       "val_coverage": acc_v.mean(), "val_selective_risk": 1 - (predv[acc_v] == yv[acc_v]).mean(),
                       "test_coverage": acc_t.mean(),
                       "test_selective_acc": (predt[acc_t] == yt[acc_t]).mean(),
                       "test_n_accepted": int(acc_t.sum()), "test_n_rejected": int(rej.sum()),
                       "test_rejected_acc": (predt[rej] == yt[rej]).mean() if rej.any() else np.nan,
                       "test_n_correct_among_rejected": int((predt[rej] == yt[rej]).sum()),
                       "MEL_sensitivity_all": (predt[yt == MEL] == MEL).mean(),
                       "MEL_sensitivity_accepted": (predt[acc_t & (yt == MEL)] == MEL).mean(),
                       "MEL_coverage": acc_t[yt == MEL].mean(),
                       "MEL_cases_rejected": int((rej & (yt == MEL)).sum()),
                       "balanced_acc_accepted": np.mean([(predt[acc_t & (yt == c)] == c).mean()
                                                         for c in range(K) if (acc_t & (yt == c)).any()])}
                thr_rows.append(row)
    pd.DataFrame(thr_rows).to_csv(D / "F_threshold_transfer_val_to_test.csv", index=False)

    # class-wise effects at the paper operating point (MSP_T, target 0.85)
    k = "resnet50_E1"
    pt = probs_T[k]
    acc_t = pt.max(1) >= sel_tau[k]
    pred = pt.argmax(1)
    cw_rows = []
    for c in range(K):
        m = yt == c
        cw_rows.append({"class": CLASSES[c], "n_test": int(m.sum()), "coverage": acc_t[m].mean(),
                        "n_rejected": int((~acc_t & m).sum()),
                        "recall_all": (pred[m] == c).mean(),
                        "recall_accepted": (pred[m & acc_t] == c).mean() if (m & acc_t).any() else np.nan,
                        "share_of_rejections": (~acc_t & m).sum() / (~acc_t).sum()})
    pd.DataFrame(cw_rows).to_csv(D / "F_classwise_selective_resnet50_E1.csv", index=False)

    # SGR (Geifman & El-Yaniv 2017): risk-controlled threshold on validation
    sgr_rows = []
    for k, r in runs.items():
        cv = softmax(r["val"]["logits"] / T[k], 1)
        conf_v, err_v = cv.max(1), (cv.argmax(1) != r["val"]["y"]).astype(int)
        conf_t, err_t = probs_T[k].max(1), (probs_T[k].argmax(1) != yt).astype(int)
        for r_star in [0.10, 0.15, 0.20, 0.25]:
            for delta in [0.05]:
                theta, bound, cov_v, risk_v = sgr(conf_v, err_v, r_star, delta)
                a = conf_t >= theta if theta is not None else np.zeros_like(conf_t, bool)
                sgr_rows.append({"model": k, "target_risk": r_star, "delta": delta, "threshold": theta,
                                 "val_coverage": cov_v, "val_risk": risk_v, "risk_bound": bound,
                                 "test_coverage": a.mean(),
                                 "test_risk": err_t[a].mean() if a.any() else np.nan,
                                 "test_risk_le_target": bool(err_t[a].mean() <= r_star) if a.any() else None})
    pd.DataFrame(sgr_rows).to_csv(D / "F_SGR_risk_control.csv", index=False)

    # ---------------- G: split conformal (calibration = validation)
    conf_rows, cc_rows = [], []
    for k, r in runs.items():
        pv = softmax(r["val"]["logits"] / T[k], 1)
        yv = r["val"]["y"]
        pt_ = probs_T[k]
        for alpha in [0.10, 0.05]:
            sets = {}
            # LAC
            sc = 1 - pv[np.arange(len(yv)), yv]
            q = np.quantile(sc, min(1, ceil((len(yv) + 1) * (1 - alpha)) / len(yv)), method="higher")
            sets["LAC"] = (1 - pt_) <= q
            # APS (non-randomised)
            order_v = np.argsort(-pv, 1)
            cum_v = np.cumsum(np.take_along_axis(pv, order_v, 1), 1)
            rank_y = np.argmax(order_v == yv[:, None], 1)
            sc = cum_v[np.arange(len(yv)), rank_y]
            q = np.quantile(sc, min(1, ceil((len(yv) + 1) * (1 - alpha)) / len(yv)), method="higher")
            order_t = np.argsort(-pt_, 1)
            cum_t = np.cumsum(np.take_along_axis(pt_, order_t, 1), 1)
            incl_sorted = (cum_t - np.take_along_axis(pt_, order_t, 1)) < q
            S = np.zeros_like(pt_, bool)
            np.put_along_axis(S, order_t, incl_sorted, 1)
            sets["APS"] = S
            # class-conditional (Mondrian) LAC
            S = np.zeros_like(pt_, bool)
            for c in range(K):
                scc = 1 - pv[yv == c, c]
                n_c = len(scc)
                lvl = min(1, ceil((n_c + 1) * (1 - alpha)) / n_c)
                qc = np.quantile(scc, lvl, method="higher")
                S[:, c] = (1 - pt_[:, c]) <= qc
            sets["classwise_LAC"] = S
            for name, S in sets.items():
                cover = S[np.arange(len(yt)), yt]
                size = S.sum(1)
                single = size == 1
                conf_rows.append({"model": k, "alpha": alpha, "method": name, "coverage": cover.mean(),
                                  "avg_set_size": size.mean(), "frac_singleton": single.mean(),
                                  "frac_empty": (size == 0).mean(),
                                  "acc_on_singletons": (S[single].argmax(1) == yt[single]).mean() if single.any() else np.nan,
                                  "MEL_coverage": cover[yt == MEL].mean(),
                                  "min_class_coverage": min(cover[yt == c].mean() for c in range(K))})
                for c in range(K):
                    cc_rows.append({"model": k, "alpha": alpha, "method": name, "class": CLASSES[c],
                                    "coverage": cover[yt == c].mean(), "avg_set_size": size[yt == c].mean()})
    pd.DataFrame(conf_rows).to_csv(D / "G_conformal_test.csv", index=False)
    pd.DataFrame(cc_rows).to_csv(D / "G_conformal_classwise_test.csv", index=False)

    save_json(res, D / "summary.json")
    print("done ->", D)


def paper_threshold(conf_val, target_cov):
    """Exactly the rule used for the paper (src/evaluation/selective._threshold_for_coverage):
    the largest threshold on a 0.001 grid whose VALIDATION coverage is >= the target."""
    from src.evaluation.selective import _threshold_for_coverage
    return float(_threshold_for_coverage(conf_val, target_cov))


def sgr(conf, err, r_star, delta):
    """Selection with Guaranteed Risk (Geifman & El-Yaniv, 2017), binary search over
    the number of accepted (most confident) validation samples with a Clopper-Pearson bound
    and a union bound over ceil(log2 m) tests."""
    order = np.argsort(-conf, kind="mergesort")
    c, e = conf[order], err[order]
    m = len(c)
    k = ceil(log2(m))
    lo, hi = 1, m
    best = None
    for _ in range(k):
        mid = (lo + hi) // 2
        n_err = int(e[:mid].sum())
        b = beta.ppf(1 - delta / k, n_err + 1, mid - n_err) if n_err < mid else 1.0
        if b <= r_star:
            best = (mid, b)
            lo = mid + 1
        else:
            hi = mid - 1
        if lo > hi:
            break
    if best is None:
        return None, None, 0.0, np.nan
    mid, b = best
    return float(c[mid - 1]), float(b), mid / m, float(e[:mid].mean())


if __name__ == "__main__":
    main()
