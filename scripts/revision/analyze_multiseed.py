#!/usr/bin/env python
"""
Multi-seed aggregation (seed 42 = paper run, seeds 43-46 = revision batch).

For every finished run (paper protocol = stopping-epoch checkpoint; and, for the new
seeds, the corrected best-epoch checkpoint) compute validation/test/official-test
metrics with every post-hoc step fitted on the internal validation split only.
Then report mean +- SD over seeds, paired per-seed differences (E1-E0, ResNet-EffNet),
validation-only model selection per seed, and a deep ensemble across seeds.

Outputs -> results/revision/multiseed/
"""
import sys
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common import (  # noqa: E402
    OUT, K, BATCH_EXP, COND_TO_KEY, PAPER_RUNS, load_dump, fit_temperature, calib_metrics,
    cls_metrics, uncertainty_scores, aurc, entropy, save_json)
from src.evaluation.selective import _threshold_for_coverage  # noqa: E402

D = OUT / "multiseed"
D.mkdir(parents=True, exist_ok=True)
UNK = 8


def discover():
    runs = []
    for k, p in PAPER_RUNS.items():
        if p.exists():
            runs.append({"model": k, "seed": 42, "protocol": "paper", "path": p})
    for d in sorted(BATCH_EXP.glob("seed*_*")):
        if not (d / "DONE").exists():
            continue
        seed = int(d.name.split("_")[0][4:])
        key = COND_TO_KEY[d.name.split("_", 1)[1]]
        if (d / "dump.npz").exists():
            runs.append({"model": key, "seed": seed, "protocol": "paper", "path": d / "dump.npz"})
        if (d / "dump_besttrue.npz").exists():
            runs.append({"model": key, "seed": seed, "protocol": "true_best", "path": d / "dump_besttrue.npz"})
    return runs


def run_metrics(r):
    d = load_dump(r["path"])
    T = fit_temperature(d["val"]["logits"], d["val"]["y"])
    out = {"model": r["model"], "seed": r["seed"], "protocol": r["protocol"], "T": T}
    for split in ["val", "test"]:
        y, z = d[split]["y"], d[split]["logits"]
        for m, v in cls_metrics(y, softmax(z, 1)).items():
            out[f"{split}_{m}"] = v
    y, z = d["test"]["y"], d["test"]["logits"]
    for m, v in calib_metrics(y, softmax(z, 1)).items():
        out[f"test_{m}_before"] = v
    for m, v in calib_metrics(y, softmax(z / T, 1)).items():
        out[f"test_{m}_afterT"] = v
    from scripts.revision.common import fit_vector_scaling
    w, b = fit_vector_scaling(d["val"]["logits"], d["val"]["y"])
    pvs = softmax(z * w + b, 1)
    cvs = calib_metrics(y, pvs)
    out["test_VS_NLL"], out["test_VS_cwECE"], out["test_VS_ECE"] = cvs["NLL"], cvs["ECE_classwise"], cvs["ECE"]
    out["test_VS_balanced_accuracy"] = cls_metrics(y, pvs)["balanced_accuracy"]
    if "_E0" in r["model"]:
        # post-hoc logit adjustment (prior correction towards balanced classes) for the unweighted model:
        # z + log w_c with w_c = N/(K n_c) from the training split, i.e. z - log(prior) + const
        import json as _json
        from scripts.revision.common import ROOT as _ROOT, CLASSES as _CL
        cw = _json.loads((_ROOT / "experiments" / "resnet50_baseline" / "class_weights.json").read_text())
        lw = np.log(np.array([cw[c] for c in _CL]))
        zv_a, zt_a = d["val"]["logits"] + lw, z + lw
        T_a = fit_temperature(zv_a, d["val"]["y"])
        pa = softmax(zt_a / T_a, 1)
        ca = calib_metrics(y, pa)
        cm = cls_metrics(y, pa)
        ea = (pa.argmax(1) != y).astype(int)
        out.update({"test_LA_balanced_accuracy": cm["balanced_accuracy"], "test_LA_accuracy": cm["accuracy"],
                    "test_LA_macro_f1": cm["macro_f1"], "test_LA_ECE_afterT": ca["ECE"], "test_LA_NLL_afterT": ca["NLL"],
                    "test_LA_cwECE": ca["ECE_classwise"], "test_LA_Brier": ca["Brier"],
                    "test_LA_AUROC_MSP": roc_auc_score(ea, 1 - pa.max(1)), "test_LA_AURC_MSP": aurc(ea, 1 - pa.max(1))})
        oa = softmax((d["official_test"]["logits"] + lw) / T_a, 1)
        yo_ = d["official_test"]["y"]
        out["off_LA_bal_acc_known"] = balanced_accuracy_score(yo_[yo_ < UNK], oa[yo_ < UNK].argmax(1))
    s, p = uncertainty_scores(z, d["test"]["mc_logits"], T)
    err = (p["det"].argmax(1) != y).astype(int)
    err_mc = (p["mc_mean"].argmax(1) != y).astype(int)
    out["test_AUROC_MSP"] = roc_auc_score(err, s["MSP_T"])
    out["test_AUROC_softmax_entropy"] = roc_auc_score(err, s["softmax_entropy_T"])
    out["test_AUROC_MC_entropy_deterr"] = roc_auc_score(err, s["MC_entropy_T"])
    out["test_AUROC_MC_entropy_MCerr"] = roc_auc_score(err_mc, s["MC_entropy_T"])
    out["test_AUROC_MC_MI"] = roc_auc_score(err, s["MC_mutual_info_T"])
    out["test_AURC_MSP"] = aurc(err, s["MSP_T"])
    out["test_AURC_MC_entropy"] = aurc(err, s["MC_entropy_T"])
    out["test_risk_full"] = err.mean()
    cv = softmax(d["val"]["logits"] / T, 1).max(1)
    tau = _threshold_for_coverage(cv, 0.85)
    acc = p["det"].max(1) >= tau
    out["sel_tau_val85"] = tau
    out["test_sel_coverage"] = acc.mean()
    out["test_sel_acc"] = (p["det"].argmax(1)[acc] == y[acc]).mean()
    # conformal LAC alpha = 0.1
    pv = softmax(d["val"]["logits"] / T, 1)
    yv = d["val"]["y"]
    sc = 1 - pv[np.arange(len(yv)), yv]
    q = np.quantile(sc, min(1, ceil((len(yv) + 1) * 0.9) / len(yv)), method="higher")
    S = (1 - p["det"]) <= q
    out["test_conformal90_coverage"] = S[np.arange(len(y)), y].mean()
    out["test_conformal90_setsize"] = S.sum(1).mean()
    # official test
    o = d["official_test"]
    yo = o["y"]
    kn = yo < UNK
    po_raw = softmax(o["logits"], 1)
    po = softmax(o["logits"] / T, 1)
    out["off_bal_acc_known"] = balanced_accuracy_score(yo[kn], po[kn].argmax(1))
    out["off_acc_known"] = (po[kn].argmax(1) == yo[kn]).mean()
    out["off_macro_f1_known"] = cls_metrics(yo[kn], po_raw[kn])["macro_f1"]
    cb = calib_metrics(yo[kn], po_raw[kn])
    ca = calib_metrics(yo[kn], po[kn])
    out["off_ECE_before"], out["off_ECE_afterT"] = cb["ECE"], ca["ECE"]
    out["off_NLL_before"], out["off_NLL_afterT"] = cb["NLL"], ca["NLL"]
    so, _ = uncertainty_scores(o["logits"], o["mc_logits"], T)
    u = (yo == UNK).astype(int)
    out["off_OOD_AUROC_MSP"] = roc_auc_score(u, so["MSP_T"])
    out["off_OOD_AUROC_MC_entropy"] = roc_auc_score(u, so["MC_entropy_T"])
    out["off_OOD_AUROC_MC_MI"] = roc_auc_score(u, so["MC_mutual_info_T"])
    acc_o = po.max(1) >= tau
    out["off_sel_coverage_known"] = acc_o[kn].mean()
    out["off_sel_acc_known"] = (po.argmax(1)[kn & acc_o] == yo[kn & acc_o]).mean()
    out["off_UNK_accepted"] = acc_o[yo == UNK].mean()
    out["off_bal_acc9_reject_as_UNK"] = balanced_accuracy_score(yo, np.where(acc_o, po.argmax(1), UNK))
    return out, d, T


def summarise(df, cols):
    rows = []
    for (m, prot), g in df.groupby(["model", "protocol"]):
        for c in cols:
            v = g[c].astype(float).values
            n = len(v)
            sd = v.std(ddof=1) if n > 1 else np.nan
            half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
            rows.append({"model": m, "protocol": prot, "metric": c, "n_seeds": n, "mean": v.mean(),
                         "sd": sd, "ci95_low": v.mean() - half, "ci95_high": v.mean() + half,
                         "min": v.min(), "max": v.max(), "seeds": ",".join(map(str, g.seed.tolist()))})
    return pd.DataFrame(rows)


def paired(df, a, b, cols, protocol="paper"):
    A = df[(df.model == a) & (df.protocol == protocol)].set_index("seed")
    B = df[(df.model == b) & (df.protocol == protocol)].set_index("seed")
    common = sorted(set(A.index) & set(B.index))
    rows = []
    for c in cols:
        dlt = (A.loc[common, c] - B.loc[common, c]).astype(float).values
        n = len(dlt)
        p = stats.ttest_1samp(dlt, 0).pvalue if n > 1 else np.nan
        try:
            pw = stats.wilcoxon(dlt).pvalue if n > 5 else np.nan
        except ValueError:
            pw = np.nan
        rows.append({"comparison": f"{a} - {b}", "metric": c, "n_paired_seeds": n, "mean_diff": dlt.mean(),
                     "sd_diff": dlt.std(ddof=1) if n > 1 else np.nan, "paired_t_p": p, "wilcoxon_p": pw,
                     "n_positive": int((dlt > 0).sum()), "per_seed": ";".join(f"{s}:{x:+.4f}" for s, x in zip(common, dlt))})
    return pd.DataFrame(rows)


def ensemble(dumps_T, name):
    """Deep ensemble = mean of member softmax (each temperature-scaled on validation)."""
    out = {"ensemble": name, "n_members": len(dumps_T)}
    for split in ["test", "official_test"]:
        P = np.stack([softmax(d[split]["logits"] / T, 1) for d, T in dumps_T])  # (M, N, K)
        y = dumps_T[0][0][split]["y"]
        mean = P.mean(0)
        H = entropy(mean)
        MI = H - entropy(P).mean(0)
        if split == "test":
            err = (mean.argmax(1) != y).astype(int)
            out.update({f"{split}_{m}": v for m, v in cls_metrics(y, mean).items()})
            out.update({f"{split}_{m}": v for m, v in calib_metrics(y, mean).items()})
            out["test_AUROC_ens_MSP"] = roc_auc_score(err, 1 - mean.max(1))
            out["test_AUROC_ens_entropy"] = roc_auc_score(err, H)
            out["test_AUROC_ens_MI"] = roc_auc_score(err, MI)
            out["test_AURC_ens_MSP"] = aurc(err, 1 - mean.max(1))
            out["test_AURC_ens_entropy"] = aurc(err, H)
        else:
            kn = y < UNK
            out["off_bal_acc_known"] = balanced_accuracy_score(y[kn], mean[kn].argmax(1))
            out["off_ECE_known"] = calib_metrics(y[kn], mean[kn])["ECE"]
            u = (y == UNK).astype(int)
            out["off_OOD_AUROC_ens_MSP"] = roc_auc_score(u, 1 - mean.max(1))
            out["off_OOD_AUROC_ens_entropy"] = roc_auc_score(u, H)
            out["off_OOD_AUROC_ens_MI"] = roc_auc_score(u, MI)
    return out


def main():
    runs = discover()
    rows, store = [], {}
    for r in runs:
        m, d, T = run_metrics(r)
        rows.append(m)
        store[(r["model"], r["seed"], r["protocol"])] = (d, T)
        print(f"  {r['model']} seed{r['seed']} {r['protocol']}: test bal_acc={m['test_balanced_accuracy']:.4f}", flush=True)
    df = pd.DataFrame(rows).sort_values(["model", "protocol", "seed"])
    df.to_csv(D / "per_run_metrics.csv", index=False)
    cols = [c for c in df.columns if c not in ("model", "seed", "protocol")]
    summarise(df, cols).to_csv(D / "summary_mean_sd.csv", index=False)
    key = ["test_accuracy", "test_balanced_accuracy", "test_macro_f1", "test_roc_auc_macro_ovr",
           "test_ECE_before", "test_ECE_afterT", "test_NLL_before", "test_NLL_afterT", "test_Brier_afterT",
           "test_AUROC_MSP", "test_AUROC_MC_entropy_deterr", "test_AURC_MSP", "test_sel_acc", "test_sel_coverage",
           "val_balanced_accuracy", "val_macro_f1", "off_bal_acc_known", "off_ECE_afterT", "off_OOD_AUROC_MSP",
           "T"]
    pr = pd.concat([paired(df, "resnet50_E1", "resnet50_E0", key),
                    paired(df, "resnet50_E1", "effnetb3_E1", key)])
    if (df.protocol == "true_best").any():
        pr = pd.concat([pr, paired(df, "resnet50_E1", "resnet50_E0", key, "true_best")])
    pr.to_csv(D / "paired_differences.csv", index=False)
    # per-seed validation-only model selection
    sel = []
    for seed in sorted(df.seed.unique()):
        g = df[(df.seed == seed) & (df.protocol == "paper") & df.model.isin(["resnet50_E1", "effnetb3_E1"])]
        if len(g) == 2:
            best = g.sort_values(["val_macro_f1", "val_balanced_accuracy"], ascending=False).iloc[0]
            sel.append({"seed": seed, "selected_on_val_macro_f1": best.model,
                        **{f"val_macro_f1_{r.model}": r.val_macro_f1 for r in g.itertuples()},
                        **{f"test_bal_acc_{r.model}": r.test_balanced_accuracy for r in g.itertuples()}})
    pd.DataFrame(sel).to_csv(D / "validation_model_selection_per_seed.csv", index=False)
    # deep ensembles
    ens = []
    for mk in ["resnet50_E1", "resnet50_E0", "effnetb3_E1"]:
        members = [(v[0], v[1]) for (m, s, p), v in store.items() if m == mk and p == "paper"]
        if len(members) >= 2:
            ens.append(ensemble(members, f"{mk}_paper_x{len(members)}"))
    if ens:
        pd.DataFrame(ens).to_csv(D / "deep_ensembles.csv", index=False)
    print("done ->", D)


if __name__ == "__main__":
    main()
