"""
Shared helpers for the reviewer-revision analyses (no training).

Everything is computed from the inference dumps written by
scripts/dump_predictions.py (raw logits + MC-Dropout logits), so every number in
the revision can be regenerated from saved predictions.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from scipy.special import softmax, log_softmax
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "revision"
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
K = 8

# seed-42 runs = the runs reported in the submitted paper
PAPER_RUNS = {
    "resnet50_E1": ROOT / "experiments" / "resnet50_baseline" / "dump.npz",
    "resnet50_E0": ROOT / "experiments" / "resnet50_e0" / "dump.npz",
    "effnetb3_E1": ROOT / "experiments" / "efficientnet_b3_baseline" / "dump.npz",
}
BATCH_EXP = ROOT / "runs" / "repeat_batch_20261107" / "experiments"
COND_TO_KEY = {"resnet50_weighted": "resnet50_E1", "resnet50_unweighted": "resnet50_E0",
               "efficientnet_b3_weighted": "effnetb3_E1",
               "resnet50_weighted_r320": "resnet50_E1_r320", "resnet50_unweighted_r320": "resnet50_E0_r320"}


def save_json(obj, path):
    def _d(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_d), encoding="utf-8")


def load_dump(path):
    d = np.load(path, allow_pickle=False)
    out = {}
    for split in ["val", "test", "official_test"]:
        if f"{split}_logits" in d:
            out[split] = {"ids": d[f"{split}_ids"], "y": d[f"{split}_y"].astype(int),
                          "logits": d[f"{split}_logits"].astype(np.float64),
                          "mc_logits": d[f"{split}_mc_logits"].astype(np.float64)}
    return out


def all_runs():
    """{model_key: {seed: dump_path}} including seed 42 (paper) + finished batch seeds."""
    runs = {k: {42: v} for k, v in PAPER_RUNS.items() if v.exists()}
    if BATCH_EXP.exists():
        for d in sorted(BATCH_EXP.glob("seed*_*")):
            if not (d / "DONE").exists() or not (d / "dump.npz").exists():
                continue
            seed = int(d.name.split("_")[0].replace("seed", ""))
            cond = d.name.split("_", 1)[1]
            runs.setdefault(COND_TO_KEY[cond], {})[seed] = d / "dump.npz"
    return runs


# ---------------------------------------------------------------- groups
def split_groups(split):
    """Lesion-group id per image (lesion_id, else the image itself) for cluster bootstrap."""
    csv = {"val": "validation.csv", "test": "test.csv"}[split]
    df = pd.read_csv(ROOT / "data" / "splits" / csv)
    g = df["lesion_id"].fillna("img_" + df["image_id"]).values
    return df["image_id"].values, g


# ---------------------------------------------------------------- calibration
def fit_temperature(logits, y, bounds=(0.1, 10.0)):
    """Same procedure as src/calibration/temperature.py (bounded scalar NLL)."""
    def nll(t):
        return -log_softmax(logits / t, axis=1)[np.arange(len(y)), y].mean()
    return float(minimize_scalar(nll, bounds=bounds, method="bounded").x)


def fit_vector_scaling(logits, y, l2=1e-3):
    """Vector scaling (Guo et al. 2017): z' = w * z + b, w,b in R^K, fitted on NLL."""
    n = len(y)

    def f(theta):
        w, b = theta[:K], theta[K:]
        z = logits * w + b
        ls = log_softmax(z, axis=1)
        p = np.exp(ls)
        loss = -ls[np.arange(n), y].mean() + l2 * np.sum(b ** 2)
        g = p.copy()
        g[np.arange(n), y] -= 1
        g /= n
        gw = (g * logits).sum(0)
        gb = g.sum(0) + 2 * l2 * b
        return loss, np.concatenate([gw, gb])
    t = fit_temperature(logits, y)
    theta0 = np.concatenate([np.full(K, 1.0 / t), np.zeros(K)])
    res = minimize(f, theta0, jac=True, method="L-BFGS-B")
    return res.x[:K], res.x[K:]


def ece_equal_width(y, probs, n_bins=15):
    """Identical to src/calibration/temperature.compute_ece (top-label, equal width)."""
    conf = probs.max(1)
    acc = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            e += abs(acc[m].mean() - conf[m].mean()) * m.mean()
    return float(e)


def mce_equal_width(y, probs, n_bins=15):
    conf = probs.max(1)
    acc = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    g = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            g = max(g, abs(acc[m].mean() - conf[m].mean()))
    return float(g)


def ece_adaptive(y, probs, n_bins=15):
    """Equal-mass (adaptive) top-label ECE (Nixon et al. 2019)."""
    conf = probs.max(1)
    acc = (probs.argmax(1) == y).astype(float)
    order = np.argsort(conf)
    e = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            e += abs(acc[chunk].mean() - conf[chunk].mean()) * len(chunk) / len(y)
    return float(e)


def ece_classwise(y, probs, n_bins=15):
    """Class-wise ECE (Kull et al. 2019; 'SCE' in Nixon et al.): mean over classes of
    the equal-width ECE of p_k against 1[y=k]."""
    edges = np.linspace(0, 1, n_bins + 1)
    tot = 0.0
    for k in range(probs.shape[1]):
        p = probs[:, k]
        t = (y == k).astype(float)
        e = 0.0
        for i in range(n_bins):
            m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p >= edges[i]) & (p <= edges[i + 1])
            if m.any():
                e += abs(t[m].mean() - p[m].mean()) * m.mean()
        tot += e
    return float(tot / probs.shape[1])


def brier(y, probs):
    oh = np.eye(probs.shape[1])[y]
    return float(((probs - oh) ** 2).sum(1).mean())


def nll(y, probs):
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean())


def calib_metrics(y, probs):
    return {"ECE": ece_equal_width(y, probs), "MCE": mce_equal_width(y, probs),
            "ECE_adaptive": ece_adaptive(y, probs), "ECE_classwise": ece_classwise(y, probs),
            "Brier": brier(y, probs), "NLL": nll(y, probs)}


# ---------------------------------------------------------------- classification
def cls_metrics(y, probs):
    pred = probs.argmax(1)
    out = {"accuracy": accuracy_score(y, pred),
           "balanced_accuracy": balanced_accuracy_score(y, pred),
           "macro_f1": f1_score(y, pred, average="macro", labels=list(range(K)), zero_division=0)}
    try:
        out["roc_auc_macro_ovr"] = roc_auc_score(y, probs, multi_class="ovr", average="macro",
                                                 labels=list(range(K)))
    except ValueError:
        out["roc_auc_macro_ovr"] = float("nan")
    return out


def per_class(y, probs):
    pred = probs.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(range(K)), zero_division=0)
    return pd.DataFrame({"class": CLASSES, "precision": p, "recall": r, "f1": f, "support": s})


# ---------------------------------------------------------------- uncertainty
def entropy(p, axis=-1):
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis)


def uncertainty_scores(logits, mc_logits, T=1.0):
    """Higher = more uncertain. Returns dict of per-sample scores + predictions."""
    p_raw = softmax(logits, 1)
    p_T = softmax(logits / T, 1)
    mc_T = softmax(mc_logits / T, 2)          # (P, N, K)
    mc_raw = softmax(mc_logits, 2)
    mean_T = mc_T.mean(0)
    mean_raw = mc_raw.mean(0)
    s = {
        "MSP_raw": 1 - p_raw.max(1),
        "MSP_T": 1 - p_T.max(1),
        "margin_T": -(np.sort(p_T, 1)[:, -1] - np.sort(p_T, 1)[:, -2]),
        "softmax_entropy_raw": entropy(p_raw),
        "softmax_entropy_T": entropy(p_T),
        "MC_entropy_T": entropy(mean_T),
        "MC_entropy_raw": entropy(mean_raw),
        "MC_expected_entropy_T": entropy(mc_T).mean(0),       # aleatoric part
        "MC_mutual_info_T": entropy(mean_T) - entropy(mc_T).mean(0),  # epistemic part
        "MC_variance_T": mc_T.std(0).mean(1),
        "MC_MSP_T": 1 - mean_T.max(1),
    }
    return s, {"det": p_T, "mc_mean": mean_T}


def aurc(risk_indicator, score):
    """Area under the risk-coverage curve; samples rejected in order of decreasing score."""
    order = np.argsort(score, kind="mergesort")    # most confident (lowest uncertainty) first
    err = risk_indicator[order].astype(float)
    n = len(err)
    cum_risk = np.cumsum(err) / np.arange(1, n + 1)
    return float(cum_risk.mean())


def risk_coverage_curve(err, score, n_points=200):
    order = np.argsort(score, kind="mergesort")
    e = err[order].astype(float)
    n = len(e)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(e) / np.arange(1, n + 1)
    idx = np.unique(np.linspace(0, n - 1, n_points).astype(int))
    return cov[idx], risk[idx]


def eaurc(err, score):
    r = err.mean()
    opt = r + (1 - r) * np.log(1 - r) if r < 1 else 1.0   # optimal AURC (Geifman et al. 2019)
    return aurc(err, score) - opt


# ---------------------------------------------------------------- bootstrap
def cluster_bootstrap_indices(groups, n_boot=2000, seed=0):
    """Resample lesion groups with replacement; yields index arrays."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [[] for _ in range(len(uniq))]
    for i, g in enumerate(inv):
        members[g].append(i)
    members = [np.array(m) for m in members]
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([members[j] for j in pick])


def ci(vals, alpha=0.05):
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    return [float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))]


def fmt_ci(point, lohi, nd=3):
    return f"{point:.{nd}f} [{lohi[0]:.{nd}f}, {lohi[1]:.{nd}f}]"
