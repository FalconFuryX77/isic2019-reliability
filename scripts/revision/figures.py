#!/usr/bin/env python
"""Figures for the revised manuscript -> results/revision/figures/ (300 dpi PNG + PDF)."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common import (ROOT, OUT, CLASSES, PAPER_RUNS, load_dump, fit_temperature,  # noqa
                                     uncertainty_scores, risk_coverage_curve, ece_equal_width)
from src.evaluation.selective import _threshold_for_coverage  # noqa

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

F = OUT / "figures"
F.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "font.family": "serif"})
COL = {"MSP_T": "#1f4e79", "MC_entropy_T": "#c55a11", "MC_mutual_info_T": "#7f7f7f",
       "softmax_entropy_T": "#548235", "random": "black", "MSP_raw": "#5b9bd5"}
LAB = {"MSP_T": "max. calibrated prob. (MSP)", "MC_entropy_T": "MC-Dropout pred. entropy",
       "MC_mutual_info_T": "MC-Dropout mutual information", "softmax_entropy_T": "softmax entropy",
       "random": "random rejection", "MSP_raw": "MSP, uncalibrated softmax"}
RAW_LS = (0, (4, 1.5, 1, 1.5))


def save(fig, name):
    fig.savefig(F / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(F / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def risk_coverage(d, T, tau):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, split in zip(axes, ["val", "test"]):
        s, p = uncertainty_scores(d[split]["logits"], d[split]["mc_logits"], T)
        y = d[split]["y"]
        err = (p["det"].argmax(1) != y).astype(int)
        for n in ["MSP_T", "MC_entropy_T", "MC_mutual_info_T"]:
            c, r = risk_coverage_curve(err, s[n], 400)
            ax.plot(c, r, color=COL[n], lw=1.2, label=LAB[n])
        c, r = risk_coverage_curve(err, s["MSP_raw"], 400)
        ax.plot(c, r, color=COL["MSP_raw"], lw=1.0, ls=RAW_LS, label=LAB["MSP_raw"])
        ax.axhline(err.mean(), color="black", ls=":", lw=1, label=LAB["random"])
        conf = p["det"].max(1)
        a = conf >= tau
        ax.plot(a.mean(), err[a].mean(), "o", color="#1f4e79", ms=4)
        ax.annotate(f"$\\tau$={tau:.3f}: coverage {a.mean():.3f}, risk {err[a].mean():.3f}",
                    (a.mean(), err[a].mean()), xytext=(0.30, 0.04), textcoords="axes fraction", fontsize=6.5,
                    arrowprops=dict(arrowstyle="-", lw=0.5))
        ax.set_title({"val": "Internal validation (threshold fitted here)", "test": "Internal test (threshold locked)"}[split])
        ax.set_xlabel("coverage")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("selective risk (error rate)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.12))
    save(fig, "fig_risk_coverage_val_test")


def reliability(d, T, name, n_bins=15):
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0), sharey=True)
    for j, (split, tt, lab) in enumerate([("val", 1.0, "val, uncalibrated"), ("val", T, f"val, T={T:.2f}"),
                                          ("test", 1.0, "test, uncalibrated"), ("test", T, f"test, T={T:.2f}")]):
        ax = axes[j]
        pr = softmax(d[split]["logits"] / tt, 1)
        y = d[split]["y"]
        conf, acc = pr.max(1), (pr.argmax(1) == y).astype(float)
        edges = np.linspace(0, 1, n_bins + 1)
        cs, accs, ns = [], [], []
        for i in range(n_bins):
            m = (conf > edges[i]) & (conf <= edges[i + 1])
            if m.any():
                cs.append(conf[m].mean())
                accs.append(acc[m].mean())
                ns.append(m.sum())
        ax.bar(edges[:-1], [0] * n_bins, width=1 / n_bins)
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.scatter(cs, accs, s=np.array(ns) / max(ns) * 40 + 4, color="#1f4e79", zorder=3)
        ax.plot(cs, accs, color="#1f4e79", lw=0.8)
        ax.set_title(f"{lab}\nECE$_{{15}}$={ece_equal_width(y, pr):.3f}")
        ax.set_xlabel("confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    save(fig, name)


def parse_log(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r"epoch +(\d+) \| tr_loss ([\d.]+) \| val_loss ([\d.]+) \| tr_acc ([\d.]+) \| val_acc ([\d.]+) \| val_f1_macro ([\d.]+)", line)
        if m:
            rows.append(dict(zip(["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "val_f1"],
                                 [int(m.group(1))] + [float(x) for x in m.groups()[1:]])))
    return pd.DataFrame(rows)


def training_curves():
    logs = {"ResNet-50 E1 (weighted CE)": ("logs/train_resnet50.log", 24),
            "ResNet-50 E0 (plain CE)": ("logs/e0_train.log", 21),
            "EfficientNet-B3 E1": ("logs/train_efficientnet.log", 25)}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    out = []
    for ax, (name, (lp, saved)) in zip(axes, logs.items()):
        h = parse_log(ROOT / lp)
        h.insert(0, "run", name)
        out.append(h)
        ax2 = ax.twinx()
        ax.plot(h.epoch, h.val_loss, color="#c55a11", lw=1.2, label="val. loss (training objective)")
        ax.plot(h.epoch, h.train_loss, color="#c55a11", lw=0.8, ls="--", label="train loss")
        ax2.plot(h.epoch, h.val_f1, color="#1f4e79", lw=1.2, label="val. macro-F1")
        b = h.loc[h.val_f1.idxmax()]
        ax2.axvline(b.epoch, color="#1f4e79", ls=":", lw=0.8)
        ax.axvline(saved, color="black", ls="-", lw=0.6)
        ax.axvline(10.5, color="grey", ls="--", lw=0.5)
        ax.set_title(f"{name}\nbest val. F1 ep. {int(b.epoch)}, saved ep. {saved}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss", color="#c55a11")
        ax2.set_ylabel("val. macro-F1", color="#1f4e79")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color="#c55a11", lw=1.2), Line2D([], [], color="#c55a11", lw=0.8, ls="--"),
          Line2D([], [], color="#1f4e79", lw=1.2), Line2D([], [], color="#1f4e79", ls=":", lw=0.8),
          Line2D([], [], color="black", lw=0.6), Line2D([], [], color="grey", ls="--", lw=0.5)]
    ls_ = ["val. loss (training objective)", "train loss", "val. macro-F1", "best val. macro-F1 epoch",
           "epoch of saved weights", "backbone unfrozen"]
    plt.tight_layout()
    fig.legend(hs, ls_, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.08), fontsize=6.5)
    save(fig, "fig_training_curves")
    pd.concat(out).to_csv(OUT / "training_history_from_logs.csv", index=False)


def classwise_selective(d, T, tau):
    pr = softmax(d["test"]["logits"] / T, 1)
    y = d["test"]["y"]
    a = pr.max(1) >= tau
    pred = pr.argmax(1)
    cov = [a[y == c].mean() for c in range(8)]
    rec_all = [(pred[y == c] == c).mean() for c in range(8)]
    rec_acc = [(pred[(y == c) & a] == c).mean() for c in range(8)]
    x = np.arange(8)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.bar(x - 0.27, cov, 0.27, label="coverage", color="#a6a6a6")
    ax.bar(x, rec_all, 0.27, label="recall (all)", color="#9dc3e6")
    ax.bar(x + 0.27, rec_acc, 0.27, label="recall (accepted)", color="#1f4e79")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={int((y == i).sum())})" for i, c in enumerate(CLASSES)], fontsize=6)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_classwise_selective")


def rc_compact(d, T, tau):
    """Single-column risk-coverage figure: internal test (solid) + validation MSP (dashed) + official known."""
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    s, p = uncertainty_scores(d["test"]["logits"], d["test"]["mc_logits"], T)
    y = d["test"]["y"]
    err = (p["det"].argmax(1) != y).astype(int)
    for n in ["MSP_T", "MC_entropy_T", "MC_mutual_info_T"]:
        c, r = risk_coverage_curve(err, s[n], 400)
        ax.plot(c, r, color=COL[n], lw=1.2, label=LAB[n].replace("MC-Dropout", "MC"))
    c, r = risk_coverage_curve(err, s["MSP_raw"], 400)
    ax.plot(c, r, color=COL["MSP_raw"], lw=0.9, ls=RAW_LS, label="MSP, uncalibrated")
    sv, pv = uncertainty_scores(d["val"]["logits"], d["val"]["mc_logits"], T)
    ev = (pv["det"].argmax(1) != d["val"]["y"]).astype(int)
    c, r = risk_coverage_curve(ev, sv["MSP_T"], 400)
    ax.plot(c, r, color=COL["MSP_T"], lw=0.9, ls="--", label="MSP, validation")
    o = d["official_test"]
    so, po = uncertainty_scores(o["logits"], o["mc_logits"], T)
    kn = o["y"] < 8
    eo = (po["det"].argmax(1)[kn] != o["y"][kn]).astype(int)
    c, r = risk_coverage_curve(eo, so["MSP_T"][kn], 400)
    ax.plot(c, r, color="#7030a0", lw=0.9, ls="-.", label="MSP, official test")
    ax.axhline(err.mean(), color="black", ls=":", lw=0.9, label="random (test)")
    a = p["det"].max(1) >= tau
    ax.plot(a.mean(), err[a].mean(), "o", color="#1f4e79", ms=3.5)
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective risk")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.45)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=5.8, loc="upper left", ncol=2)
    save(fig, "fig_rc_compact")


def confusion_e1(d, T):
    """Row-normalised confusion matrix of the submitted E1 model (the submitted version's Fig. 2, normalised)."""
    pr = softmax(d["test"]["logits"] / T, 1)
    y, pred = d["test"]["y"], pr.argmax(1)
    K = len(CLASSES)
    cm = np.zeros((K, K))
    for t, q in zip(y, pred):
        cm[t, q] += 1
    n = cm.sum(1)
    cmn = cm / n[:, None]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            v = cmn[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                        color="white" if v > 0.55 else "black")
    ax.set_xticks(range(K))
    ax.set_xticklabels(CLASSES, fontsize=6.5, rotation=45)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"{c} ({int(k)})" for c, k in zip(CLASSES, n)], fontsize=6.5)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class (test images)")
    save(fig, "fig_confusion_E1")


def seed_curves():
    """Validation macro-F1 and validation loss per epoch for every run, with best and stored epochs marked."""
    B = ROOT / "runs" / "repeat_batch_20261107" / "experiments"
    logs42 = {"resnet50_weighted": "logs/train_resnet50.log", "resnet50_unweighted": "logs/e0_train.log",
              "efficientnet_b3_weighted": "logs/train_efficientnet.log"}
    conds = [("resnet50_weighted", "ResNet-50 E1"), ("resnet50_unweighted", "ResNet-50 E0"),
             ("efficientnet_b3_weighted", "EfficientNet-B3 E1"), ("resnet50_weighted_r320", "ResNet-50 E1, 320 px"),
             ("resnet50_unweighted_r320", "ResNet-50 E0, 320 px")]
    cmap = plt.get_cmap("tab10")
    seeds_all = [42, 43, 44, 45, 46]
    fig, axes = plt.subplots(2, len(conds), figsize=(7.2, 3.6), sharex=True)
    rows = []
    for j, (cond, title) in enumerate(conds):
        for s in seeds_all:
            if s == 42 and cond in logs42:
                h = parse_log(ROOT / logs42[cond]).rename(columns={"val_f1": "val_f1_macro"})
                best = int(h.loc[h.val_f1_macro.idxmax(), "epoch"])
            else:
                fh = B / f"seed{s}_{cond}" / "training_history.csv"
                if not fh.exists():
                    continue
                h = pd.read_csv(fh)
                if not len(h):
                    continue
                best = int(h.loc[h.val_f1_macro.idxmax(), "epoch"])
            # the checkpoint bug: when early stopping fires (patience 6) the weights of that epoch are stored;
            # if training reaches its last epoch first, the stored weights are those of the best epoch
            last = int(h.epoch.max())
            triggered = last - best >= 6
            stored = last if triggered else best
            col = cmap(seeds_all.index(s))
            for i, met in enumerate(["val_f1_macro", "val_loss"]):
                ax = axes[i, j]
                ax.plot(h.epoch, h[met], color=col, lw=0.9, label=f"seed {s}")
                ax.plot(best, h.loc[h.epoch == best, met].iloc[0], "o", color=col, ms=3)
                ax.plot(stored, h.loc[h.epoch == stored, met].iloc[0], "x", color=col, ms=4)
            lmin = int(h.loc[h.val_loss.idxmin(), "epoch"])
            vl = h.set_index("epoch").val_loss
            rows.append({"condition": cond, "seed": s, "best_epoch": best, "last_epoch": last,
                         "early_stop_triggered": triggered, "stored_epoch": stored, "min_val_loss_epoch": lmin,
                         "val_loss_min": float(vl.min()), "val_loss_at_stored": float(vl.loc[stored])})
        axes[0, j].set_title(title, fontsize=7)
        for i in range(2):
            axes[i, j].axvline(10.5, color="grey", ls="--", lw=0.5)
            axes[i, j].grid(alpha=0.3)
            axes[i, j].tick_params(labelsize=6)
        axes[1, j].set_xlabel("epoch", fontsize=7)
    axes[0, 0].set_ylabel("val. macro-F1", fontsize=7)
    axes[1, 0].set_ylabel("val. loss", fontsize=7)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=cmap(k), lw=1) for k in range(len(seeds_all))] + \
         [Line2D([], [], color="black", marker="o", ls="", ms=3), Line2D([], [], color="black", marker="x", ls="", ms=4),
          Line2D([], [], color="grey", ls="--", lw=0.5)]
    ls_ = [f"seed {s}" for s in seeds_all] + ["best val. macro-F1", "stored weights", "backbone unfrozen"]
    plt.tight_layout()
    fig.legend(hs, ls_, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=6)
    save(fig, "fig_seed_curves")
    pd.DataFrame(rows).to_csv(OUT / "figures" / "best_vs_stored_epochs.csv", index=False)


def gradcam_mean():
    """Mean Grad-CAM map over the 300 random test images (centre-bias check), with the central quarter marked."""
    G_ = OUT / "gradcam"
    panels = [("mean_map_conv3_paper.npy", "last bottleneck conv3 (as submitted)"),
              ("mean_map_layer4_output.npy", "layer4 output")]
    if not all((G_ / f).exists() for f, _ in panels):
        return
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.3))
    for ax, (f, t) in zip(axes, panels):
        mmap = np.load(G_ / f)
        h, w = mmap.shape[-2:]
        mmap = mmap.reshape(h, w) / mmap.max()
        im = ax.imshow(mmap, cmap="viridis", vmin=0, vmax=1)
        ax.add_patch(Rectangle((w * 0.25, h * 0.25), w * 0.5, h * 0.5, fill=False, ec="white", ls="--", lw=0.8))
        ax.set_title(t, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("mean map (relative)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    save(fig, "fig_gradcam_mean")


def main():
    d = load_dump(PAPER_RUNS["resnet50_E1"])
    T = fit_temperature(d["val"]["logits"], d["val"]["y"])
    tau = _threshold_for_coverage(softmax(d["val"]["logits"] / T, 1).max(1), 0.85)
    risk_coverage(d, T, tau)
    rc_compact(d, T, tau)
    reliability(d, T, "fig_reliability_resnet50_E1")
    d0 = load_dump(PAPER_RUNS["resnet50_E0"])
    reliability(d0, fit_temperature(d0["val"]["logits"], d0["val"]["y"]), "fig_reliability_resnet50_E0")
    training_curves()
    classwise_selective(d, T, tau)
    confusion_e1(d, T)
    seed_curves()
    gradcam_mean()
    print("done ->", F)


if __name__ == "__main__":
    main()
