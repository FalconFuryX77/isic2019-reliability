#!/usr/bin/env python
"""
Quantitative Grad-CAM checks for the revision (inference only, seed-42 ResNet-50 E1).

1. Model-parameter randomisation test (Adebayo et al., 2018): cascading re-initialisation
   head -> layer4 -> layer3 -> layer2 -> layer1; Spearman rank correlation and top-10%
   IoU between original and randomised maps.
2. Deletion / insertion faithfulness (Petsiuk et al., 2018) for Grad-CAM vs. a random
   map, a fixed centre-Gaussian prior and the dataset-mean Grad-CAM map.
3. Centre bias: dataset-mean map, centroid offset, share of mass in the central region.
4. Link to uncertainty/errors: spatial entropy / peak share of each test map vs.
   correctness and MC predictive entropy.
5. Uncurated example figure: images drawn at random (fixed seed) from each
   correct/incorrect x high/low-confidence bucket.

Same target layer as the paper (features[-1][-1].conv3); layer4 output is also
reported as a robustness check.  Outputs -> results/revision/gradcam/
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common import ROOT, OUT, CLASSES, load_dump, fit_temperature, uncertainty_scores, save_json  # noqa
from src.models.builders import build_model  # noqa
from src.data.dataset import make_transforms  # noqa

D = OUT / "gradcam"
D.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE = ROOT / "data" / "processed" / "cache224" / "train"
TF = make_transforms(224, augment=False)


def load_model():
    m = build_model("resnet50", 8, pretrained=False, dense_size=128, dropout=0.5, use_batchnorm=True, device=DEV)
    m.load_state_dict(torch.load(ROOT / "experiments" / "resnet50_baseline" / "best_resnet50.pt",
                                 map_location=DEV, weights_only=True))
    return m.eval()


def batch_cam(model, x, layer, target=None):
    """Batched Grad-CAM (eval mode => samples independent). Returns maps (B,224,224) in [0,1], logits."""
    store = {}
    h1 = layer.register_forward_hook(lambda mod, i, o: store.__setitem__("a", o))
    x = x.clone().requires_grad_(True)
    logits = model(x)
    if target is None:
        target = logits.argmax(1)
    a = store["a"]
    g = torch.autograd.grad(logits[torch.arange(len(x)), target].sum(), a)[0]
    h1.remove()
    w = g.mean((2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[:, 0]
    mx = cam.flatten(1).max(1)[0].clamp_min(1e-12)[:, None, None]
    return (cam / mx).detach(), logits.detach(), target


def load_batch(ids):
    xs = []
    for i in ids:
        with Image.open(CACHE / f"{i}.png") as im:
            xs.append(TF(im.convert("RGB")))
    return torch.stack(xs)


def spatial_stats(maps):
    m = maps.flatten(1).double()
    p = m / m.sum(1, keepdim=True).clamp_min(1e-12)
    ent = -(p * (p.clamp_min(1e-12)).log()).sum(1) / np.log(p.shape[1])
    k = int(0.1 * p.shape[1])
    top = p.topk(k, 1)[0].sum(1)
    yy, xx = torch.meshgrid(torch.arange(224.), torch.arange(224.), indexing="ij")
    cy = (p * yy.flatten().double().to(p.device)).sum(1)
    cx = (p * xx.flatten().double().to(p.device)).sum(1)
    off = torch.sqrt((cy - 111.5) ** 2 + (cx - 111.5) ** 2) / 111.5
    c = torch.zeros(224, 224, dtype=torch.bool)
    c[56:168, 56:168] = True
    central = p[:, c.flatten().to(p.device)].sum(1)
    return ent.cpu().numpy(), top.cpu().numpy(), off.cpu().numpy(), central.cpu().numpy()


def top_iou(a, b, q=0.9):
    ta = a.flatten(1) >= torch.quantile(a.flatten(1).float(), q, dim=1, keepdim=True)
    tb = b.flatten(1) >= torch.quantile(b.flatten(1).float(), q, dim=1, keepdim=True)
    return ((ta & tb).sum(1) / (ta | tb).sum(1).clamp_min(1)).cpu().numpy()


@torch.no_grad()
def del_ins_curves(model, x, sal, target, steps=20, blur=None):
    """Deletion: remove most-salient pixels (replace by blurred image); insertion: start from
    blurred image and add most-salient pixels. Returns AUC of predicted-class probability."""
    B = len(x)
    order = sal.flatten(1).argsort(1, descending=True)
    n = order.shape[1]
    del_p, ins_p = [], []
    for s in range(steps + 1):
        k = int(round(n * s / steps))
        mask = torch.zeros(B, n, device=x.device)
        if k:
            mask.scatter_(1, order[:, :k], 1.0)
        mask = mask.view(B, 1, 224, 224)
        xd = x * (1 - mask) + blur * mask
        xi = blur * (1 - mask) + x * mask
        pd_ = torch.softmax(model(xd), 1)[torch.arange(B), target]
        pi_ = torch.softmax(model(xi), 1)[torch.arange(B), target]
        del_p.append(pd_.cpu().numpy())
        ins_p.append(pi_.cpu().numpy())
    del_p, ins_p = np.stack(del_p, 1), np.stack(ins_p, 1)
    auc = lambda c: np.trapezoid(c, dx=1 / steps, axis=1)  # noqa
    return auc(del_p), auc(ins_p)


def gaussian_blur(x, k=31, sigma=10):
    t = torch.arange(k, device=x.device) - k // 2
    g = torch.exp(-(t.float() ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, 1, k).repeat(3, 1, 1, 1)
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g, groups=3)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.transpose(2, 3), groups=3)


def main(n_sample=300, bs=8):
    torch.manual_seed(0)
    if torch.cuda.is_available():
        # share the 6 GB GPU with the running training batch: cap this process so that it,
        # not the training job, fails first if memory runs out
        torch.cuda.set_per_process_memory_fraction(0.28)
    model = load_model()
    layers = {"conv3_paper": model.features[-1][-1].conv3, "layer4_output": model.features[-1]}
    df = pd.read_csv(ROOT / "data" / "splits" / "test.csv")
    ids = df.image_id.tolist()
    dump = load_dump(ROOT / "experiments" / "resnet50_baseline" / "dump.npz")
    T = fit_temperature(dump["val"]["logits"], dump["val"]["y"])
    s, p = uncertainty_scores(dump["test"]["logits"], dump["test"]["mc_logits"], T)
    y = dump["test"]["y"]
    pred = p["det"].argmax(1)
    conf = p["det"].max(1)
    correct = pred == y
    res = {}

    # ---- full test set maps -> spatial stats, centre bias, uncertainty link
    stats = {ln: {"ent": [], "top": [], "off": [], "central": []} for ln in layers}
    mean_map = {ln: torch.zeros(224, 224, device=DEV) for ln in layers}
    for b in range(0, len(ids), bs):
        x = load_batch(ids[b:b + bs]).to(DEV)
        for ln, L in layers.items():
            cam, _, _ = batch_cam(model, x, L)
            mean_map[ln] += (cam / cam.flatten(1).sum(1).clamp_min(1e-12)[:, None, None]).sum(0)
            e, t, o, c = spatial_stats(cam)
            for key, v in zip(["ent", "top", "off", "central"], [e, t, o, c]):
                stats[ln][key].append(v)
    rows = []
    for ln in layers:
        st = {k: np.concatenate(v) for k, v in stats[ln].items()}
        mm = (mean_map[ln] / len(ids)).cpu().numpy()
        np.save(D / f"mean_map_{ln}.npy", mm)
        rows.append({"layer": ln,
                     "mean_spatial_entropy(norm)": st["ent"].mean(),
                     "mean_top10pct_mass": st["top"].mean(),
                     "mean_centroid_offset(frac_of_half_width)": st["off"].mean(),
                     "mean_mass_in_central_25pct_area": st["central"].mean(),
                     "AUROC_error_spatial_entropy": roc_auc_score(~correct, st["ent"]),
                     "AUROC_error_top10_mass(neg)": roc_auc_score(~correct, -st["top"]),
                     "spearman_entropyMap_vs_MCentropy": spearmanr(st["ent"], s["MC_entropy_T"]).correlation,
                     "spearman_entropyMap_vs_MSP": spearmanr(st["ent"], s["MSP_T"]).correlation,
                     "mean_spatial_entropy_correct": st["ent"][correct].mean(),
                     "mean_spatial_entropy_incorrect": st["ent"][~correct].mean()})
        pd.DataFrame({"image_id": ids, "correct": correct, "conf_T": conf, "MC_entropy_T": s["MC_entropy_T"],
                      **{f"map_{k}": v for k, v in st.items()}}).to_csv(D / f"per_image_map_stats_{ln}.csv", index=False)
    pd.DataFrame(rows).to_csv(D / "centre_bias_and_uncertainty_link.csv", index=False)

    # ---- random sample for randomisation + deletion/insertion
    rng = np.random.default_rng(42)
    samp = rng.choice(len(ids), n_sample, replace=False)
    # 1. cascading randomisation
    stages = ["head", "layer4", "layer3", "layer2", "layer1"]
    rand_rows = []
    rmodel = copy.deepcopy(model)
    gen = torch.Generator().manual_seed(1)
    orig = {ln: [] for ln in layers}
    for b in range(0, n_sample, bs):
        x = load_batch([ids[i] for i in samp[b:b + bs]]).to(DEV)
        for ln, L in layers.items():
            orig[ln].append(batch_cam(model, x, L)[0].cpu())
    orig = {ln: torch.cat(v) for ln, v in orig.items()}
    feat_idx = {"layer4": 7, "layer3": 6, "layer2": 5, "layer1": 4}
    for stg in stages:
        mod = rmodel.head if stg == "head" else rmodel.features[feat_idx[stg]]
        for sub in mod.modules():
            if hasattr(sub, "reset_parameters"):
                sub.reset_parameters()
        rmodel.eval()
        rl = {"conv3_paper": rmodel.features[-1][-1].conv3, "layer4_output": rmodel.features[-1]}
        for ln in layers:
            maps = []
            for b in range(0, n_sample, bs):
                x = load_batch([ids[i] for i in samp[b:b + bs]]).to(DEV)
                # explain the ORIGINAL predicted class
                tgt = torch.as_tensor(pred[samp[b:b + bs]], device=DEV)
                maps.append(batch_cam(rmodel, x, rl[ln], target=tgt)[0].cpu())
            maps = torch.cat(maps)
            o = orig[ln]
            sp = [spearmanr(o[i, ::4, ::4].flatten(), maps[i, ::4, ::4].flatten()).correlation for i in range(n_sample)]
            sp = np.nan_to_num(np.array(sp, dtype=float))
            rand_rows.append({"layer": ln, "randomised_up_to": stg, "mean_spearman": sp.mean(),
                              "median_spearman": np.median(sp), "mean_top10_IoU": top_iou(o, maps).mean()})
    pd.DataFrame(rand_rows).to_csv(D / "randomisation_test.csv", index=False)

    # 2. deletion / insertion
    yy, xx = torch.meshgrid(torch.arange(224.), torch.arange(224.), indexing="ij")
    centre = torch.exp(-((yy - 111.5) ** 2 + (xx - 111.5) ** 2) / (2 * 50.0 ** 2)).to(DEV)
    di = {k: {"del": [], "ins": []} for k in ["gradcam_conv3", "gradcam_layer4", "random", "centre_gaussian", "mean_gradcam"]}
    mm = torch.as_tensor(np.load(D / "mean_map_conv3_paper.npy"), device=DEV)
    for b in range(0, n_sample, 6):
        x = load_batch([ids[i] for i in samp[b:b + 6]]).to(DEV)
        tgt = torch.as_tensor(pred[samp[b:b + 6]], device=DEV)
        blur = gaussian_blur(x)
        sal = {"gradcam_conv3": batch_cam(model, x, layers["conv3_paper"], tgt)[0],
               "gradcam_layer4": batch_cam(model, x, layers["layer4_output"], tgt)[0],
               "random": torch.rand(len(x), 224, 224, device=DEV),
               "centre_gaussian": centre.expand(len(x), 224, 224) + 1e-6 * torch.rand(len(x), 224, 224, device=DEV),
               "mean_gradcam": mm.expand(len(x), 224, 224) + 1e-9 * torch.rand(len(x), 224, 224, device=DEV)}
        for k, smap in sal.items():
            d_, i_ = del_ins_curves(model, x, smap, tgt, blur=blur)
            di[k]["del"].append(d_)
            di[k]["ins"].append(i_)
    di_rows = []
    for k, v in di.items():
        d_, i_ = np.concatenate(v["del"]), np.concatenate(v["ins"])
        di_rows.append({"saliency": k, "deletion_AUC(lower=better)": d_.mean(), "deletion_sd": d_.std(),
                        "insertion_AUC(higher=better)": i_.mean(), "insertion_sd": i_.std(), "n": len(d_)})
        np.save(D / f"delins_{k}.npy", np.stack([d_, i_]))
    pd.DataFrame(di_rows).to_csv(D / "deletion_insertion.csv", index=False)
    # paired comparison gradcam vs centre prior
    g = np.load(D / "delins_gradcam_conv3.npy")
    c = np.load(D / "delins_centre_gaussian.npy")
    from scipy.stats import wilcoxon
    res["paired_gradcam_vs_centre"] = {
        "deletion_diff_mean": float((g[0] - c[0]).mean()), "deletion_wilcoxon_p": float(wilcoxon(g[0], c[0]).pvalue),
        "insertion_diff_mean": float((g[1] - c[1]).mean()), "insertion_wilcoxon_p": float(wilcoxon(g[1], c[1]).pvalue)}
    save_json(res, D / "summary.json")

    # 5. uncurated figure + mean map figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    buckets = {"Correct, high conf.": correct & (conf >= 0.85), "Incorrect, high conf.": ~correct & (conf >= 0.85),
               "Correct, low conf.": correct & (conf < 0.6), "Incorrect, low conf.": ~correct & (conf < 0.6)}
    rng2 = np.random.default_rng(2026)
    fig, ax = plt.subplots(4, 6, figsize=(12, 8.6))
    for r_, (bn, msk) in enumerate(buckets.items()):
        pick = rng2.choice(np.where(msk)[0], 3, replace=False)
        x = load_batch([ids[i] for i in pick]).to(DEV)
        cam = batch_cam(model, x, layers["conv3_paper"])[0].cpu().numpy()
        for j, i in enumerate(pick):
            with Image.open(CACHE / f"{ids[i]}.png") as im:
                img = np.asarray(im.convert("RGB"))
            ax[r_, 2 * j].imshow(img)
            ax[r_, 2 * j].set_title(f"{CLASSES[y[i]]}->{CLASSES[pred[i]]} p={conf[i]:.2f}", fontsize=7)
            ax[r_, 2 * j + 1].imshow(img)
            ax[r_, 2 * j + 1].imshow(cam[j], cmap="jet", alpha=0.45)
            ax[r_, 2 * j + 1].set_title(f"H={s['MC_entropy_T'][i]:.2f}", fontsize=7)
        ax[r_, 0].set_ylabel(bn, fontsize=8)
    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])
    plt.suptitle("Grad-CAM, images drawn at random (seed 2026) within each bucket; T-scaled confidence", fontsize=9)
    plt.tight_layout()
    plt.savefig(D / "gradcam_uncurated_random.png", dpi=170)
    plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    for a, ln in zip(ax, layers):
        a.imshow(np.load(D / f"mean_map_{ln}.npy"), cmap="viridis")
        a.set_title(f"mean Grad-CAM ({ln})", fontsize=8)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(D / "gradcam_mean_maps.png", dpi=160)
    plt.close()
    print("done ->", D)


if __name__ == "__main__":
    main()
