#!/usr/bin/env python
"""
Test-time-augmentation (TTA) uncertainty baseline (inference only; cf. Combalia et al. 2020).
For the seed-42 models: 16 passes with the training augmentation pipeline on validation,
internal test and official test images; stores logits and reports error-detection and
UNK-detection AUROC of TTA entropy / mutual information next to MSP and MC-Dropout.
Outputs -> results/revision/tta/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common import (ROOT, OUT, PAPER_RUNS, load_dump, fit_temperature, entropy,  # noqa
                                     uncertainty_scores, aurc, cls_metrics, calib_metrics)
from scripts.dump_predictions import eval_sets, _ImgList  # noqa
from src.data.dataset import make_transforms  # noqa
from src.models.builders import build_model  # noqa
import yaml  # noqa

D = OUT / "tta"
D.mkdir(parents=True, exist_ok=True)
CKPT = {"resnet50_E1": ("experiments/resnet50_baseline/best_resnet50.pt", "resnet50"),
        "resnet50_E0": ("experiments/resnet50_e0/best_resnet50.pt", "resnet50"),
        "effnetb3_E1": ("experiments/efficientnet_b3_baseline/best_efficientnet_b3.pt", "efficientnet_b3")}


class _Seeded(torch.utils.data.Dataset):
    def __init__(self, base, seed):
        self.base, self.seed = base, seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        torch.manual_seed(self.seed * 1_000_003 + i)   # reproducible augmentation per (pass, image)
        import random
        random.seed(self.seed * 1_000_003 + i)
        return self.base[i]


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def gpu_augment(x, gen):
    """Per-sample random augmentation on a [0,1] image batch on GPU, mirroring the training
    augmentations: horizontal flip (p=0.5), rotation +-15 deg, scale 0.9-1.1, translation +-5%,
    brightness and contrast factors 0.9-1.1."""
    import torch.nn.functional as F
    B, dev = x.shape[0], x.device
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(B, generator=gen, device=dev)  # noqa
    ang = u(-15, 15) * np.pi / 180
    sc = u(0.9, 1.1)
    tx, ty = u(-0.1, 0.1), u(-0.1, 0.1)          # +-5% of size = +-0.1 in normalised [-1,1] coords
    flip = torch.where(torch.rand(B, generator=gen, device=dev) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    x = x * u(0.9, 1.1).view(B, 1, 1, 1)
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).mean((1, 2)).view(B, 1, 1, 1)
    x = (x - gray) * u(0.9, 1.1).view(B, 1, 1, 1) + gray
    return x.clamp(0, 1)


@torch.no_grad()
def tta_logits(key, passes=16, bs=64):
    dev = torch.device("cuda")
    ck, bb = CKPT[key]
    model = build_model(bb, 8, pretrained=False, dense_size=128, dropout=0.5, use_batchnorm=True, device=dev)
    model.load_state_dict(torch.load(ROOT / ck, map_location=dev, weights_only=True))
    model.eval()
    import torchvision.transforms as Tv
    to_tensor = Tv.Compose([Tv.ToTensor()])          # cached images are already 224x224
    mean, std = MEAN.to(dev), STD.to(dev)
    out = {}
    for name, (ids, y, paths) in eval_sets().items():
        dl = torch.utils.data.DataLoader(_ImgList(paths, to_tensor), batch_size=bs, num_workers=3)
        allp = [[] for _ in range(passes)]
        gen = torch.Generator(device=dev).manual_seed(2026)
        for x in dl:
            x = x.to(dev)
            for s in range(passes):
                xa = gpu_augment(x, gen)
                allp[s].append(model((xa - mean) / std).float().cpu())
        out[f"{name}_tta_logits"] = np.stack([torch.cat(a).numpy() for a in allp]).astype(np.float32)
        out[f"{name}_y"] = y
        print(f"  {key} {name}: done", flush=True)
    np.savez_compressed(D / f"tta_{key}.npz", **out)


def analyse():
    rows = []
    for key in CKPT:
        f = D / f"tta_{key}.npz"
        if not f.exists():
            continue
        t = np.load(f)
        d = load_dump(PAPER_RUNS[key])
        T = fit_temperature(d["val"]["logits"], d["val"]["y"])
        for split in ["test", "official_test"]:
            y = d[split]["y"]
            P = softmax(t[f"{split}_tta_logits"] / T, 2)
            mean = P.mean(0)
            H = entropy(mean)
            MI = H - entropy(P).mean(0)
            s, p = uncertainty_scores(d[split]["logits"], d[split]["mc_logits"], T)
            if split == "test":
                err = (p["det"].argmax(1) != y).astype(int)
                err_tta = (mean.argmax(1) != y).astype(int)
                rows.append({"model": key, "split": split, "TTA_acc": 1 - err_tta.mean(),
                             "TTA_bal_acc": cls_metrics(y, mean)["balanced_accuracy"],
                             "TTA_ECE": calib_metrics(y, mean)["ECE"],
                             "AUROC_err_MSP": roc_auc_score(err, s["MSP_T"]),
                             "AUROC_err_TTA_entropy": roc_auc_score(err, H),
                             "AUROC_err_TTA_MI": roc_auc_score(err, MI),
                             "AUROC_err_TTA_entropy_owntarget": roc_auc_score(err_tta, H),
                             "AURC_TTA_entropy": aurc(err, H), "AURC_MSP": aurc(err, s["MSP_T"])})
                # rows for the main error-detection table, same format as analyze_seed42 (paired bootstrap)
                if key == "resnet50_E1":
                    from scripts.revision.common import split_groups, cluster_bootstrap_indices, ci
                    _, groups = split_groups("test")
                    tab = []
                    for nm, sc in [("TTA_entropy", H), ("TTA_mutual_info", MI)]:
                        diffs = [roc_auc_score(err[i], sc[i]) - roc_auc_score(err[i], s["MSP_T"][i])
                                 for i in cluster_bootstrap_indices(groups, 2000, seed=7)]
                        lo, hi = ci(diffs)
                        tab.append({"score": nm, "AUROC": roc_auc_score(err, sc),
                                    "diff_vs_MSP_T": roc_auc_score(err, sc) - roc_auc_score(err, s["MSP_T"]),
                                    "ci_low": lo, "ci_high": hi,
                                    "pAUROC": roc_auc_score(err, sc, max_fpr=0.2), "AURC": aurc(err, sc)})
                    pd.DataFrame(tab).to_csv(D / "tta_table_rows_resnet50_E1.csv", index=False)
            else:
                u = (y == 8).astype(int)
                rows.append({"model": key, "split": split,
                             "AUROC_UNK_MSP": roc_auc_score(u, s["MSP_T"]),
                             "AUROC_UNK_MC_entropy": roc_auc_score(u, s["MC_entropy_T"]),
                             "AUROC_UNK_MC_MI": roc_auc_score(u, s["MC_mutual_info_T"]),
                             "AUROC_UNK_TTA_entropy": roc_auc_score(u, H),
                             "AUROC_UNK_TTA_MI": roc_auc_score(u, MI)})
    pd.DataFrame(rows).to_csv(D / "tta_results.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    if "--analyse-only" not in sys.argv:
        torch.cuda.set_per_process_memory_fraction(0.25)
        keys = [a for a in sys.argv[1:] if a in CKPT] or ["resnet50_E1"]
        for k in keys:
            if not (D / f"tta_{k}.npz").exists():
                tta_logits(k)
    analyse()
