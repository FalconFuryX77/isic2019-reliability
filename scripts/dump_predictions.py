#!/usr/bin/env python
"""
Dump raw logits + MC-Dropout logits for a trained checkpoint (inference only).

For each evaluation set (internal validation, internal test, official ISIC 2019
test incl. UNK) we store:
    ids, y            (y = class index 0..7, UNK = 8 on the official test set)
    logits            (N, 8)  deterministic eval-mode logits (raw, T = 1)
    mc_logits         (P, N, 8) logits from P passes with ONLY the head Dropout
                      active (BatchNorm in eval mode) -- identical semantics to
                      src/uncertainty/mc_dropout.py. Backbone features are computed
                      once because the backbone is deterministic in eval mode.
All downstream analyses (temperature fitting, ECE, MC entropy, selective
prediction, conformal sets...) are done from these files, never by retraining.

Usage:
    python scripts/dump_predictions.py --ckpt experiments/resnet50_baseline/best_resnet50.pt \
        --backbone resnet50 --out experiments/resnet50_baseline/dump.npz
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import ISIC_CLASSES  # noqa: E402
from src.data.dataset import make_transforms  # noqa: E402
from src.models.builders import build_model  # noqa: E402

CACHE = ROOT / "data" / "processed" / "cache224"


class _ImgList(torch.utils.data.Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB"))


def eval_sets(size=224):
    CACHE = ROOT / "data" / "processed" / f"cache{size}"
    sets = {}
    cls = {c: i for i, c in enumerate(ISIC_CLASSES)}
    for name, csv in [("val", "validation.csv"), ("test", "test.csv")]:
        df = pd.read_csv(ROOT / "data" / "splits" / csv)
        sets[name] = (df["image_id"].tolist(), np.array([cls[l] for l in df["label"]]),
                      [str(CACHE / "train" / f"{i}.png") for i in df["image_id"]])
    gt = pd.read_csv(ROOT / "data" / "raw" / "ISIC_2019_Test_GroundTruth.csv")
    cols = ISIC_CLASSES + ["UNK"]
    y = gt[cols].values.argmax(1)
    sets["official_test"] = (gt["image"].tolist(), y,
                             [str(CACHE / "official_test" / f"{i}.png") for i in gt["image"]])
    return sets


@torch.no_grad()
def features(model, loader, device):
    feats = []
    for x in loader:
        x = x.to(device, non_blocking=True)
        f = model.pool(model.features(x))
        feats.append(torch.flatten(f, 1).float().cpu())
    return torch.cat(feats)


@torch.no_grad()
def run(ckpt, backbone, out, passes=30, batch_size=64, workers=4, mc_seed=0, sets_wanted=None, size=224):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(backbone_name=backbone, num_classes=len(ISIC_CLASSES), pretrained=False,
                        dense_size=128, dropout=0.5, use_batchnorm=True, device=device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    tf = make_transforms(size, augment=False)
    payload = {"passes": passes, "classes": np.array(ISIC_CLASSES + ["UNK"])}
    for name, (ids, y, paths) in eval_sets(size).items():
        if sets_wanted and name not in sets_wanted:
            continue
        t0 = time.time()
        loader = torch.utils.data.DataLoader(_ImgList(paths, tf), batch_size=batch_size,
                                             shuffle=False, num_workers=workers, pin_memory=True)
        F = features(model, loader, device).to(device)
        model.eval()
        logits = torch.cat([model.head(F[i:i + 4096]) for i in range(0, len(F), 4096)]).cpu().numpy()
        model.enable_dropout(True)
        torch.manual_seed(mc_seed)
        mc = np.stack([torch.cat([model.head(F[i:i + 4096]) for i in range(0, len(F), 4096)]).cpu().numpy()
                       for _ in range(passes)])
        model.enable_dropout(False)
        model.eval()
        payload[f"{name}_ids"] = np.array(ids)
        payload[f"{name}_y"] = y
        payload[f"{name}_logits"] = logits.astype(np.float32)
        payload[f"{name}_mc_logits"] = mc.astype(np.float32)
        print(f"  {name}: N={len(ids)}  acc={np.mean(logits.argmax(1) == y):.4f}  ({time.time() - t0:.0f}s)",
              flush=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    print(f"  saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--size", type=int, default=224)
    a = ap.parse_args()
    run(a.ckpt, a.backbone, a.out, passes=a.passes, workers=a.workers, size=a.size)
