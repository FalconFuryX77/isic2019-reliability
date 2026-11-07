#!/usr/bin/env python
"""
Dermoscopic-artefact check (reviewer: "dermoscopic artefacts (vignettes, rulers) are not discussed").

A dark circular vignette (the field stop of the dermatoscope) is detected automatically from the lossless
224 px cache: an image is flagged when at least 3 of its 4 corner patches (16x16 px) have a mean grey level
below 35/255. Rulers, ink marks and hair cannot be detected reliably without manual annotation and are not
quantified.

For the submitted seed-42 ResNet-50 E1 model (temperature and tau fitted on validation) we report accuracy,
calibrated ECE, coverage at the 85% validation target and error AUROC separately for images with and without a
vignette: on the internal test split (pooled and within BCN_20000, the source in which vignettes are common)
and on the known classes of the official test set.

Outputs: results/revision/artefacts/{vignette_per_image.csv, vignette_summary.csv, vignette_examples.png}
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common import ROOT, OUT, PAPER_RUNS, load_dump, fit_temperature, ece_equal_width  # noqa: E402
from src.evaluation.selective import _threshold_for_coverage  # noqa: E402

CACHE = ROOT / "data" / "processed" / "cache224"
D = OUT / "artefacts"
D.mkdir(parents=True, exist_ok=True)
PATCH, DARK = 16, 35.0
UNK = 8


def corner_means(path):
    g = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    p = PATCH
    return [g[:p, :p].mean(), g[:p, -p:].mean(), g[-p:, :p].mean(), g[-p:, -p:].mean()]


def flag_images(ids, sub):
    rows = []
    for i in ids:
        c = corner_means(CACHE / sub / f"{i}.png")
        rows.append({"image_id": i, "n_dark_corners": int(sum(v < DARK for v in c)), "min_corner": min(c)})
    df = pd.DataFrame(rows)
    df["vignette"] = df.n_dark_corners >= 3
    return df


def group_metrics(y, p, tau, k):
    pred = p.argmax(1)
    conf = p.max(1)
    acc = conf >= tau
    err = (pred[k] != y[k]).astype(int)
    return {"n": int(k.sum()), "accuracy": float((pred[k] == y[k]).mean()),
            "ECE_T": float(ece_equal_width(y[k], p[k])), "coverage": float(acc[k].mean()),
            "accepted_accuracy": float((pred[k & acc] == y[k & acc]).mean()) if (k & acc).any() else np.nan,
            "error_AUROC_MSP": float(roc_auc_score(err, 1 - conf[k])) if 0 < err.sum() < len(err) else np.nan}


def main():
    d = load_dump(PAPER_RUNS["resnet50_E1"])
    T = fit_temperature(d["val"]["logits"], d["val"]["y"])
    tau = _threshold_for_coverage(softmax(d["val"]["logits"] / T, 1).max(1), 0.85)

    # ---------------- internal test
    split = pd.read_csv(ROOT / "data" / "splits" / "test.csv")
    assert list(split.image_id) == [str(x) for x in d["test"]["ids"]], "dump order differs from test.csv"
    lid = split["lesion_id"].fillna("")
    source = np.where(lid.str.startswith("HAM_"), "HAM10000",
                      np.where(lid.str.startswith("BCN_"), "BCN_20000", "MSK"))
    ft = flag_images(split.image_id, "train")
    ft["split"], ft["source"] = "internal_test", source
    y, p = d["test"]["y"], softmax(d["test"]["logits"] / T, 1)
    v = ft.vignette.values
    rows = []
    for scope, base in [("internal test, all sources", np.ones(len(y), bool)),
                        ("internal test, BCN_20000 only", source == "BCN_20000")]:
        for flag in [True, False]:
            r = group_metrics(y, p, tau, base & (v == flag))
            rows.append({"scope": scope, "vignette": flag, **r})
    prev = pd.crosstab(ft.source, ft.vignette, normalize="index")

    # ---------------- official test (known classes; UNK acceptance reported separately)
    o = d["official_test"]
    fo = flag_images([str(x) for x in o["ids"]], "official_test")
    fo["split"], fo["source"] = "official_test", "official"
    yo, po = o["y"], softmax(o["logits"] / T, 1)
    vo = fo.vignette.values
    known = yo < UNK
    for flag in [True, False]:
        k = known & (vo == flag)
        r = group_metrics(np.where(known, yo, 0), po, tau, k)
        u = (~known) & (vo == flag)
        r["UNK_share"] = float(u.sum() / max(1, (vo == flag).sum()))
        r["UNK_accepted"] = float((po[u].max(1) >= tau).mean()) if u.any() else np.nan
        rows.append({"scope": "official test, known classes", "vignette": flag, **r})

    pd.concat([ft, fo]).to_csv(D / "vignette_per_image.csv", index=False)
    summ = pd.DataFrame(rows)
    summ.to_csv(D / "vignette_summary.csv", index=False)
    prev.to_csv(D / "vignette_prevalence_by_source.csv")
    print(f"T={T:.3f} tau={tau:.4f}")
    print("vignette prevalence by source (internal test):\n", prev.round(3))
    print("official test vignette prevalence:", round(float(vo.mean()), 3))
    print(summ.round(3).to_string())

    # ---------------- contact sheet for visual validation of the detector (fixed seed)
    rng = np.random.default_rng(0)
    pick = [("vignette", ft[ft.vignette].image_id.values), ("no vignette", ft[~ft.vignette].image_id.values)]
    tiles = []
    for _, ids in pick:
        sel = rng.choice(ids, size=min(12, len(ids)), replace=False)
        tiles.append([Image.open(CACHE / "train" / f"{i}.png").convert("RGB").resize((112, 112)) for i in sel])
    sheet = Image.new("RGB", (112 * 12, 112 * 2), "white")
    for r_, row in enumerate(tiles):
        for c_, im in enumerate(row):
            sheet.paste(im, (c_ * 112, r_ * 112))
    sheet.save(D / "vignette_examples.png")


if __name__ == "__main__":
    main()
