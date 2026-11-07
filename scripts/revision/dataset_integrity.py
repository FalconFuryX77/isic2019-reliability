#!/usr/bin/env python
"""
Dataset-integrity audit for the revision (no training).

1. Source attribution (HAM10000 / BCN_20000 / MSK) and lesion-ID coverage per split.
2. Exact duplicates: MD5 of the original JPEG bytes and of the decoded 224 px pixels.
3. Near-duplicates: 64-bit DCT perceptual hash (pHash) Hamming distance AND cosine
   similarity of ImageNet ResNet-50 embeddings (independent of our trained models).
   Cross-split candidate pairs are written out with a contact sheet for visual review.
4. Same checks between the internal data (all 25,331) and the official ISIC 2019 test set.

Outputs -> results/revision/integrity/
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dct

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.revision.common import ROOT, OUT, save_json  # noqa: E402

D = OUT / "integrity"
D.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "data" / "processed" / "cache224"


def source_of(row):
    lid = row["lesion_id"]
    if isinstance(lid, str) and lid.startswith("HAM_"):
        return "HAM10000"
    if isinstance(lid, str) and lid.startswith("BCN_"):
        return "BCN_20000"
    return "MSK"


def load_internal():
    frames = []
    for split, csv in [("train", "train.csv"), ("val", "validation.csv"), ("test", "test.csv")]:
        df = pd.read_csv(ROOT / "data" / "splits" / csv)
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["source"] = df.apply(source_of, axis=1)
    df["has_lesion_id"] = df["lesion_id"].notna()
    return df


def md5_file(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def phash_and_pixmd5(png):
    with Image.open(png) as im:
        rgb = np.asarray(im.convert("RGB"))
    pix = hashlib.md5(rgb.tobytes()).hexdigest()
    g = Image.fromarray(rgb).convert("L").resize((32, 32), Image.LANCZOS)
    a = np.asarray(g, dtype=np.float64)
    c = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    bits = (c > np.median(c)).flatten()
    return pix, np.packbits(bits).view(np.uint64)[0]


def _hash_worker(args):
    split_dir, image_id, raw = args
    pix, ph = phash_and_pixmd5(CACHE / split_dir / f"{image_id}.png")
    return image_id, md5_file(raw), pix, int(ph)


POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_pairs(ha, hb, thr, same=False):
    """All (i, j, d) with Hamming(ha[i], hb[j]) <= thr."""
    a8 = ha.view(np.uint8).reshape(-1, 8)
    b8 = hb.view(np.uint8).reshape(-1, 8)
    out = []
    for s in range(0, len(a8), 512):
        x = a8[s:s + 512, None, :] ^ b8[None, :, :]
        d = POP[x].sum(-1)
        ii, jj = np.nonzero(d <= thr)
        for i, j in zip(ii, jj):
            gi = s + i
            if same and j <= gi:
                continue
            out.append((gi, j, int(d[i, j])))
    return out


class _PathDS:
    def __init__(self, paths):
        from src.data.dataset import make_transforms
        self.paths, self.tf = paths, make_transforms(224, augment=False)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB"))


def embeddings(paths, batch=128):
    import torch
    import torchvision.models as tvm
    from src.data.dataset import make_transforms
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    m.eval().to(dev)
    dl = torch.utils.data.DataLoader(_PathDS(paths), batch_size=batch, num_workers=3)
    feats = []
    with torch.no_grad():
        for x in dl:
            f = m(x.to(dev))
            feats.append(torch.nn.functional.normalize(f, dim=1).half().cpu())
    return torch.cat(feats)


def cosine_pairs(fa, fb, thr, same=False):
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = fb.to(dev)
    out = []
    for s in range(0, len(fa), 2048):
        sim = fa[s:s + 2048].to(dev) @ B.T
        ii, jj = torch.nonzero(sim >= thr, as_tuple=True)
        vals = sim[ii, jj].float().cpu().numpy()
        for i, j, v in zip(ii.cpu().numpy(), jj.cpu().numpy(), vals):
            gi = s + i
            if same and j <= gi:
                continue
            out.append((gi, int(j), float(v)))
    return out


def contact_sheet(pairs, paths_a, paths_b, labels, path, n=40):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pairs = pairs[:n]
    if not pairs:
        return
    cols = 4
    rows = int(np.ceil(len(pairs) / cols))
    fig, ax = plt.subplots(rows, cols * 2, figsize=(cols * 4, rows * 2.2))
    ax = np.atleast_2d(ax)
    for k in range(rows * cols):
        r, c = divmod(k, cols)
        for t in range(2):
            ax[r, 2 * c + t].axis("off")
        if k >= len(pairs):
            continue
        i, j, txt = pairs[k]
        for t, p in enumerate([paths_a[i], paths_b[j]]):
            with Image.open(p) as im:
                ax[r, 2 * c + t].imshow(im.convert("RGB"))
        ax[r, 2 * c].set_title(labels[k], fontsize=6)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def main():
    from multiprocessing import Pool
    df = load_internal()
    # ---- 1. sources / lesion ids
    src = df.pivot_table(index="source", columns="split", values="image_id", aggfunc="count").fillna(0).astype(int)
    lid = df.pivot_table(index="split", columns="has_lesion_id", values="image_id", aggfunc="count").fillna(0).astype(int)
    cls_src = df.pivot_table(index="label", columns="source", values="image_id", aggfunc="count").fillna(0).astype(int)
    src.to_csv(D / "source_by_split.csv")
    lid.to_csv(D / "lesion_id_coverage_by_split.csv")
    cls_src.to_csv(D / "class_by_source.csv")
    groups = df.assign(g=df["lesion_id"].fillna("img_" + df["image_id"]))
    imgs_per_lesion = groups.groupby("g").size()
    summary = {"n_images": len(df), "n_with_lesion_id": int(df.has_lesion_id.sum()),
               "n_without_lesion_id": int((~df.has_lesion_id).sum()),
               "sources": df.source.value_counts().to_dict(),
               "without_lesion_id_by_source": df[~df.has_lesion_id].source.value_counts().to_dict(),
               "n_groups": int(imgs_per_lesion.size),
               "groups_with_multiple_images": int((imgs_per_lesion > 1).sum()),
               "max_images_per_lesion": int(imgs_per_lesion.max())}
    # cross-split lesion overlap (re-check)
    gs = {s: set(groups[groups.split == s]["g"]) for s in ["train", "val", "test"]}
    summary["lesion_overlap"] = {"train_val": len(gs["train"] & gs["val"]),
                                 "train_test": len(gs["train"] & gs["test"]),
                                 "val_test": len(gs["val"] & gs["test"])}

    # ---- 2/3. hashes
    off = pd.read_csv(ROOT / "data" / "raw" / "ISIC_2019_Test_GroundTruth.csv")
    jobs = [("train", i, ROOT / "data" / "raw" / "ISIC_2019_Training_Input" / f"{i}.jpg") for i in df.image_id] + \
           [("official_test", i, ROOT / "data" / "raw" / "ISIC_2019_Test_Input" / f"{i}.jpg") for i in off.image]
    if (D / "hashes.csv").exists():
        h = pd.read_csv(D / "hashes.csv")
    else:
        with Pool(6) as pool:
            res = pool.map(_hash_worker, jobs, chunksize=64)
        h = pd.DataFrame(res, columns=["image_id", "md5_file", "md5_pixels", "phash"])
        h["set"] = ["internal"] * len(df) + ["official_test"] * len(off)
        h.to_csv(D / "hashes.csv", index=False)
    hi = h[h.set == "internal"].set_index("image_id").loc[df.image_id].reset_index()
    ho = h[h.set == "official_test"].reset_index(drop=True)

    split_of = df["split"].values
    for col in ["md5_file", "md5_pixels"]:
        dup = hi[hi.duplicated(col, keep=False)]
        summary[f"exact_dup_groups_{col}"] = int(dup[col].nunique())
        summary[f"exact_dup_images_{col}"] = int(len(dup))
        xs = 0
        for _, g in dup.groupby(col):
            if len(set(split_of[g.index])) > 1:
                xs += 1
        summary[f"exact_dup_groups_cross_split_{col}"] = xs
        summary[f"exact_dup_internal_vs_official_{col}"] = int(ho[col].isin(set(hi[col])).sum())

    ph_i = np.array([int(v) for v in hi["phash"].astype(str)], dtype=np.uint64)
    ph_o = np.array([int(v) for v in ho["phash"].astype(str)], dtype=np.uint64)
    pairs_ph = hamming_pairs(ph_i, ph_i, thr=6, same=True)
    # ---- embeddings
    paths_i = [str(CACHE / "train" / f"{i}.png") for i in df.image_id]
    paths_o = [str(CACHE / "official_test" / f"{i}.png") for i in off.image]
    import torch
    if (D / "emb_internal.pt").exists():
        fi, fo = torch.load(D / "emb_internal.pt"), torch.load(D / "emb_official.pt")
    else:
        fi, fo = embeddings(paths_i), embeddings(paths_o)
        torch.save(fi, D / "emb_internal.pt")
        torch.save(fo, D / "emb_official.pt")
    pairs_cos = cosine_pairs(fi, fi, thr=0.95, same=True)
    pairs_cos_o = cosine_pairs(fo, fi, thr=0.95)
    pairs_ph_o = hamming_pairs(ph_o, ph_i, thr=6)

    import torch
    def cos(a, b, A=fi, B=fi):
        return float((A[a].float() * B[b].float()).sum())

    def ham(a, b, A=ph_i, B=ph_i):
        return int(POP[(np.array([A[a] ^ B[b]], dtype=np.uint64).view(np.uint8))].sum())

    rows = {}
    for i, j, d in pairs_ph:
        rows[(i, j)] = {"i": i, "j": j, "phash_dist": d, "cos": cos(i, j)}
    for i, j, c in pairs_cos:
        rows.setdefault((i, j), {"i": i, "j": j, "phash_dist": ham(i, j), "cos": c})
    cand = pd.DataFrame(list(rows.values()))
    if len(cand):
        cand["a"] = df.image_id.values[cand.i]
        cand["b"] = df.image_id.values[cand.j]
        cand["split_a"] = split_of[cand.i]
        cand["split_b"] = split_of[cand.j]
        cand["label_a"] = df.label.values[cand.i]
        cand["label_b"] = df.label.values[cand.j]
        cand["lesion_a"] = df.lesion_id.values[cand.i]
        cand["lesion_b"] = df.lesion_id.values[cand.j]
        cand["src_a"] = df.source.values[cand.i]
        cand["src_b"] = df.source.values[cand.j]
        cand["cross_split"] = cand.split_a != cand.split_b
        cand["strong"] = (cand.phash_dist <= 4) & (cand.cos >= 0.97)
        cand["loose"] = cand.cos >= 0.97
        cand = cand.sort_values(["cross_split", "cos"], ascending=[False, False])
    cand.to_csv(D / "near_duplicate_candidates_internal.csv", index=False)
    xs = cand[cand.cross_split] if len(cand) else cand
    within = cand[~cand.cross_split & cand.strong] if len(cand) else cand
    summary["strict_rule_within_split_pairs"] = int(len(within))
    summary["strict_rule_within_split_same_lesion_frac"] = float((within.lesion_a == within.lesion_b).mean()) if len(within) else None
    summary["near_identical_cross_split_pairs"] = int(((cand.phash_dist <= 2) & (cand.cos >= 0.99) & cand.cross_split).sum())
    lx = cand[cand.cross_split & cand.loose]
    summary["loose_cross_split_pairs"] = int(len(lx))
    summary["loose_cross_split_test_images"] = int(len(set(lx[lx.split_a == "test"].a) | set(lx[lx.split_b == "test"].b)))
    summary["near_dup_candidates_internal"] = int(len(cand))
    summary["near_dup_candidates_cross_split"] = int(len(xs))
    summary["near_dup_strong_cross_split"] = int(xs.strong.sum()) if len(xs) else 0
    if len(xs):
        summary["near_dup_strong_cross_split_by_pair"] = {f"{a}-{b}": int(v) for (a, b), v in xs[xs.strong].groupby(["split_a", "split_b"]).size().items()}
        summary["near_dup_strong_cross_split_by_source"] = {f"{a}-{b}": int(v) for (a, b), v in xs[xs.strong].groupby(["src_a", "src_b"]).size().items()}
        summary["near_dup_strong_cross_split_test_images"] = int(len(set(
            xs[xs.strong & (xs.split_a == "test")].a) | set(xs[xs.strong & (xs.split_b == "test")].b)))
        top = xs.head(40)
        contact_sheet([(r.i, r.j, "") for r in top.itertuples()], paths_i, paths_i,
                      [f"{r.a}({r.split_a},{r.label_a}) | {r.b}({r.split_b},{r.label_b}) h={r.phash_dist} c={r.cos:.3f}"
                       for r in top.itertuples()], D / "cross_split_candidates_top40.png")
    # official test vs internal
    rows_o = {}
    for i, j, c in pairs_cos_o:
        rows_o[(i, j)] = {"i": i, "j": j, "cos": c, "phash_dist": ham(i, j, ph_o, ph_i)}
    for i, j, d in pairs_ph_o:
        rows_o.setdefault((i, j), {"i": i, "j": j, "cos": cos(i, j, fo, fi), "phash_dist": d})
    co = pd.DataFrame(list(rows_o.values()))
    if len(co):
        co["official"] = off.image.values[co.i]
        co["internal"] = df.image_id.values[co.j]
        co["internal_split"] = split_of[co.j]
        co["strong"] = (co.phash_dist <= 4) & (co.cos >= 0.97)
        co["loose"] = co.cos >= 0.97
        co = co.sort_values("cos", ascending=False)
        top = co.head(40)
        contact_sheet([(r.i, r.j, "") for r in top.itertuples()], paths_o, paths_i,
                      [f"{r.official} | {r.internal}({r.internal_split}) h={r.phash_dist} c={r.cos:.3f}"
                       for r in top.itertuples()], D / "official_vs_internal_top40.png")
    co.to_csv(D / "near_duplicate_candidates_official_vs_internal.csv", index=False)
    summary["official_vs_internal_candidates"] = int(len(co))
    summary["official_vs_internal_strong"] = int(co.strong.sum()) if len(co) else 0
    summary["official_images_with_strong_match"] = int(co[co.strong].official.nunique()) if len(co) else 0
    save_json(summary, D / "integrity_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
