#!/usr/bin/env python
"""
Build a lossless 224x224 image cache (I/O optimisation only).

The training/eval transform is  Resize((224,224), BICUBIC) -> [augment] -> ToTensor -> Normalize.
Resizing is the first, deterministic step, so we store its exact output as PNG.
At load time Resize((224,224)) on a 224x224 PIL image is an identity copy (Pillow
returns self.copy() when the size is unchanged), so the tensors fed to the network
are bit-identical to the uncached pipeline. `--verify` checks this on a sample.

Usage:
    python scripts/build_image_cache.py --size 224 --workers 10
    python scripts/build_image_cache.py --size 224 --verify 200
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SOURCES = {
    "train": ROOT / "data" / "raw" / "ISIC_2019_Training_Input",
    "official_test": ROOT / "data" / "raw" / "ISIC_2019_Test_Input",
}


def _resize_one(args):
    src, dst, size = args
    if dst.exists() and dst.stat().st_size > 0:
        return 0
    import torchvision.transforms as T
    from PIL import Image
    with Image.open(src) as im:
        img = im.convert("RGB")
    out = T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC)(img)
    tmp = dst.with_suffix(".tmp.png")
    out.save(tmp, format="PNG", compress_level=1)
    os.replace(tmp, dst)
    return 1


def build(size, workers):
    jobs = []
    for split, src_dir in SOURCES.items():
        dst_dir = ROOT / "data" / "processed" / f"cache{size}" / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(src_dir.glob("*.jpg")):
            jobs.append((p, dst_dir / (p.stem + ".png"), size))
    print(f"{len(jobs)} images to cache (existing files are skipped)", flush=True)
    done = 0
    with Pool(workers) as pool:
        for i, n in enumerate(pool.imap_unordered(_resize_one, jobs, chunksize=16), 1):
            done += n
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)} processed, {done} written", flush=True)
    print(f"finished: {done} newly written", flush=True)


def verify(size, n):
    import random
    import torch
    from PIL import Image
    from src.data.dataset import make_transforms
    tf = make_transforms(size, augment=False)
    files = sorted(SOURCES["train"].glob("*.jpg")) + sorted(SOURCES["official_test"].glob("*.jpg"))
    random.Random(0).shuffle(files)
    bad = 0
    for p in files[:n]:
        split = "train" if p.parent == SOURCES["train"] else "official_test"
        c = ROOT / "data" / "processed" / f"cache{size}" / split / (p.stem + ".png")
        with Image.open(p) as im:
            a = tf(im.convert("RGB"))
        with Image.open(c) as im:
            b = tf(im.convert("RGB"))
        if not torch.equal(a, b):
            bad += 1
            print("MISMATCH", p.name, float((a - b).abs().max()))
    print(f"verify: {n - bad}/{n} bit-identical")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--verify", type=int, default=0)
    a = ap.parse_args()
    if a.verify:
        sys.exit(1 if verify(a.size, a.verify) else 0)
    build(a.size, a.workers)
