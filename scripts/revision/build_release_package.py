#!/usr/bin/env python
"""
Assemble a self-contained reproducibility package (local folder only; nothing is
uploaded). Contents: code, configs, lesion-grouped split files, per-image
predictions (logits + MC-Dropout logits) for every run, all revision result
tables/figures, and a README explaining how each table/figure is regenerated.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "release_package"


def copytree(src, dst, patterns=("*",), exclude=("__pycache__",)):
    for p in Path(src).rglob("*"):
        if p.is_dir() or any(e in p.parts for e in exclude):
            continue
        if not any(p.match(pat) for pat in patterns):
            continue
        out = Path(dst) / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)


def main():
    if PKG.exists():
        shutil.rmtree(PKG)
    PKG.mkdir()
    copytree(ROOT / "src", PKG / "src", ("*.py",))
    copytree(ROOT / "scripts", PKG / "scripts", ("*.py",))
    copytree(ROOT / "configs", PKG / "configs", ("*.yaml",))
    copytree(ROOT / "data" / "splits", PKG / "data" / "splits", ("*.csv", "*.json"))
    copytree(ROOT / "results" / "revision", PKG / "results" / "revision",
             ("*.csv", "*.json", "*.png", "*.pdf", "*.md"), exclude=("__pycache__", "hashes.csv"))
    shutil.copy2(ROOT / "requirements.txt", PKG / "requirements.txt")
    preds = PKG / "predictions"
    preds.mkdir()
    for name, d in [("seed42_resnet50_E1", "experiments/resnet50_baseline"),
                    ("seed42_resnet50_E0", "experiments/resnet50_e0"),
                    ("seed42_effnetb3_E1", "experiments/efficientnet_b3_baseline")]:
        if (ROOT / d / "dump.npz").exists():
            shutil.copy2(ROOT / d / "dump.npz", preds / f"{name}.npz")
    batch = ROOT / "runs" / "repeat_batch_20261107" / "experiments"
    for d in sorted(batch.glob("seed*_*")):
        if (d / "DONE").exists():
            for f in ["dump.npz", "dump_besttrue.npz"]:
                if (d / f).exists():
                    shutil.copy2(d / f, preds / f"{d.name}{'' if f == 'dump.npz' else '_besttrue'}.npz")
            for f in ["training_history.csv", "class_weights.json", "true_best_epoch.json", "experiment_config.json"]:
                if (d / f).exists():
                    (PKG / "runs" / d.name).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(d / f, PKG / "runs" / d.name / f)
    for f in ["train_resnet50.log", "e0_train.log", "train_efficientnet.log"]:
        (PKG / "logs").mkdir(exist_ok=True)
        shutil.copy2(ROOT / "logs" / f, PKG / "logs" / f)
    (PKG / "README.md").write_text(README, encoding="utf-8")
    size = sum(p.stat().st_size for p in PKG.rglob("*") if p.is_file()) / 1e6
    print(f"package at {PKG} ({size:.1f} MB)")


README = """# Reproducibility package: Confidence Calibration and Uncertainty-Aware Prediction for Reliable Skin Lesion Classification

Images are NOT included (ISIC licence): download the ISIC 2019 training and test data from
https://challenge.isic-archive.com/data/#2019 into `data/raw/` (same layout as in `configs/baseline.yaml`).

## Splits
`data/splits/{train,validation,test}.csv` list image id, label and lesion id of the lesion-grouped
80/10/10 split (20,235 / 2,573 / 2,523 images). No lesion group occurs in two files.

## Predictions
`predictions/*.npz` contain, for the internal validation, internal test and official ISIC 2019 test images:
`<split>_ids`, `<split>_y` (UNK = 8), `<split>_logits` (raw logits) and `<split>_mc_logits`
(30 MC-Dropout passes, head dropout only). `seed42_*` are the models of the paper; `seed4X_*` the additional
seeds; `*_besttrue` the best-validation-epoch checkpoints (sensitivity analysis).

## Regenerating every table and figure (no GPU needed)
```
python scripts/revision/analyze_seed42.py        # calibration, uncertainty, selective, SGR, conformal
python scripts/revision/analyze_multiseed.py     # mean +- SD over seeds, paired tests, ensembles
python scripts/revision/analyze_official_test.py # shift + UNK evaluation on the official test set
python scripts/revision/figures.py
python scripts/revision/make_tex_assets.py       # LaTeX tables + number macros used by the manuscript
```
(The analysis scripts read the prediction dumps from the original experiment folders; adjust the paths in
`scripts/revision/common.py` to point at `predictions/` if you use this package stand-alone.)

## Training
`python scripts/run_train.py --model resnet50 --config configs/baseline.yaml` (E1, class-weighted) or
`configs/ablation_e0.yaml` (E0). Additional seeds: `scripts/run_repeat_batch.py`. Logits for a checkpoint:
`scripts/dump_predictions.py`. Duplicate audit: `scripts/revision/dataset_integrity.py`.
Grad-CAM checks: `scripts/revision/gradcam_checks.py`.
"""

if __name__ == "__main__":
    main()
