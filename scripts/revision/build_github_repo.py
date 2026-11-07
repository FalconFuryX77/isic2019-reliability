#!/usr/bin/env python
"""
Assemble the public GitHub repository in <project>/github_repo (code, configs, lesion-grouped splits,
revision results, supplementary PDF) and a folder of release assets (per-image predictions and the
seed-42 checkpoints) in <project>/github_release_assets. Absolute local paths are rewritten to
relative ones; images, caches, embeddings and virtual environments are never copied.
"""
import re
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "github_repo"
ASSETS = ROOT / "github_release_assets"
ABS = [str(ROOT) + "\\", str(ROOT).replace("\\", "/") + "/", str(ROOT).replace("\\", "\\\\") + "\\\\",
       str(ROOT).replace("\\", "\\\\"), str(ROOT).replace("\\", "/"), str(ROOT)]
LEAK = re.compile("|".join(re.escape(x) for x in [ROOT.parts[1], "AppData"]) + r"|[A-Z]:\\\\?Users")
TEXT_EXT = {".py", ".yaml", ".yml", ".json", ".csv", ".md", ".txt", ".log", ".tex"}


def scrub(text):
    for a in ABS:
        text = text.replace(a, "")
    return text


def put(src, rel):
    dst = REPO / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in TEXT_EXT:
        dst.write_text(scrub(src.read_text(encoding="utf-8", errors="ignore")), encoding="utf-8")
    else:
        shutil.copy2(src, dst)


def main():
    if REPO.exists():
        keep_git = REPO / ".git"
        for p in REPO.iterdir():
            if p != keep_git:
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    REPO.mkdir(exist_ok=True)
    for d, pats in [("src", ["*.py"]), ("scripts", ["*.py"]), ("configs", ["*.yaml"])]:
        for pat in pats:
            for f in (ROOT / d).rglob(pat):
                if "__pycache__" not in f.parts:
                    put(f, f.relative_to(ROOT))
    # splits with relative image paths
    for csv in ["train.csv", "validation.csv", "test.csv"]:
        df = pd.read_csv(ROOT / "data" / "splits" / csv)
        df["image_path"] = "data/raw/ISIC_2019_Training_Input/" + df["image_id"] + ".jpg"
        (REPO / "data" / "splits").mkdir(parents=True, exist_ok=True)
        df.to_csv(REPO / "data" / "splits" / csv, index=False)
    put(ROOT / "data" / "splits" / "split_summary.json", Path("data/splits/split_summary.json"))
    # revision results (tables/figures/json), no logs, embeddings or prediction arrays
    for f in (ROOT / "results" / "revision").rglob("*"):
        if f.is_file() and f.suffix.lower() in {".csv", ".json", ".png", ".pdf", ".md"} and f.name != "hashes.csv":
            put(f, f.relative_to(ROOT))
    # training records of every run (no checkpoints)
    batch = ROOT / "runs" / "repeat_batch_20261107"
    for f in [batch / "plan_revision2.json"] + list((batch / "configs").glob("*.yaml")):
        put(f, Path("runs") / f.relative_to(batch))
    for d in sorted((batch / "experiments").glob("seed*_*")):
        if (d / "DONE").exists():
            for n in ["training_history.csv", "class_weights.json", "true_best_epoch.json", "experiment_config.json"]:
                if (d / n).exists():
                    put(d / n, Path("runs") / "experiments" / d.name / n)
            if (batch / "logs" / f"{d.name}.log").exists():
                put(batch / "logs" / f"{d.name}.log", Path("runs") / "logs" / f"{d.name}.log")
    for exp in ["resnet50_baseline", "resnet50_e0", "efficientnet_b3_baseline"]:
        for n in ["class_weights.json", "experiment_config.json", "calibration_metrics.json", "selective_summary.json"]:
            if (ROOT / "experiments" / exp / n).exists():
                put(ROOT / "experiments" / exp / n, Path("experiments") / exp / n)
    for n in ["train_resnet50.log", "e0_train.log", "train_efficientnet.log"]:
        put(ROOT / "logs" / n, Path("logs") / n)
    put(ROOT / "requirements.txt", Path("requirements.txt"))
    sup = ROOT / "paper_revision" / "build" / "supplementary.pdf"
    if sup.exists():
        (REPO / "paper").mkdir(exist_ok=True)
        shutil.copy2(sup, REPO / "paper" / "supplementary_material.pdf")
    (REPO / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    (REPO / "README.md").write_text(README, encoding="utf-8")
    # leak check
    leaks = []
    for f in REPO.rglob("*"):
        if f.is_file() and ".git" not in f.parts and f.suffix.lower() in TEXT_EXT:
            t = f.read_text(encoding="utf-8", errors="ignore")
            if LEAK.search(t):
                leaks.append(str(f.relative_to(REPO)))
    # release assets
    if ASSETS.exists():
        shutil.rmtree(ASSETS)
    ASSETS.mkdir()
    for name, src in [("seed42_resnet50_E1", "experiments/resnet50_baseline"), ("seed42_resnet50_E0", "experiments/resnet50_e0"),
                      ("seed42_effnetb3_E1", "experiments/efficientnet_b3_baseline")]:
        shutil.copy2(ROOT / src / "dump.npz", ASSETS / f"predictions_{name}.npz")
    for ck, name in [("experiments/resnet50_baseline/best_resnet50.pt", "checkpoint_seed42_resnet50_E1.pt"),
                     ("experiments/resnet50_e0/best_resnet50.pt", "checkpoint_seed42_resnet50_E0.pt"),
                     ("experiments/efficientnet_b3_baseline/best_efficientnet_b3.pt", "checkpoint_seed42_effnetb3_E1.pt")]:
        shutil.copy2(ROOT / ck, ASSETS / name)
    for d in sorted((batch / "experiments").glob("seed*_*")):
        if (d / "DONE").exists():
            for f, suffix in [("dump.npz", ""), ("dump_besttrue.npz", "_besttrue")]:
                if (d / f).exists():
                    shutil.copy2(d / f, ASSETS / f"predictions_{d.name}{suffix}.npz")
    tta = ROOT / "results" / "revision" / "tta" / "tta_resnet50_E1.npz"
    if tta.exists():
        shutil.copy2(tta, ASSETS / "tta_logits_seed42_resnet50_E1.npz")
    n_files = sum(1 for f in REPO.rglob("*") if f.is_file() and ".git" not in f.parts)
    size = sum(f.stat().st_size for f in REPO.rglob("*") if f.is_file() and ".git" not in f.parts) / 1e6
    asize = sum(f.stat().st_size for f in ASSETS.iterdir()) / 1e6
    print(f"repo: {n_files} files, {size:.1f} MB; release assets: {len(list(ASSETS.iterdir()))} files, {asize:.0f} MB")
    print("leaks:", leaks or "none")


GITIGNORE = """# data and large artefacts are not versioned
data/raw/
data/processed/
experiments/**/*.pt
runs/**/*.pt
*.npz
.venv/
__pycache__/
*.pyc
"""

README = """# Reliability evaluation of skin-lesion classifiers on ISIC 2019

Code, data splits and results for the paper **"Confidence Calibration and Uncertainty-Aware Prediction for
Reliable Skin Lesion Classification"** (R. Mahajan, P. Paliwal, Z. Shah, R. K. Nagra; NMIMS Mumbai).

The study evaluates discrimination, calibration, error detection, selective prediction and conformal
prediction of ResNet-50 / EfficientNet-B3 classifiers on ISIC 2019 under a lesion-grouped, duplicate-screened
split, over several training seeds, with the official ISIC 2019 test set (including the unknown class) as an
untouched confirmatory evaluation. The supplementary material is in [`paper/supplementary_material.pdf`](paper/supplementary_material.pdf).

## Contents

| Path | What |
|---|---|
| `data/splits/` | Lesion-grouped 80/10/10 split (image id, label, lesion id); no lesion group appears in two files |
| `src/` | Data pipeline, models (ResNet-50 / EfficientNet-B3 with MC-Dropout head), training, calibration, selective prediction |
| `scripts/` | Training (`run_train.py`, `run_plan_batch.py`), inference dumps (`dump_predictions.py`), image cache |
| `scripts/revision/` | All analyses of the paper: calibration, uncertainty, selective/conformal prediction, official-test evaluation, duplicate audit, Grad-CAM checks, figures and LaTeX tables |
| `results/revision/` | Every table and figure as CSV/JSON/PNG |
| `runs/`, `experiments/`, `logs/` | Per-run configurations, class weights, training histories and logs |

Per-image predictions (raw logits and 30 MC-Dropout passes for the internal validation/test splits and the
official test set) for every run, and the checkpoints of the three seed-42 models, are attached to the
GitHub release of this repository.

## Reproducing

1. Download the ISIC 2019 training and test data (images and ground truth) from
   <https://challenge.isic-archive.com/data/#2019> into `data/raw/` (`ISIC_2019_Training_Input/`,
   `ISIC_2019_Test_Input/` and the CSV files). The images are licensed by ISIC (CC BY-NC) and are not redistributed here.
2. `pip install -r requirements.txt` (PyTorch with CUDA recommended).
3. Optional speed-up: `python scripts/build_image_cache.py --size 224` (lossless, bit-identical inputs).
4. Train: `python scripts/run_train.py --model resnet50 --config configs/baseline.yaml` (E1, class-weighted)
   or `configs/ablation_e0.yaml` (E0, plain cross-entropy). Multi-seed plans: `scripts/run_plan_batch.py`.
5. Export predictions: `python scripts/dump_predictions.py --ckpt <checkpoint> --backbone resnet50 --out dump.npz`.
6. Analyses (CPU only, from the prediction files):
   `scripts/revision/analyze_seed42.py`, `analyze_multiseed.py`, `analyze_official_test.py`,
   `dataset_integrity.py`, `gradcam_checks.py`, `tta_uncertainty.py`, `figures.py`, `make_tex_assets.py`.
   The analysis scripts expect prediction files at the paths used during the study
   (`experiments/<run>/dump.npz`, `runs/repeat_batch_20261107/experiments/<run>/dump.npz`); place the release
   files there or adapt `scripts/revision/common.py`.

## Notes

* Seed 42 corresponds to the models of the submitted manuscript. Because of a checkpoint-logic error, their
  stored weights are those of the early-stopping epoch; the corrected code also stores the best-epoch weights
  (`best_true_*.pt`), and both are analysed in the paper.
* Every number in the paper is produced by `scripts/revision/make_tex_assets.py` from the result files.
"""

if __name__ == "__main__":
    main()
