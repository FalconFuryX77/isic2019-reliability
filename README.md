# Reliability evaluation of skin-lesion classifiers on ISIC 2019

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
