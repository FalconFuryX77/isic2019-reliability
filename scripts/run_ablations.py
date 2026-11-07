#!/usr/bin/env python
"""
Run ablation studies sequentially: train + evaluate each ablation config.
Ablations (all on ResNet-50, compared against resnet50_baseline):
  A) no_aug          - augmentation disabled
  B) no_weights      - no class weighting (plain CE)
  C) single_phase    - skip frozen phase, fine-tune 25 epochs
  D) small_head      - smaller head (32 dense, 0.3 dropout, no BN)
  E) highres         - 256px input instead of 224
  F) no_dropout      - head dropout set to 0.0
Usage:
    python scripts/run_ablations.py [--model resnet50]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ABLATIONS = {
    "ablation_no_aug": "no_aug",
    "ablation_no_weights": "no_weights",
    "ablation_single_phase": "single_phase",
    "ablation_small_head": "small_head",
    "ablation_highres": "highres",
    "ablation_no_dropout": "no_dropout",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--only", default=None, help="run a single ablation key, e.g. no_aug")
    args = parser.parse_args()

    project = Path(__file__).resolve().parent.parent
    py = str(project / ".venv" / "Scripts" / "python.exe")
    env = dict(os.environ)

    items = [(k, v) for k, v in ABLATIONS.items()]
    if args.only:
        items = [it for it in items if it[1] == args.only]
        if not items:
            print(f"[ERROR] unknown ablation: {args.only}")
            sys.exit(1)

    for cfg_name, tag in items:
        exp_id = f"{args.model}_abl_{tag}"
        print(f"\n{'=' * 62}\n  ABLATION: {tag}  ({cfg_name})  ->  {exp_id}\n{'=' * 62}",
              flush=True)

        train = subprocess.run(
            [py, "-u", str(project / "scripts" / "run_train.py"),
             "--model", args.model, "--config", str(project / "configs" / f"{cfg_name}.yaml"),
             "--exp-id", exp_id],
            env=env, capture_output=True, text=True)
        print(train.stdout)
        if train.returncode != 0:
            print("[ERROR] training failed:\n", train.stderr)
            sys.exit(1)

        eval_ = subprocess.run(
            [py, "-u", str(project / "scripts" / "run_evaluate.py"),
             "--model", args.model, "--exp-id", exp_id],
            env=env, capture_output=True, text=True)
        print(eval_.stdout)
        if eval_.returncode != 0:
            print("[ERROR] evaluation failed:\n", eval_.stderr)
            sys.exit(1)

    print(f"\n{'=' * 62}\n  ALL ABLATIONS COMPLETE\n{'=' * 62}")


if __name__ == "__main__":
    main()