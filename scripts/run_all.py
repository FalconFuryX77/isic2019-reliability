#!/usr/bin/env python
"""
Run the full pipeline end-to-end:
  audit -> split -> sanity -> train -> evaluate -> calibrate -> uncertainty -> explain -> selective
Usage:
    python scripts/run_all.py --model resnet50
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run(script, extra=None):
    cmd = [sys.executable, str(ROOT / "scripts" / script)]
    if extra:
        cmd += extra
    print(f"\n{'#'*70}\n  $ {' '.join(cmd)}\n{'#'*70}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="resnet50 or efficientnet_b3")
    parser.add_argument("--exp-id", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--method", default="gradcam")
    args = parser.parse_args()

    exp_id = args.exp_id or f"{args.model}_baseline"

    run("run_audit.py")
    run("run_split.py")
    run("run_sanity.py", ["--model", args.model])
    if not args.skip_train:
        run("run_train.py", ["--model", args.model, "--exp-id", exp_id])
    run("run_evaluate.py", ["--model", args.model, "--exp-id", exp_id])
    run("run_calibrate.py", ["--model", args.model, "--exp-id", exp_id])
    run("run_uncertainty.py", ["--model", args.model, "--exp-id", exp_id])
    run("run_explain.py", ["--model", args.model, "--exp-id", exp_id,
                           "--method", args.method])
    run("run_selective.py", ["--model", args.model, "--exp-id", exp_id])
    run("run_summary.py")

    print("\n" + "=" * 70)
    print("  [OK] FULL PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()