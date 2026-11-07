#!/usr/bin/env python
"""
Background refresher: whenever a new training job of the repeat batch finishes
(DONE marker), re-run the multi-seed analysis, regenerate all LaTeX assets, lint,
rebuild the Overleaf zip and the reproducibility package. Exits when the batch
reports a terminal state and everything is up to date. Log: results/revision/refresh.log
"""
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "runs" / "repeat_batch_20261107"
LOG = ROOT / "results" / "revision" / "refresh.log"
PY = sys.executable


def log(msg):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")


def done_jobs():
    return sorted(d.name for d in (BATCH / "experiments").glob("seed*_*") if (d / "DONE").exists())


def refresh():
    for script in ["analyze_multiseed.py", "package_paper.py", "build_release_package.py"]:
        r = subprocess.run([PY, str(ROOT / "scripts" / "revision" / script)], cwd=ROOT,
                           capture_output=True, text=True)
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        log(f"{script}: exit {r.returncode}; " + " | ".join(tail))


def main():
    seen = set()
    log("refresher started")
    while True:
        jobs = set(done_jobs())
        if jobs - seen:
            log(f"new finished jobs: {sorted(jobs - seen)}")
            refresh()
            seen = jobs
        if (BATCH / "FINISHED").exists() and set(done_jobs()) == seen:
            log("plan finished; all results refreshed; exiting")
            break
        time.sleep(120)


if __name__ == "__main__":
    main()
