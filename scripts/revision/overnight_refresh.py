#!/usr/bin/env python
"""
Background refresher. Whenever a new training job finishes (DONE marker):
  1. re-run the multi-seed analysis,
  2. rebuild the PDFs (main/supplementary/response) and the Overleaf zip,
  3. rebuild the public GitHub repo copy; if the leak check is clean, commit + push, and upload new
     prediction files to the GitHub release (existing assets are left untouched).
Exits when the training plan reports FINISHED and all results are refreshed.
Log: results/revision/refresh.log
"""
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCH = ROOT / "runs" / "repeat_batch_20261107"
LOG = ROOT / "results" / "revision" / "refresh.log"
REPO = ROOT / "github_repo"
ASSETS = ROOT / "github_release_assets"
GH_REPO = "FalconFuryX77/isic2019-reliability"
TAG = "v1.0-revision"
PY = sys.executable


def log(msg):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")


def run(cmd, cwd=ROOT):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def done_jobs():
    return sorted(d.name for d in (BATCH / "experiments").glob("seed*_*") if (d / "DONE").exists())


def refresh(new_jobs):
    for script in ["analyze_multiseed.py", "build_pdfs.py", "package_paper.py", "build_github_repo.py"]:
        rc, out = run([PY, str(ROOT / "scripts" / "revision" / script)])
        log(f"{script}: exit {rc}; " + " | ".join(out.splitlines()[-3:]))
        if script == "build_github_repo.py":
            leaks = [l for l in out.splitlines() if l.startswith("leaks:")]
            clean = leaks and ("none" in leaks[0] or leaks[0].strip() == "leaks: ['scripts\\\\revision\\\\build_github_repo.py']")
            if not clean:
                log("leak check not clean -> not pushing")
                return
    rc, status = run(["git", "status", "--porcelain"], cwd=REPO)
    if status.strip():
        msg = (f"Update results after runs: {', '.join(new_jobs)}\n\n"
               "Regenerated analyses, tables, figures and supplementary material.\n\n"
               "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
        run(["git", "add", "-A"], cwd=REPO)
        rc1, o1 = run(["git", "commit", "-q", "-m", msg], cwd=REPO)
        rc2, o2 = run(["git", "push", "-q"], cwd=REPO)
        log(f"git commit {rc1}, push {rc2} {o2[-200:]}")
    rc, existing = run(["gh", "release", "view", TAG, "--repo", GH_REPO, "--json", "assets", "--jq", ".assets[].name"])
    have = set(existing.split())
    new = [str(f) for f in sorted(ASSETS.glob("*")) if f.name not in have]
    if new:
        rc, out = run(["gh", "release", "upload", TAG, "--repo", GH_REPO] + new)
        log(f"release upload {len(new)} files: exit {rc} {out[-200:]}")


def main():
    seen = set(done_jobs())
    log(f"refresher started; {len(seen)} finished jobs already processed")
    while True:
        jobs = set(done_jobs())
        if jobs - seen:
            new = sorted(jobs - seen)
            log(f"new finished jobs: {new}")
            refresh(new)
            seen = jobs
        if (BATCH / "FINISHED").exists() and set(done_jobs()) == seen:
            log("plan finished; all results refreshed; exiting")
            break
        time.sleep(120)


if __name__ == "__main__":
    main()
