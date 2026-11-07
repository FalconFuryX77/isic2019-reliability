"""Run a list of training jobs from a JSON plan (resumable; same conventions as run_repeat_batch.py).

Each job: {"seed": int, "model": "resnet50"|"efficientnet_b3", "base_config": path,
           "condition": str, "size": int (default 224)}
Jobs are written to runs/repeat_batch_20261107/experiments/seed{seed}_{condition}/ with a DONE
marker; finished jobs are skipped. If another orchestrator is still running (status.json state
"running"), this script first creates STOP for it and waits until it has stopped.
Creates FINISHED when the whole plan is done.  Usage:
    python scripts/run_plan_batch.py runs/repeat_batch_20261107/plan_revision2.json
"""
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "runs" / "repeat_batch_20261107"
CONFIGS, LOGS, EXPERIMENTS = BATCH / "configs", BATCH / "logs", BATCH / "experiments"
STATUS, STEPS, STOP, FINISHED = BATCH / "status.json", BATCH / "steps.jsonl", BATCH / "STOP", BATCH / "FINISHED"
ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def record(event, **fields):
    with STEPS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": dt.datetime.now().astimezone().isoformat(), "event": event, **fields}) + "\n")


def save(status):
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    tmp.replace(STATUS)


def wait_for_previous():
    try:
        st = json.loads(STATUS.read_text())
    except Exception:
        return
    if st.get("state") == "running":
        STOP.write_text("stop requested by run_plan_batch.py", encoding="utf-8")
        record("waiting_for_previous_orchestrator", current_job=st.get("current_job"))
        while json.loads(STATUS.read_text()).get("state") == "running":
            time.sleep(30)
    if STOP.exists():
        STOP.unlink()


def main(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    wait_for_previous()
    if FINISHED.exists():
        FINISHED.unlink()
    status = {"batch": BATCH.name, "plan_file": str(plan_path), "started_at": dt.datetime.now().astimezone().isoformat(),
              "state": "running", "total_jobs": len(plan), "completed_jobs": 0, "failed_jobs": [],
              "durations_min": {}, "plan": plan}
    save(status)
    record("plan_started", plan=str(plan_path), jobs=len(plan))
    try:
        for i, job in enumerate(plan, 1):
            if STOP.exists():
                status["state"] = "stopped_by_user"
                record("stop_requested", before_job=i)
                break
            seed, model, cond, size = job["seed"], job["model"], job["condition"], job.get("size", 224)
            exp_id = f"seed{seed}_{cond}"
            exp_dir = EXPERIMENTS / exp_id
            if (exp_dir / "DONE").exists():
                status["completed_jobs"] += 1
                record("job_skipped_done", exp_id=exp_id)
                save(status)
                continue
            if exp_dir.exists():
                shutil.move(str(exp_dir), str(EXPERIMENTS / f"_incomplete_{exp_id}_{dt.datetime.now():%Y%m%d%H%M%S}"))
            cfg = yaml.safe_load((ROOT / job["base_config"]).read_text(encoding="utf-8"))
            cfg["seed"] = seed
            cfg["paths"]["experiments_dir"] = str(EXPERIMENTS)
            cfg["preprocessing"]["image_size"] = size
            cfg["preprocessing"]["cache_dir"] = f"data/processed/cache{size}/train"
            cfg["training"]["num_workers"] = 4
            cfg_path = CONFIGS / f"{exp_id}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            log_path = LOGS / f"{exp_id}.log"
            status["current_job"] = exp_id
            save(status)
            record("job_started", index=i, total=len(plan), seed=seed, model=model, condition=cond, size=size, exp_id=exp_id)
            t0 = time.time()
            with log_path.open("w", encoding="utf-8") as log:
                log.write(f"plan={plan_path}\njob={i}/{len(plan)}\nseed={seed}\nmodel={model}\ncondition={cond}\nsize={size}\n\n")
                log.flush()
                tr = subprocess.run([str(PY), "scripts/run_train.py", "--model", model, "--config", str(cfg_path),
                                     "--exp-id", exp_id], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=ENV)
                rc_dump = None
                for ck, out in [(f"best_{model}.pt", "dump.npz"), (f"best_true_{model}.pt", "dump_besttrue.npz")]:
                    if (exp_dir / ck).exists():
                        log.write(f"\n\n==== dump_predictions {ck} ====\n")
                        log.flush()
                        d = subprocess.run([str(PY), "scripts/dump_predictions.py", "--ckpt", str(exp_dir / ck),
                                            "--backbone", model, "--out", str(exp_dir / out), "--size", str(size)],
                                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=ENV)
                        rc_dump = rc_dump or d.returncode
            minutes = round((time.time() - t0) / 60, 1)
            ok = tr.returncode == 0 and rc_dump == 0
            res = {"index": i, "seed": seed, "condition": cond, "size": size, "train_exit_code": tr.returncode,
                   "dump_exit_code": rc_dump, "minutes": minutes, "exp_id": exp_id}
            record("job_finished" if ok else "job_failed", **res)
            status["durations_min"][exp_id] = minutes
            status["completed_jobs"] += 1
            if ok:
                (exp_dir / "DONE").write_text(dt.datetime.now().isoformat(), encoding="utf-8")
            else:
                status["failed_jobs"].append(res)
            save(status)
        if status["state"] == "running":
            status["state"] = "completed_with_failures" if status["failed_jobs"] else "completed"
    except BaseException as exc:
        status["state"] = "orchestrator_error"
        status["error"] = repr(exc)
        record("orchestrator_error", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        status.pop("current_job", None)
        status["finished_at"] = dt.datetime.now().astimezone().isoformat()
        save(status)
        record("plan_finished", state=status["state"])
        FINISHED.write_text(status["state"], encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
