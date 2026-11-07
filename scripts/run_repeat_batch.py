"""Multi-seed retraining batch for the reviewer revision (resumable).

* Same training protocol as the paper (configs/baseline.yaml = E1 weighted,
  configs/ablation_e0.yaml = E0 plain CE); only the seed changes.
* Seed 42 is the original paper run (experiments/resnet50_baseline, resnet50_e0,
  efficientnet_b3_baseline), so this batch trains seeds 43-46.
* Engineering-only changes (no effect on the method): images are read from the
  lossless 224 px cache (scripts/build_image_cache.py, bit-identical network
  inputs) and the DataLoader uses 4 workers instead of 2.
* Training touches train + validation only. After each job the checkpoint is
  passed to scripts/dump_predictions.py (inference-only logits/MC logits on
  validation, internal test and official ISIC 2019 test).
* Order is seed-major so that stopping at any point leaves balanced conditions.
* Resumable: finished jobs carry a DONE marker and are skipped. Create the file
  runs/repeat_batch_20261107/STOP to stop cleanly after the current job.
"""
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "runs" / "repeat_batch_20261107"
CONFIGS = BATCH / "configs"
LOGS = BATCH / "logs"
EXPERIMENTS = BATCH / "experiments"
SEEDS = [43, 44, 45, 46]
CONDITIONS = [
    ("resnet50", "configs/baseline.yaml", "resnet50_weighted"),
    ("resnet50", "configs/ablation_e0.yaml", "resnet50_unweighted"),
    ("efficientnet_b3", "configs/baseline.yaml", "efficientnet_b3_weighted"),
]
CACHE_DIR = "data/processed/cache224/train"
ENV = {**os.environ, "PYTHONUNBUFFERED": "1"}
NUM_WORKERS = 4

for directory in (CONFIGS, LOGS, EXPERIMENTS):
    directory.mkdir(parents=True, exist_ok=True)
status_path = BATCH / "status.json"
steps_path = BATCH / "steps.jsonl"
stop_path = BATCH / "STOP"
plan = [{"seed": seed, "model": model, "condition": condition, "base_config": base}
        for seed in SEEDS for model, base, condition in CONDITIONS]
status = {"batch": BATCH.name, "started_at": dt.datetime.now().astimezone().isoformat(),
          "state": "running", "total_jobs": len(plan), "completed_jobs": 0,
          "failed_jobs": [], "durations_min": {}, "plan": plan,
          "note": "seed 42 = original paper runs in experiments/; this batch adds seeds 43-46"}


def record(event, **fields):
    item = {"time": dt.datetime.now().astimezone().isoformat(), "event": event, **fields}
    with steps_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def save_status():
    tmp = status_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    tmp.replace(status_path)


python = ROOT / ".venv" / "Scripts" / "python.exe"
record("batch_started", seeds=SEEDS, jobs=len(plan),
       test_evaluation="training uses train/validation only; test logits dumped after training")
save_status()

try:
    for index, item in enumerate(plan, start=1):
        if stop_path.exists():
            record("stop_requested", before_job=index)
            status["state"] = "stopped_by_user"
            break
        model, base_name, condition, seed = item["model"], item["base_config"], item["condition"], item["seed"]
        exp_id = f"seed{seed}_{condition}"
        exp_dir = EXPERIMENTS / exp_id
        if (exp_dir / "DONE").exists():
            status["completed_jobs"] += 1
            record("job_skipped_done", index=index, exp_id=exp_id)
            save_status()
            continue
        if exp_dir.exists():  # partial run from an interrupted batch
            stamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
            shutil.move(str(exp_dir), str(EXPERIMENTS / f"_incomplete_{exp_id}_{stamp}"))
        cfg = yaml.safe_load((ROOT / base_name).read_text(encoding="utf-8"))
        cfg["seed"] = seed
        cfg["paths"]["experiments_dir"] = str(EXPERIMENTS)
        cfg["preprocessing"]["cache_dir"] = CACHE_DIR
        cfg["training"]["num_workers"] = NUM_WORKERS
        config_path = CONFIGS / f"{exp_id}.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        log_path = LOGS / f"{exp_id}.log"
        record("job_started", index=index, total=len(plan), seed=seed, model=model,
               condition=condition, config=str(config_path), log=str(log_path), exp_id=exp_id)
        status["current_job"] = exp_id
        save_status()
        t0 = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"batch={BATCH.name}\njob={index}/{len(plan)}\nseed={seed}\nmodel={model}\n"
                      f"condition={condition}\nconfig={config_path}\n\n")
            log.flush()
            proc = subprocess.run([str(python), "scripts/run_train.py", "--model", model,
                                   "--config", str(config_path), "--exp-id", exp_id],
                                  cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, check=False,
                                  env=ENV)
            ckpt = exp_dir / f"best_{model}.pt"
            dump_rc = None
            if ckpt.exists():
                log.write("\n\n==== dump_predictions ====\n")
                log.flush()
                dump = subprocess.run([str(python), "scripts/dump_predictions.py", "--ckpt", str(ckpt),
                                       "--backbone", model, "--out", str(exp_dir / "dump.npz")],
                                      cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, check=False,
                                      env=ENV)
                dump_rc = dump.returncode
                true_ckpt = exp_dir / f"best_true_{model}.pt"
                if true_ckpt.exists():
                    log.write("\n\n==== dump_predictions (true best epoch) ====\n")
                    log.flush()
                    dump2 = subprocess.run([str(python), "scripts/dump_predictions.py", "--ckpt", str(true_ckpt),
                                            "--backbone", model, "--out", str(exp_dir / "dump_besttrue.npz")],
                                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, check=False,
                                           env=ENV)
                    dump_rc = dump_rc or dump2.returncode
        minutes = round((time.time() - t0) / 60, 1)
        ok = proc.returncode == 0 and dump_rc == 0
        result = {"index": index, "total": len(plan), "seed": seed, "model": model,
                  "condition": condition, "train_exit_code": proc.returncode, "dump_exit_code": dump_rc,
                  "minutes": minutes, "log": str(log_path), "experiment": str(exp_dir)}
        record("job_finished" if ok else "job_failed", **result)
        status["durations_min"][exp_id] = minutes
        status["completed_jobs"] += 1
        if ok:
            (exp_dir / "DONE").write_text(dt.datetime.now().isoformat(), encoding="utf-8")
        else:
            status["failed_jobs"].append(result)
        save_status()
    if status["state"] == "running":
        status["state"] = "completed_with_failures" if status["failed_jobs"] else "completed"
except BaseException as exc:
    record("orchestrator_error", error=repr(exc), traceback=traceback.format_exc())
    status["state"] = "orchestrator_error"
    status["error"] = repr(exc)
    save_status()
    raise
finally:
    status["finished_at"] = dt.datetime.now().astimezone().isoformat()
    status.pop("current_job", None)
    record("batch_finished", state=status["state"], completed_jobs=status["completed_jobs"],
           failed_jobs=len(status["failed_jobs"]))
    save_status()
