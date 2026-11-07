#!/usr/bin/env python
"""
Generate the master experiment summary (experiment_summary.csv + .md).

Preserves every existing per-experiment row and adds explicit staged
component-ablation rows for the ResNet-50 reliability study:

  E0_resnet50_plain_ce            = resnet50_e0 (plain CE)
  E1_resnet50_class_weighted      = resnet50_baseline (class-weighted loss)
  E2_resnet50_temperature_scaled  = E1 model + temperature scaling (val-fit)
  E3_resnet50_uncertainty         = E1 model + calibration + MC-dropout uncertainty
                                    + selective prediction (th locked on val)

Config columns record the exact setting used for each row.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._common import load_config  # noqa: E402


def read_json(p):
    if Path(p).exists():
        return json.loads(Path(p).read_text())
    return {}


def base_row(experiment, backbone, stage, config_descr):
    return {
        "experiment": experiment,
        "stage": stage,
        "backbone": backbone,
        "config": config_descr,
    }


def add_metrics(row, exp_dir):
    m = read_json(exp_dir / "metrics.json").get("macro", {})
    for k, v in m.items():
        row[f"test_{k}"] = v


def add_calib(row, exp_dir):
    c = read_json(exp_dir / "calibration_metrics.json")
    if not c:
        return
    row["temperature"] = c.get("temperature")
    for k, v in c.get("before_calibration", {}).items():
        row[f"calib_before_{k}"] = v
    for k, v in c.get("after_calibration", {}).items():
        row[f"calib_after_{k}"] = v


def add_uncertainty(row, exp_dir):
    u = read_json(exp_dir / "uncertainty_metrics.json")
    if not u:
        return
    ent = u.get("predictive_entropy", {})
    for k in ("mean_correct", "mean_incorrect", "cohens_d",
              "mannwhitney_p", "error_detection_auroc"):
        if k in ent:
            row[f"unc_entropy_{k}"] = ent[k]


def add_selective(row, exp_dir):
    s = read_json(exp_dir / "selective_summary.json").get("summary", {}).get("cov85", {})
    if not s:
        return
    row["sel_threshold_locked"] = s.get("locked_threshold")
    row["sel_val_coverage"] = s.get("val_coverage")
    row["sel_test_coverage"] = s.get("test_coverage")
    row["sel_test_accuracy"] = s.get("test_accuracy")
    row["sel_test_risk"] = s.get("test_risk")
    row["sel_test_n_accepted"] = s.get("test_n_accepted")
    row["sel_test_n_rejected"] = s.get("test_n_rejected")


def add_config_details(row, exp_dir):
    c = read_json(exp_dir / "experiment_config.json")
    if not c:
        return
    tr = c.get("training", {})
    au = c.get("augmentation", {})
    hd = c.get("head", {})
    row["use_class_weights"] = tr.get("use_class_weights")
    row["phase1_epochs"] = tr.get("phase1_epochs")
    row["phase2_epochs"] = tr.get("phase2_epochs")
    row["batch_size"] = tr.get("batch_size")
    row["optimizer"] = tr.get("optimizer")
    row["image_size"] = c.get("preprocessing", {}).get("image_size")
    row["augmentation"] = "on" if au.get("enabled") else "off"
    row["dense_size"] = hd.get("dense_size")
    row["dropout"] = hd.get("dropout")
    row["seed"] = c.get("seed")


def main():
    cfg = load_config()
    exp_root = Path(cfg.paths.experiments_dir)
    results = Path(cfg.paths.results_dir)
    results.mkdir(parents=True, exist_ok=True)

    rows = []

    # ── Staged ResNet-50 component study (E0–E3) ──
    e0_dir = exp_root / "resnet50_e0"
    e1_dir = exp_root / "resnet50_baseline"

    e0 = base_row("E0_resnet50_plain_ce", "resnet50", "E0",
                  "resnet50_e0: plain CE, no class weights, no calibration, no uncertainty")
    add_metrics(e0, e0_dir)
    add_config_details(e0, e0_dir)
    rows.append(e0)

    e1 = base_row("E1_resnet50_class_weighted", "resnet50", "E1",
                  "resnet50_baseline: class-weighted CE (train-split weights), no calibration")
    add_metrics(e1, e1_dir)
    add_config_details(e1, e1_dir)
    rows.append(e1)

    e2 = base_row("E2_resnet50_temperature_scaled", "resnet50", "E2",
                  "E1 model, temperature scaling (T fit on validation only, applied to test)")
    add_metrics(e2, e1_dir)
    add_config_details(e2, e1_dir)
    add_calib(e2, e1_dir)
    rows.append(e2)

    e3 = base_row("E3_resnet50_uncertainty", "resnet50", "E3",
                  "E1 model + calibrated probs + MC-dropout uncertainty + selective "
                  "(threshold locked on validation, applied to test)")
    add_metrics(e3, e1_dir)
    add_config_details(e3, e1_dir)
    add_calib(e3, e1_dir)
    add_uncertainty(e3, e1_dir)
    add_selective(e3, e1_dir)
    rows.append(e3)

    # ── Existing per-experiment rows (preserved, incl. EfficientNet-B3) ──
    for exp_dir in sorted(exp_root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name in ("resnet50_e0", "resnet50_baseline"):
            continue
        row = base_row(exp_dir.name, exp_dir.name.replace("_baseline", ""), "-",
                       "full pipeline baseline (class-weighted, calibrated, uncertainty, selective)")
        add_config_details(row, exp_dir)
        add_metrics(row, exp_dir)
        add_calib(row, exp_dir)
        add_uncertainty(row, exp_dir)
        add_selective(row, exp_dir)
        if row.get("test_accuracy") is not None:
            rows.append(row)

    # ── Write CSV ──
    out_csv = results / "experiment_summary.csv"
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows to {out_csv}")

    # ── Write markdown ──
    md = ["# Master Experiment Summary (final)",
          "",
          "All metrics on the **internal test split** (2,523 images), evaluated only after",
          "all choices were locked. Temperature and selective thresholds fitted on the",
          "**validation split only**.",
          "",
          "| ID | Backbone | Class wgt | Aug | Acc | Bal. acc | F1 macro | AUC-ROC | ECE→ | NLL→ | Sel acc @85% |",
          "|---|---|---|---|---:|---:|---:|---:|---|---|---:|"]
    for r in rows:
        def g(k):
            return r.get(k)
        ece = f"{g('calib_before_ECE'):.4f}→{g('calib_after_ECE'):.4f}" if g('calib_before_ECE') is not None else "-"
        nll = f"{g('calib_before_NLL'):.4f}→{g('calib_after_NLL'):.4f}" if g('calib_before_NLL') is not None else "-"
        sel = f"{g('sel_test_accuracy'):.3f} @ {g('sel_test_coverage')*100:.0f}%" if g('sel_test_accuracy') is not None else "-"
        cw = g('use_class_weights')
        cw_s = {True: "yes", False: "no", None: "-"}.get(cw, "-")
        md.append(f"| {r['experiment']} | {r['backbone']} | {cw_s} | "
                  f"{g('augmentation') or '-'} | {g('test_accuracy'):.4f} | "
                  f"{g('test_balanced_accuracy'):.4f} | {g('test_f1_macro'):.4f} | "
                  f"{g('test_auc_roc_macro'):.4f} | {ece} | {nll} | {sel} |")
    out_md = results / "experiment_summary.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"  Wrote {out_md}")

    # ── E3 uncertainty/selective detail block ──
    e3_details = ["",
                  "### E3 uncertainty & selective detail (ResNet-50, entropy on calibrated probs)",
                  "",
                  "| Quantity | Value |",
                  "|---|---:|"]
    ent = read_json(e1_dir / "uncertainty_metrics.json").get("predictive_entropy", {})
    s = read_json(e1_dir / "selective_summary.json").get("summary", {}).get("cov85", {})
    for label, k in [("Entropy mean (correct)", "mean_correct"),
                     ("Entropy mean (incorrect)", "mean_incorrect"),
                     ("Cohen's d", "cohens_d"),
                     ("Mann-Whitney p", "mannwhitney_p"),
                     ("Error-detection AUROC", "error_detection_auroc")]:
        if k in ent:
            e3_details.append(f"| {label} | {ent[k]:.4f} |")
    for label, k in [("Locked threshold", "locked_threshold"),
                     ("Val coverage", "val_coverage"),
                     ("Test coverage", "test_coverage"),
                     ("Test accuracy", "test_accuracy"),
                     ("Test n accepted", "test_n_accepted"),
                     ("Test n rejected", "test_n_rejected")]:
        if k in s:
            v = s[k]
            e3_details.append(f"| {label} | {int(v) if isinstance(v, (int, float)) and k.startswith('test_n') else round(v, 4)} |")
    with open(out_md, "a", encoding="utf-8") as f:
        f.write("\n".join(e3_details) + "\n")


if __name__ == "__main__":
    main()