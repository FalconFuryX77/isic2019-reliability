#!/usr/bin/env python
"""Component ablation summary: E0 (plain CE) vs E1 (class-weighted) on ResNet-50.

E2 = E1 + calibration, E3 = E1/E2 + uncertainty/selective (reused artifacts).
All metrics on the internal test split (never used for tuning).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJ = Path(__file__).resolve().parent.parent


def load_json(rel):
    p = PROJ / rel
    return json.loads(p.read_text()) if p.exists() else None


def get_metrics(exp):
    m = load_json(f"experiments/{exp}/metrics.json")
    if not m:
        return {}
    macro = m.get("macro", m)
    return {
        "accuracy": macro.get("accuracy", m.get("test_accuracy")),
        "balanced_accuracy": macro.get("balanced_accuracy", m.get("test_balanced_accuracy")),
        "f1_macro": macro.get("f1_macro", m.get("test_f1_macro")),
        "auc_roc_macro": macro.get("auc_roc_macro", m.get("test_auc_roc_macro")),
        "auc_pr_macro": macro.get("auc_pr_macro", m.get("test_auc_pr_macro")),
    }


def get_calib(exp):
    c = load_json(f"experiments/{exp}/calibration_metrics.json")
    if not c:
        return {}
    return {
        "T": c.get("temperature"),
        "ece_before": c["before_calibration"].get("ECE"),
        "ece_after": c["after_calibration"].get("ECE"),
        "nll_before": c["before_calibration"].get("NLL"),
        "nll_after": c["after_calibration"].get("NLL"),
    }


def get_selective(exp):
    s = load_json(f"experiments/{exp}/selective_summary.json")
    if not s:
        return {}
    cov85 = s.get("summary", {}).get("cov85", {})
    return {
        "sel_cov85_th": cov85.get("locked_threshold"),
        "sel_cov85_acc": cov85.get("test_accuracy"),
        "sel_cov85_coverage": cov85.get("test_coverage"),
    }


# ── E0 and E1 core classification ──
e0 = get_metrics("resnet50_e0")
e1 = get_metrics("resnet50_baseline")
c1 = get_calib("resnet50_baseline")
s1 = get_selective("resnet50_baseline")

# E1, E2, E3 all use the SAME checkpoint (class-weighted baseline). The staged
# rows show what each component ADDS on top: E2 = calibration (does not change
# argmax, only probabilities), E3 = uncertainty + selective (analysis on the
# calibrated probabilities). Classification metrics therefore only differ
# between E0 and E1.
rows = []

def e1_class():
    return {f"test_{k}": v for k, v in e1.items()}

rows.append({"component": "E0 baseline (plain CE)", "class_weighting": "no",
             "calibration": "no", "uncertainty/selective": "no",
             **{f"test_{k}": v for k, v in e0.items()}})
rows.append({"component": "E1 + class weighting", "class_weighting": "yes",
             "calibration": "no", "uncertainty/selective": "no", **e1_class()})
rows.append({"component": "E2 + temperature calibration", "class_weighting": "yes",
             "calibration": "yes", "uncertainty/selective": "no", **e1_class(),
             **{f"calib_{k}": v for k, v in c1.items()}})
rows.append({"component": "E3 + uncertainty / selective", "class_weighting": "yes",
             "calibration": "yes", "uncertainty/selective": "yes", **e1_class(),
             **{f"calib_{k}": v for k, v in c1.items()},
             **{f"sel_{k}": v for k, v in s1.items()}})

df = pd.DataFrame(rows)
out = PROJ / "results" / "tables"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "component_ablation.csv", index=False)

# Markdown table
md = ["# Component Ablation — ResNet-50 (ISIC 2019, internal test split)",
      "",
      "E0: plain CE, no reliability components. E1: + class-weighted loss.",
      "E2: + temperature calibration (T fit on validation, applied to test).",
      "E3: + MC-dropout uncertainty and selective prediction (thresholds locked on validation).",
      "E1/E2/E3 share the same trained checkpoint (the completed baseline); each row",
      "shows what the named component ADDS. Calibration does not change argmax, so",
      "classification metrics are identical between E1, E2, E3.",
      "",
      "| Component | Class wgt | Calib | Unc/Sel | Acc | Bal. acc | F1 macro | AUC-ROC |",
      "|---|---|---|---|---:|---:|---:|---:|"]
def fmt(v):
    return "-" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:.4f}"


for _, r in df.iterrows():
    md.append(f"| {r['component']} | {r['class_weighting']} | {r['calibration']} | "
              f"{r['uncertainty/selective']} | {fmt(r.get('test_accuracy'))} | "
              f"{fmt(r.get('test_balanced_accuracy'))} | {fmt(r.get('test_f1_macro'))} | "
              f"{fmt(r.get('test_auc_roc_macro'))} |")

if "calib_T" in df.columns:
    md += ["", "**Calibration (E2, temperature scaling, T fit on validation, evaluated on test):**", "",
           "| Component | T | ECE before | ECE after | NLL before | NLL after |",
           "|---|---|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        if "calib_T" in r and pd.notna(r.get("calib_T")):
            md.append(f"| {r['component']} | {r['calib_T']:.4f} | {r['calib_ece_before']:.4f} | "
                      f"{r['calib_ece_after']:.4f} | {r['calib_nll_before']:.4f} | {r['calib_nll_after']:.4f} |")

if "sel_cov85_acc" in df.columns:
    md += ["", "**Selective prediction (E3, thresholds locked on validation, applied to test):**", "",
           "| Component | th (locked) | cov (test) | acc (test) |",
           "|---|---|---:|---:|"]
    for _, r in df.iterrows():
        if "sel_cov85_acc" in r and pd.notna(r.get("sel_cov85_acc")):
            md.append(f"| {r['component']} | {r['sel_cov85_th']:.3f} | {r['sel_cov85_coverage']:.3f} | "
                      f"{r['sel_cov85_acc']:.3f} |")

(out / "component_ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(df.to_string(index=False))
print(f"\nSaved {out / 'component_ablation.csv'} and .md")