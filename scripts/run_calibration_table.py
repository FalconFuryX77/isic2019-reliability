#!/usr/bin/env python
"""Build the calibration comparison table from existing calibration_metrics.json files."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJ = Path(__file__).resolve().parent.parent
EXPS = ["resnet50_baseline", "efficientnet_b3_baseline"]
DISPLAY = {"resnet50_baseline": "ResNet-50", "efficientnet_b3_baseline": "EfficientNet-B3"}

rows = []
for exp in EXPS:
    p = PROJ / "experiments" / exp / "calibration_metrics.json"
    if not p.exists():
        print(f"[SKIP] missing {p}")
        continue
    m = json.loads(p.read_text())
    before = m.get("before_calibration", {})
    after = m.get("after_calibration", {})
    rows.append({
        "Model": DISPLAY[exp],
        "Temperature T": round(m.get("temperature", float("nan")), 4),
        "ECE (raw)": round(before.get("ECE", float("nan")), 4),
        "ECE (scaled)": round(after.get("ECE", float("nan")), 4),
        "MCE (raw)": round(before.get("MCE", float("nan")), 4),
        "MCE (scaled)": round(after.get("MCE", float("nan")), 4),
        "Brier (raw)": round(before.get("Brier", float("nan")), 4),
        "Brier (scaled)": round(after.get("Brier", float("nan")), 4),
        "NLL (raw)": round(before.get("NLL", float("nan")), 4),
        "NLL (scaled)": round(after.get("NLL", float("nan")), 4),
    })

if not rows:
    print("No calibration files found.")
    sys.exit(1)

df = pd.DataFrame(rows)

# Add relative improvement columns
for col in ["ECE", "MCE", "Brier", "NLL"]:
    df[f"\u0394{col} (%)"] = ((df[f"{col} (scaled)"] - df[f"{col} (raw)"]) / df[f"{col} (raw)"] * 100).round(1)

out = PROJ / "results" / "tables"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "calibration_comparison.csv", index=False)

md = ["# Calibration Comparison (temperature scaling; T fit on validation, evaluated on test)\n",
      "| Model | T | ECE raw | ECE scaled | \u0394ECE % | MCE raw | MCE scaled | Brier raw | Brier scaled | NLL raw | NLL scaled |",
      "|-------|--:|--------:|-----------:|--------:|--------:|-----------:|----------:|-------------:|--------:|-----------:|"]
for _, r in df.iterrows():
    md.append(f"| {r['Model']} | {r['Temperature T']:.4f} | {r['ECE (raw)']:.4f} | {r['ECE (scaled)']:.4f} | "
              f"{r['\u0394ECE (%)']:.1f} | {r['MCE (raw)']:.4f} | {r['MCE (scaled)']:.4f} | "
              f"{r['Brier (raw)']:.4f} | {r['Brier (scaled)']:.4f} | {r['NLL (raw)']:.4f} | {r['NLL (scaled)']:.4f} |")
(out / "calibration_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")

print(df.to_string(index=False))
print(f"\nSaved {out / 'calibration_comparison.csv'} and .md")