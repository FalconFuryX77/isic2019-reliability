#!/usr/bin/env python
"""Write a plain-text rendering of main.tex with all result macros substituted (for proofreading)."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "paper_revision"
BS = chr(92)
mac = {}
for line in (P / "generated" / "macros.tex").read_text(encoding="utf-8").splitlines():
    m = re.match(re.escape(BS) + r"newcommand\{" + re.escape(BS) + r"([A-Za-z]+)\}\{(.*)\}$", line)
    if m:
        mac[m.group(1)] = m.group(2)
src = (P / (sys.argv[1] if len(sys.argv) > 1 else "main.tex")).read_text(encoding="utf-8")
out = re.sub(re.escape(BS) + r"([A-Z][A-Za-z]+)(\{\})?", lambda m: mac.get(m.group(1), m.group(0)), src)
(P / "generated" / "rendered_preview.txt").write_text(out, encoding="utf-8")
print("written", P / "generated" / "rendered_preview.txt")
