#!/usr/bin/env python
"""Regenerate assets and compile main.pdf, supplementary.pdf and response_to_reviewers.pdf with Tectonic.

Checks the venue constraints: main paper <= 8 pages and <= 14 distinct citations.
The response letter uses xr (\\externaldocument{main}, [S-]{supplementary}); the two .aux files are
staged next to the sources while it compiles. PDFs end up in paper_revision/build/.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "paper_revision"
B = P / "build"
TECTONIC = shutil.which("tectonic") or str(Path.home() / "scoop" / "shims" / "tectonic.exe")


def tex(name):
    r = subprocess.run([TECTONIC, "-X", "compile", "--keep-intermediates", "--keep-logs", "--outdir", str(B), name],
                       cwd=P, capture_output=True, text=True)
    errs = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("error")]
    over = sorted({l for l in (r.stdout + r.stderr).splitlines() if "Overfull" in l})
    return r.returncode, errs, over


def pages(pdf):
    r = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
    m = re.search(r"Pages:\s+(\d+)", r.stdout)
    return int(m.group(1)) if m else -1


def main():
    B.mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/revision/make_tex_assets.py")], check=True,
                   capture_output=True)
    ok = True
    for name in ["main.tex", "supplementary.tex"]:
        rc, errs, over = tex(name)
        print(f"{name}: exit {rc}, {pages(B / name.replace('.tex', '.pdf'))} pages; errors {errs[:3]}; overfull {len(over)}")
        for o in over[:5]:
            print("   ", o)
        ok &= rc == 0
    staged = []
    for a in ["main.aux", "supplementary.aux"]:
        shutil.copy2(B / a, P / a)
        staged.append(P / a)
    try:
        rc, errs, over = tex("response_to_reviewers.tex")
        print(f"response_to_reviewers.tex: exit {rc}, {pages(B / 'response_to_reviewers.pdf')} pages; errors {errs[:3]}")
        ok &= rc == 0
    finally:
        for f in staged:
            f.unlink(missing_ok=True)
    n_pages = pages(B / "main.pdf")
    cites = set()
    for g in re.findall(r"\\cite\{([^}]+)\}", (P / "main.tex").read_text(encoding="utf-8")):
        cites |= {k.strip() for k in g.split(",")}
    print(f"main.pdf pages = {n_pages} (limit 8); citations = {len(cites)} (limit 14)")
    ok &= n_pages <= 8 and len(cites) <= 14
    # unresolved references in logs
    for log in ["main.log", "supplementary.log", "response_to_reviewers.log"]:
        txt = (B / log).read_text(encoding="utf-8", errors="ignore") if (B / log).exists() else ""
        und = sorted(set(re.findall(r"Reference `([^']+)' on page", txt)))
        cit = sorted(set(re.findall(r"Citation `([^']+)'", txt)))
        if und or cit:
            print(f"  {log}: undefined refs {und[:10]} citations {cit[:10]}")
            ok = False
    print("BUILD OK" if ok else "BUILD HAS PROBLEMS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
