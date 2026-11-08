#!/usr/bin/env python
"""Regenerate assets and compile main.pdf (pdfLaTeX + BibTeX, as on Overleaf; IEEEtran conference class),
supplementary.pdf and response_to_reviewers.pdf (Tectonic).

Checks the venue constraint: main paper <= 8 pages (the distinct citations are counted for information).
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


MIKTEX = Path.home() / "scoop" / "apps" / "miktex" / "current" / "texmfs" / "install" / "miktex" / "bin" / "x64"


def tex_env():
    """MiKTeX aborts on PATH entries that are files rather than folders; drop those for its processes."""
    import os
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(e for e in env.get("PATH", "").split(os.pathsep) if e and not os.path.isfile(e))
    return env


def pdflatex_main():
    """Compile main.tex exactly as Overleaf does: pdfLaTeX, BibTeX (IEEEtran.bst), pdfLaTeX twice."""
    exe = MIKTEX / "pdflatex.exe"
    if not exe.exists():
        raise SystemExit("pdfLaTeX (MiKTeX) not found; install it with `scoop install miktex`")
    env = tex_env()
    for f in ["references_main.bib", "IEEEtran.bst"]:
        shutil.copy2(P / f, B / f)
    run = lambda: subprocess.run([str(exe), "-interaction=nonstopmode", f"-output-directory={B}", "main.tex"],
                                 cwd=P, capture_output=True, text=True, env=env)
    run()
    subprocess.run([str(MIKTEX / "bibtex.exe"), "main"], cwd=B, capture_output=True, text=True, env=env)
    run()
    r = run()
    log = (B / "main.log").read_text(encoding="utf-8", errors="ignore")
    errs = [l for l in log.splitlines() if l.startswith("! ")]
    over = sorted({l for l in log.splitlines() if l.startswith("Overfull")})
    return (0 if (B / "main.pdf").exists() and not errs else 1), errs, over


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
        rc, errs, over = pdflatex_main() if name == "main.tex" else tex(name)
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
    # INCOWOCO limits pages (8), not references
    print(f"main.pdf pages = {n_pages} (limit 8); citations = {len(cites)}")
    ok &= n_pages <= 8
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
