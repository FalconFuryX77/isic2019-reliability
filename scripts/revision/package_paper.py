#!/usr/bin/env python
"""Regenerate all LaTeX assets, lint, and zip the revised paper for Overleaf.

Output: paper_revision_overleaf.zip (project root) containing main.tex,
response_to_reviewers.tex, references.bib, generated/, the figures actually
referenced, and a latexmkrc so that Overleaf builds main.aux before the response
letter (needed by the xr package for table/figure numbers in the letter).
"""
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "paper_revision"
PY = sys.executable

LATEXMKRC = r"""# Overleaf: compile main.tex automatically when the response letter needs main.aux (xr package)
add_cus_dep( 'tex', 'aux', 0, 'makeexternaldocument' );
sub makeexternaldocument {
    if (!($root_filename eq $_[0]))
    {
        system( "latexmk -cd -pdf \"$_[0]\"" );
    }
}
"""


def main():
    subprocess.run([PY, str(ROOT / "scripts/revision/make_tex_assets.py")], check=True)
    lint = subprocess.run([PY, str(ROOT / "scripts/revision/lint_tex.py")], capture_output=True, text=True)
    print(lint.stdout)
    lint2 = subprocess.run([PY, str(ROOT / "scripts/revision/lint_tex.py"), str(P), "response_to_reviewers.tex"],
                           capture_output=True, text=True)
    print(lint2.stdout)
    (P / "latexmkrc").write_text(LATEXMKRC, encoding="utf-8")
    used = set()
    for f in ["main.tex", "response_to_reviewers.tex"]:
        used |= set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", (P / f).read_text(encoding="utf-8")))
    out = ROOT / "paper_revision_overleaf.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ["main.tex", "response_to_reviewers.tex", "references.bib", "latexmkrc"]:
            z.write(P / f, f)
        for f in sorted((P / "generated").glob("*.tex")):
            z.write(f, f"generated/{f.name}")
        for g in sorted(used):
            z.write(P / g, g)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(used)} figures)")
    return 0 if "No blocking problems" in lint.stdout else 1


if __name__ == "__main__":
    sys.exit(main())
