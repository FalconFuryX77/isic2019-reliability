#!/usr/bin/env python
"""
Tracked-changes PDF: the submitted paper (original LaTeX body, set in the same IEEE preamble) against the revised
paper, marked up with latexdiff (deleted text struck out in red, added text underlined in blue).
For reviewing the revision only; it is not part of the submission.

    python scripts/revision/make_tracked_changes.py <submitted main.tex>

Output: submission/tracked_changes.pdf
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.revision.build_pdfs import MIKTEX, tex_env  # noqa: E402
from scripts.revision.make_submission_zips import expand_macros, inline_tables, drop_generated_inputs  # noqa: E402

PR = ROOT / "paper_revision"
OUT = ROOT / "submission"
REMOVED = ["russo2016", "salinas2024", "gayatri2024", "ieracitano2025", "isic2019"]


def bib_entries(text, keys):
    entries = re.split(r"\n(?=@)", text)
    return [e.strip() for e in entries if (m := re.match(r"\s*@\w+\{([^,]+),", e)) and m.group(1).strip() in keys]


def main(original):
    new = (PR / "main.tex").read_text(encoding="utf-8")
    new = expand_macros(drop_generated_inputs(inline_tables(new)))
    head, body_new = new.split("\\begin{abstract}", 1)
    head = head.replace("\\usepackage{url}", "\\usepackage{url}\n\\usepackage{booktabs}  % used by the submitted tables")
    new = head + "\\begin{abstract}" + body_new
    new = new.replace("\\bibliography{references_main}", "\\bibliography{references_diff}")

    orig = Path(original).read_text(encoding="utf-8")
    body_old = orig[orig.index("\\begin{abstract}"):orig.index("\\bibliographystyle")]
    old = head + body_old + "\\bibliographystyle{IEEEtran}\n\\bibliography{references_diff}\n\n\\end{document}\n"

    tmp = Path(tempfile.mkdtemp(prefix="trackchanges_"))
    (tmp / "old.tex").write_text(old, encoding="utf-8")
    (tmp / "new.tex").write_text(new, encoding="utf-8")
    bib = (PR / "references_main.bib").read_text(encoding="utf-8")
    have = set(re.findall(r"@\w+\{([^,]+),", bib))
    extra = bib_entries((PR / "references.bib").read_text(encoding="utf-8"), set(REMOVED) - have)
    (tmp / "references_diff.bib").write_text(bib + "\n\n" + "\n\n".join(extra) + "\n", encoding="utf-8")
    for f in ["IEEEtran.cls", "IEEEtran.bst"]:
        shutil.copy2(PR / f, tmp / f)
    shutil.copytree(PR / "Images", tmp / "Images")

    env = tex_env()
    # latexdiff-so is the self-contained variant (bundles Algorithm::Diff)
    # tables are compared as whole blocks (old version struck out, new version shown), not cell by cell
    r = subprocess.run([str(MIKTEX / "latexdiff-so.exe"), "--type=UNDERLINE", "--floattype=FLOATSAFE",
                        "--config", "PICTUREENV=(?:picture|DIFnomarkup|tabular)[\\w\\d*@]*",
                        "old.tex", "new.tex"], cwd=tmp, capture_output=True, text=True, env=env, encoding="utf-8")
    if r.returncode != 0 or "\\begin{document}" not in r.stdout:
        print(r.stderr[-3000:])
        raise SystemExit("latexdiff failed")
    (tmp / "diff.tex").write_text(r.stdout, encoding="utf-8")
    cmd = [str(MIKTEX / "pdflatex.exe"), "-interaction=nonstopmode", "diff.tex"]
    subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, env=env)
    subprocess.run([str(MIKTEX / "bibtex.exe"), "diff"], cwd=tmp, capture_output=True, text=True, env=env)
    subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, env=env)
    subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, env=env)
    log = (tmp / "diff.log").read_text(encoding="utf-8", errors="ignore")
    errs = [l for l in log.splitlines() if l.startswith("! ")]
    if not (tmp / "diff.pdf").exists():
        print("\n".join(errs[:10]))
        raise SystemExit("compiling the tracked-changes document failed")
    OUT.mkdir(exist_ok=True)
    shutil.copy2(tmp / "diff.pdf", OUT / "tracked_changes.pdf")
    pages = re.search(r"Output written on diff\.pdf \((\d+) pages", log)
    print(f"tracked_changes.pdf: {pages.group(1) if pages else '?'} pages; LaTeX errors: {len(errs)}")
    for e in errs[:5]:
        print("  ", e)
    print("work folder:", tmp)


if __name__ == "__main__":
    main(sys.argv[1])
