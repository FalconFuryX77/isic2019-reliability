#!/usr/bin/env python
"""
Self-contained LaTeX packages for submission (no generated/ folder, no cross-document xr links):

  submission/paper_latex.zip     main.tex (all numbers and tables written in), references.bib, figures/
  submission/response_latex.zip  response_to_reviewers.tex (numbers and paper/supplement references written in)
  submission/supplementary_latex.zip  supplementary.tex, references.bib (cited entries only), figures/

Run after scripts/revision/build_pdfs.py, which produces the .aux files whose label numbers are used to
resolve the response letter's references to the paper and the supplementary material.
Each package is test-compiled with tectonic and its text compared with the regular build.
"""
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PR = ROOT / "paper_revision"
G = PR / "generated"
B = PR / "build"
OUT = ROOT / "submission"

MACROS = dict(re.findall(r"^\\newcommand\{\\(\w+)\}\{(.*)\}$", (G / "macros.tex").read_text(encoding="utf-8"), re.M))
CONTROL_WORD = re.compile(r"\\([A-Za-z]+)(\{\})?([ \t]*)")


def expand_macros(text):
    """Replace every generated result macro by its value, mimicking TeX's spacing rules."""
    def repl(m):
        name, braces, space = m.groups()
        if name not in MACROS:
            return m.group(0)
        # '\Macro{} word' keeps the space; '\Macro word' swallows it, as TeX does
        return MACROS[name] + (space if braces else "")
    for _ in range(3):
        new = CONTROL_WORD.sub(repl, text)
        if new == text:
            break
        text = new
    left = sorted({n for n in re.findall(r"\\([A-Za-z]+)", text) if n in MACROS})
    assert not left, f"unexpanded macros: {left}"
    return text


def inline_tables(text):
    """Write generated tables (and the generated 320 px section) into the document."""
    def repl(m):
        body = (G / f"{m.group(1)}.tex").read_text(encoding="utf-8").rstrip("\n")
        return body
    return re.sub(r"\\input\{generated/((?:supp_)?tab_\w+|supp_r320)\}", repl, text)


def pruned_bib(bib_text, tex_text):
    """Keep only the entries cited in tex_text, in their original order."""
    keys = set()
    for group in re.findall(r"\\cite\{([^}]*)\}", tex_text):
        keys.update(k.strip() for k in group.split(","))
    entries = re.split(r"\n(?=@)", bib_text)
    kept = [e for e in entries if (m := re.match(r"\s*@\w+\{([^,]+),", e)) and m.group(1).strip() in keys]
    found = {re.match(r"\s*@\w+\{([^,]+),", e).group(1).strip() for e in kept}
    assert found == keys, f"missing bib entries: {keys - found}"
    return "\n".join(e.strip() + "\n" for e in kept), len(kept)


def drop_generated_inputs(text):
    text = re.sub(r"^% Every experimental number is generated.*\n", "", text, flags=re.M)
    text = re.sub(r"^\\input\{generated/macros\}\n", "", text, flags=re.M)
    text = re.sub(r"^\\input\{generated/fallbacks\}\n", "", text, flags=re.M)
    assert "generated/" not in text, "a generated/ input is left"
    return text


def aux_labels(aux):
    """label -> printed number, from a hyperref .aux file."""
    labels = {}
    for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux.read_text(encoding="utf-8", errors="ignore")):
        labels.setdefault(m.group(1), m.group(2))
    return labels


def compile_and_text(folder, tex):
    r = subprocess.run(["tectonic", "--keep-logs", tex], cwd=folder, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"compilation of {tex} failed")
    pdf = folder / tex.replace(".tex", ".pdf")
    pages = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf)], capture_output=True,
                                                              text=True).stdout).group(1))
    txt = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, encoding="utf-8",
                         errors="ignore").stdout
    return pages, txt


def norm(t):
    return re.sub(r"\s+", " ", t).strip()


def zip_folder(folder, zpath):
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(folder).as_posix())


def build_paper(stage):
    text = (PR / "main.tex").read_text(encoding="utf-8")
    text = drop_generated_inputs(inline_tables(text))
    text = expand_macros(text)
    (stage / "figures").mkdir()
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{Images/([^}]+)\}", text)
    for f in figs:
        shutil.copy2(PR / "Images" / f, stage / "figures" / f)
    text = text.replace("{Images/", "{figures/")
    text = text.replace("\\bibliography{references_main}", "\\bibliography{references}")
    shutil.copy2(PR / "references_main.bib", stage / "references.bib")
    (stage / "main.tex").write_text(text, encoding="utf-8")
    return figs


def build_supplementary(stage):
    text = (PR / "supplementary.tex").read_text(encoding="utf-8")
    text = expand_macros(drop_generated_inputs(inline_tables(text)))
    (stage / "figures").mkdir()
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{Images/([^}]+)\}", text)
    for f in figs:
        shutil.copy2(PR / "Images" / f, stage / "figures" / f)
    text = text.replace("{Images/", "{figures/")
    bib, n = pruned_bib((PR / "references.bib").read_text(encoding="utf-8"), text)
    (stage / "references.bib").write_text(bib, encoding="utf-8")
    (stage / "supplementary.tex").write_text(text, encoding="utf-8")
    return figs, n


def build_response(stage):
    text = (PR / "response_to_reviewers.tex").read_text(encoding="utf-8")
    main_lab, supp_lab = aux_labels(B / "main.aux"), aux_labels(B / "supplementary.aux")
    own = set(re.findall(r"\\label\{([^}]*)\}", text))   # the letter's own labels stay as LaTeX references
    missing = []

    def ref(m):
        lab = m.group(1)
        if lab in own:
            return m.group(0)
        if lab.startswith("S-"):
            val = supp_lab.get(lab[2:])
        else:
            val = main_lab.get(lab)
        if val is None:
            missing.append(lab)
            return m.group(0)
        return val
    text = re.sub(r"\\ref\{([^}]*)\}", ref, text)
    assert not missing, f"unresolved references: {missing}"
    text = re.sub(r"^\\usepackage\{xr\}\n", "", text, flags=re.M)
    text = re.sub(r"^\\externaldocument(\[[^\]]*\])?\{[^}]*\}\n", "", text, flags=re.M)
    text = expand_macros(drop_generated_inputs(text))
    assert "\\cite" not in text, "the letter cites references; add a .bib"
    (stage / "response_to_reviewers.tex").write_text(text, encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="submission_"))
    # ---- paper
    sp = tmp / "paper_latex"
    sp.mkdir()
    figs = build_paper(sp)
    test = tmp / "test_paper"
    shutil.copytree(sp, test)
    pages, txt = compile_and_text(test, "main.tex")
    ref = subprocess.run(["pdftotext", str(B / "main.pdf"), "-"], capture_output=True, text=True,
                         encoding="utf-8", errors="ignore").stdout
    same = norm(txt) == norm(ref)
    print(f"paper: {pages} pages; {len(figs)} figures; text identical to regular build: {same}")
    zip_folder(sp, OUT / "paper_latex.zip")
    # ---- response letter
    sr = tmp / "response_latex"
    sr.mkdir()
    build_response(sr)
    test = tmp / "test_response"
    shutil.copytree(sr, test)
    pages_r, txt_r = compile_and_text(test, "response_to_reviewers.tex")
    ref_r = subprocess.run(["pdftotext", str(B / "response_to_reviewers.pdf"), "-"], capture_output=True, text=True,
                           encoding="utf-8", errors="ignore").stdout
    same_r = norm(txt_r) == norm(ref_r)
    print(f"response: {pages_r} pages; text identical to regular build: {same_r}")
    zip_folder(sr, OUT / "response_latex.zip")
    # ---- supplementary material
    ss = tmp / "supplementary_latex"
    ss.mkdir()
    sfigs, nbib = build_supplementary(ss)
    test = tmp / "test_supplementary"
    shutil.copytree(ss, test)
    pages_s, txt_s = compile_and_text(test, "supplementary.tex")
    ref_s = subprocess.run(["pdftotext", str(B / "supplementary.pdf"), "-"], capture_output=True, text=True,
                           encoding="utf-8", errors="ignore").stdout
    same_s = norm(txt_s) == norm(ref_s)
    print(f"supplementary: {pages_s} pages; {len(sfigs)} figures; {nbib} references; "
          f"text identical to regular build: {same_s}")
    zip_folder(ss, OUT / "supplementary_latex.zip")
    for z in ["paper_latex.zip", "response_latex.zip", "supplementary_latex.zip"]:
        with zipfile.ZipFile(OUT / z) as zz:
            print(z, [i.filename for i in zz.infolist()])
    shutil.rmtree(tmp, ignore_errors=True)
    if not (same and same_r and same_s):
        sys.exit(1)


if __name__ == "__main__":
    main()
