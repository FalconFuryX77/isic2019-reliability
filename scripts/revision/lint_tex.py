#!/usr/bin/env python
"""Static checks for the revised LaTeX (no TeX installation needed).

Checks: environment balance, brace balance, \\ref -> \\label, \\cite -> bib keys,
\\includegraphics / \\input targets exist, non-ASCII characters, leftover
placeholders, and that the fallbacks file is empty (all results available).
Usage: python scripts/revision/lint_tex.py [paper_dir] [main.tex]
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper_revision"
MAIN = sys.argv[2] if len(sys.argv) > 2 else "main.tex"


def strip_comments(t):
    return re.sub(r"(?<!\\)%.*", "", t)


def expand(path, seen=None):
    seen = seen or set()
    txt = strip_comments(path.read_text(encoding="utf-8"))
    out = [txt]
    for inc in re.findall(r"\\input\{([^}]+)\}", txt):
        p = PAPER / (inc if inc.endswith(".tex") else inc + ".tex")
        if p.exists() and p not in seen:
            seen.add(p)
            out.append(expand(p, seen))
    return "\n".join(out)


def main():
    problems, warnings = [], []
    main_path = PAPER / MAIN
    raw = main_path.read_text(encoding="utf-8")
    body = strip_comments(raw)
    for inc in re.findall(r"\\input\{([^}]+)\}", body):
        p = PAPER / (inc if inc.endswith(".tex") else inc + ".tex")
        if not p.exists():
            problems.append(f"missing \\input file: {inc}")
    full = expand(main_path)
    # environments
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", full):
        kind, env = m.groups()
        if kind == "begin":
            stack.append(env)
        else:
            if not stack or stack[-1] != env:
                problems.append(f"environment mismatch at \\end{{{env}}} (open: {stack[-3:]})")
                if env in stack:
                    while stack and stack[-1] != env:
                        stack.pop()
                    stack.pop()
            else:
                stack.pop()
    if stack:
        problems.append(f"unclosed environments: {stack}")
    # braces (per file, ignoring escaped braces)
    for f in [main_path] + sorted((PAPER / "generated").glob("*.tex")):
        t = strip_comments(f.read_text(encoding="utf-8")).replace("\\{", "").replace("\\}", "")
        if t.count("{") != t.count("}"):
            problems.append(f"brace imbalance in {f.name}: {t.count('{')} vs {t.count('}')}")
        if t.count("$") % 2:
            warnings.append(f"odd number of $ in {f.name}")
    # refs / labels
    labels = set(re.findall(r"\\label\{([^}]+)\}", full))
    for ext in re.findall(r"\\externaldocument\{([^}]+)\}", full):
        labels |= set(re.findall(r"\\label\{([^}]+)\}", expand(PAPER / f"{ext}.tex")))
    for r in set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", full)):
        if r not in labels:
            problems.append(f"undefined reference: {r}")
    for l in labels:
        if l.startswith(("tab:", "fig:")) and f"{{{l}}}" not in re.sub(r"\\label\{[^}]+\}", "", full):
            warnings.append(f"label never referenced: {l}")
    # citations
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = set()
    for grp in re.findall(r"\\cite\{([^}]+)\}", full):
        cited |= {k.strip() for k in grp.split(",")}
    for k in sorted(cited - keys):
        problems.append(f"citation key not in references.bib: {k}")
    unused = sorted(keys - cited)
    if unused:
        warnings.append(f"bib entries not cited (harmless): {unused}")
    # graphics
    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", full):
        if not (PAPER / g).exists():
            problems.append(f"missing figure: {g}")
    # non-ascii
    for f in [main_path] + sorted((PAPER / "generated").glob("*.tex")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            bad = {c for c in line if ord(c) > 127}
            if bad:
                problems.append(f"non-ASCII {sorted(bad)} in {f.name}:{i}")
    # pending placeholders
    fb = PAPER / "generated" / "fallbacks.tex"
    if fb.exists():
        pend = re.findall(r"providecommand\{\\([A-Za-z]+)\}", fb.read_text(encoding="utf-8"))
        if pend:
            warnings.append(f"{len(pend)} result macros still pending: {pend}")
    for f in sorted((PAPER / "generated").glob("tab_*.tex")):
        if "pending" in f.read_text(encoding="utf-8")[:40]:
            warnings.append(f"table still pending: {f.name}")
    # common LaTeX pitfalls in text
    txt = strip_comments(raw)
    txt = re.sub(r"\\begin\{(equation|align)\*?\}.*?\\end\{\1\*?\}", "", txt, flags=re.S)
    clean = re.sub(r"\$[^$]*\$|\\(?:texttt|label|ref|cite|input|includegraphics|externaldocument)(?:\[[^\]]*\])?\{[^}]*\}",
                   "", txt)
    for m in re.finditer(r"(?<!\\)_", clean):
        ctx = clean[max(0, m.start() - 25): m.end() + 25].replace("\n", " ")
        problems.append(f"unescaped underscore in text: ...{ctx}...")
    print("PROBLEMS:" if problems else "No blocking problems found.")
    for p in problems:
        print("  -", p)
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print("  -", w)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
