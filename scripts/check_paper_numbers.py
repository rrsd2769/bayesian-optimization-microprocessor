"""
Flag every number in the paper's .tex files (and the generated tables) that cannot be found in
results/v2/paper_numbers.json, up to rounding.

    python scripts/check_paper_numbers.py            # exit status 1 when anything is unmatched

A number matches a stored value v when v, v x 100, v / 100, v x 1000 or v / 1000 rounds to it at the number's own
number of decimals (scientific notation: at its own significant digits). Integers up to 10 are not checked. Version
strings (a.b.c) are matched against the stored strings.
"""
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
FILES = ["main.tex", "intro.tex", "methodology.tex", "results.tex", "conclusion.tex"] + \
        sorted(str(p.relative_to(PAPER)) for p in (PAPER / "tables").glob("*.tex"))
SCALES = (1, 100, 0.01, 1000, 0.001)

STRIP = [
    r"(?<!\\)%.*$",                                   # comments
    r"\\(?:cite|ref|label|input|includegraphics|bibliography|bibliographystyle|documentclass|usepackage|graphicspath|"
    r"definecolor|renewcommand|setlength|begin|end)(?:\[[^\]]*\])?\{[^}]*\}(?:\{[^}]*\})?",
    r"\\(?:footnotesize|small|resizebox\{[^}]*\}\{[^}]*\})",
    r"CIFAR-10|Cortex-M\d|STM32\w*|F4\d\d\w*|DS\d+|[Ii]nt\d+|fp\d+|NSGA-II|IEEEtran|2\{\\times\}2|3\{\\times\}3|"
    r"\$3\{\\times\}3\$|\$2\{\\times\}2\$|\$\\times\$",
    r"tabular\}\{[^}]*\}", r"\\hline", r"\\rowcolor\{[^}]*\}", r"\[t\]",
    r"\b[HV]V?\$?_[AB]\$?", r"HV\$_[AB]\$", r"\$p\$|\$\\delta\$", r"\\ell_\{95\}", r"95th", r"\bL\d\b|\bw\d",
]


def leaves(x, out):
    if isinstance(x, dict):
        for v in x.values():
            leaves(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            leaves(v, out)
    elif isinstance(x, bool):
        pass
    elif isinstance(x, (int, float)):
        if math.isfinite(x):
            out.append(float(x))
    elif isinstance(x, str):
        out.append(x)


def numbers(text: str):
    """(value, decimals_or_sigdigits, kind, raw) for every number in the text."""
    text = text.replace("{,}", "").replace("\\,", " ")
    found = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*\\times\s*10\^\{(-?\d+)\}", text):
        mant = m.group(1)
        sig = len(mant.replace(".", "").lstrip("0")) or 1
        found.append((float(mant) * 10 ** int(m.group(2)), sig, "sci", m.group(0)))
    text = re.sub(r"(\d+(?:\.\d+)?)\s*\\times\s*10\^\{(-?\d+)\}", " ", text)
    for m in re.finditer(r"10\^\{(-?\d+)\}", text):
        found.append((10.0 ** int(m.group(1)), 1, "sci", m.group(0)))
    text = re.sub(r"10\^\{(-?\d+)\}", " ", text)
    for m in re.finditer(r"(?<![A-Za-z\d.])(\d+(?:\.\d+){2})(?![\d])", text):
        found.append((m.group(1), 0, "version", m.group(1)))
    text = re.sub(r"\d+(?:\.\d+){2}", " ", text)
    for m in re.finditer(r"(?<![A-Za-z_\\])(\d+(?:\.\d+)?)", text):
        raw = m.group(1)
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        found.append((float(raw), decimals, "dec", raw))
    return found


def matches(val, digits, kind, pool, strings):
    if kind == "version":
        return any(val in s for s in strings)
    if kind == "dec" and digits == 0 and val <= 10:
        return True
    for v in pool:
        for k in SCALES:
            w = abs(v) * k
            if kind == "sci":
                if w == 0:
                    continue
                if f"{w:.{digits - 1}e}" == f"{val:.{digits - 1}e}":
                    return True
            elif round(w, digits) == round(val, digits):
                return True
    return False


def main() -> int:
    data = json.loads((ROOT / "results" / "v2" / "paper_numbers.json").read_text())
    pool, strings = [], []
    both = []
    leaves(data, both)
    pool = [v for v in both if isinstance(v, float)]
    strings = [v for v in both if isinstance(v, str)]
    bad = 0
    for name in FILES:
        for ln, line in enumerate((PAPER / name).read_text().splitlines(), 1):
            clean = line
            for pat in STRIP:
                clean = re.sub(pat, " ", clean)
            for val, digits, kind, raw in numbers(clean):
                if not matches(val, digits, kind, pool, strings):
                    bad += 1
                    print(f"{name}:{ln}: {raw}   | {line.strip()[:90]}")
    print(f"{bad} unmatched numbers" if bad else "all numbers matched")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
