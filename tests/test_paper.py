"""The paper's standing rules: every number traceable, banned words absent, references journal-only from 2020."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

BANNED = r"\b(paper|spanning|exhaustive|we|us|our|feasibility|budget|budgets|implementation|overview|fitted model|two problem formulation)\b"
FILES = ["main", "intro", "methodology", "results", "conclusion"]


def test_every_number_is_in_paper_numbers():
    import check_paper_numbers
    assert check_paper_numbers.main() == 0


def test_banned_words_absent_and_patterns_within_limits():
    for name in FILES:
        text = (ROOT / "paper" / f"{name}.tex").read_text()
        body = re.sub(r"(?<!\\)%.*$", "", text, flags=re.M)
        assert not re.findall(BANNED, body, flags=re.I), name
        assert body.count("rather than") <= 2
        assert body.count(", so") <= 6, name
        assert " --- " not in body and " -- " not in body, name


def test_references_are_journal_articles_from_2020_and_all_cited():
    bib = (ROOT / "paper" / "refs.bib").read_text()
    entries = re.findall(r"@(\w+)\{([^,]+),(.*?)\n\}\n", bib, re.S)
    cited = set()
    for name in FILES:
        for m in re.findall(r"\\cite\{([^}]*)\}", (ROOT / "paper" / f"{name}.tex").read_text()):
            cited |= set(m.replace(" ", "").split(","))
    assert all(t == "article" for t, _, _ in entries)
    assert all(int(re.search(r"year\s*=\s*\{(\d+)", b).group(1)) >= 2020 for _, _, b in entries)
    assert {k for _, k, _ in entries} == cited
