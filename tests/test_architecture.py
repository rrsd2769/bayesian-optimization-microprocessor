"""Interface discipline: nothing outside src/evaluator/ imports from its internal files."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INTERNAL = re.compile(r"^\s*(?:from|import)\s+src\.evaluator\.(\w+)", re.M)
# tests that exercise an internal seam directly, by design
SEAM_TESTS = {"test_cost_models.py", "test_rtos_sim.py", "test_calibrate.py", "test_cloud.py"}


def offenders():
    out = []
    for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").rglob("*.py")) + list((ROOT / "config").rglob("*.py")):
        rel = p.relative_to(ROOT)
        if rel.parts[:2] == ("src", "evaluator") or p.name in SEAM_TESTS or p.name == "test_architecture.py":
            continue
        for m in INTERNAL.finditer(p.read_text()):
            out.append(f"{rel}: src.evaluator.{m.group(1)}")
    return out


def test_no_file_outside_the_evaluator_imports_its_internals():
    assert offenders() == []
